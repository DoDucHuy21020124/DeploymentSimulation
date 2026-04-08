#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

class TrtLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kINFO) {
            std::cout << "[TensorRT] " << msg << '\n';
        }
    }
};

static TrtLogger gLogger;

#define CUDA_CHECK(call)                                                                                   \
    do {                                                                                                   \
        cudaError_t err = (call);                                                                          \
        if (err != cudaSuccess) {                                                                          \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err));            \
        }                                                                                                  \
    } while (0)

struct LetterboxInfo {
    float ratio;
    float dw;
    float dh;
    int left;
    int top;
    int resized_w;
    int resized_h;
    int input_w;
    int input_h;
};

struct Detection {
    cv::Rect box;
    float score;
    int class_id;
};

struct TensorOutput {
    std::string name;
    nvinfer1::DataType dtype;
    nvinfer1::Dims dims;
    std::vector<int> shape;
    void* device_ptr;
    std::vector<float> host;
};

static std::vector<char> readBinaryFile(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        throw std::runtime_error("Failed to open engine file: " + path);
    }
    ifs.seekg(0, std::ios::end);
    size_t size = static_cast<size_t>(ifs.tellg());
    ifs.seekg(0, std::ios::beg);
    std::vector<char> data(size);
    ifs.read(data.data(), static_cast<std::streamsize>(size));
    return data;
}

static int64_t volume(const nvinfer1::Dims& dims) {
    int64_t v = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
        v *= dims.d[i];
    }
    return v;
}

static size_t dtypeSize(nvinfer1::DataType t) {
    switch (t) {
        case nvinfer1::DataType::kFLOAT:
            return 4;
        case nvinfer1::DataType::kHALF:
            return 2;
        case nvinfer1::DataType::kINT32:
            return 4;
        case nvinfer1::DataType::kINT8:
            return 1;
#if NV_TENSORRT_MAJOR >= 10
        case nvinfer1::DataType::kINT64:
            return 8;
        case nvinfer1::DataType::kBOOL:
            return 1;
        case nvinfer1::DataType::kUINT8:
            return 1;
#endif
        default:
            throw std::runtime_error("Unsupported TensorRT data type");
    }
}

static cv::Mat letterbox(const cv::Mat& image, int target_h, int target_w, LetterboxInfo& info, int pad_value = 0) {
    int src_h = image.rows;
    int src_w = image.cols;

    float ratio = std::min(static_cast<float>(target_h) / static_cast<float>(src_h),
                           static_cast<float>(target_w) / static_cast<float>(src_w));

    int resized_w = static_cast<int>(std::round(src_w * ratio));
    int resized_h = static_cast<int>(std::round(src_h * ratio));

    float dw = static_cast<float>(target_w - resized_w) / 2.0f;
    float dh = static_cast<float>(target_h - resized_h) / 2.0f;

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(resized_w, resized_h), 0, 0, cv::INTER_LINEAR);

    int top = static_cast<int>(std::round(dh - 0.1f));
    int bottom = static_cast<int>(std::round(dh + 0.1f));
    int left = static_cast<int>(std::round(dw - 0.1f));
    int right = static_cast<int>(std::round(dw + 0.1f));

    cv::Mat out;
    cv::copyMakeBorder(resized, out, top, bottom, left, right, cv::BORDER_CONSTANT,
                       cv::Scalar(pad_value, pad_value, pad_value));

    info.ratio = ratio;
    info.dw = dw;
    info.dh = dh;
    info.left = left;
    info.top = top;
    info.resized_w = resized_w;
    info.resized_h = resized_h;
    info.input_w = target_w;
    info.input_h = target_h;
    return out;
}

static void bgrToBlobCHW(const cv::Mat& bgr, std::vector<float>& blob, bool to_rgb) {
    cv::Mat img;
    if (to_rgb) {
        cv::cvtColor(bgr, img, cv::COLOR_BGR2RGB);
    } else {
        img = bgr;
    }

    cv::Mat float_img;
    img.convertTo(float_img, CV_32F, 1.0 / 255.0);

    const int channels = 3;
    const int height = float_img.rows;
    const int width = float_img.cols;
    blob.resize(static_cast<size_t>(channels * height * width));

    std::vector<cv::Mat> chw(channels);
    for (int c = 0; c < channels; ++c) {
        chw[c] = cv::Mat(height, width, CV_32F, blob.data() + static_cast<size_t>(c) * height * width);
    }
    cv::split(float_img, chw);
}

static std::vector<int> dimsToShape(const nvinfer1::Dims& d) {
    std::vector<int> shape;
    shape.reserve(static_cast<size_t>(d.nbDims));
    for (int i = 0; i < d.nbDims; ++i) {
        shape.push_back(d.d[i]);
    }
    return shape;
}

static float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

static cv::Rect scaleBoxToOriginal(float x1, float y1, float x2, float y2, const LetterboxInfo& lb, int orig_w, int orig_h) {
    float ox1 = (x1 - static_cast<float>(lb.left)) / lb.ratio;
    float oy1 = (y1 - static_cast<float>(lb.top)) / lb.ratio;
    float ox2 = (x2 - static_cast<float>(lb.left)) / lb.ratio;
    float oy2 = (y2 - static_cast<float>(lb.top)) / lb.ratio;

    ox1 = std::max(0.0f, std::min(ox1, static_cast<float>(orig_w - 1)));
    oy1 = std::max(0.0f, std::min(oy1, static_cast<float>(orig_h - 1)));
    ox2 = std::max(0.0f, std::min(ox2, static_cast<float>(orig_w - 1)));
    oy2 = std::max(0.0f, std::min(oy2, static_cast<float>(orig_h - 1)));

    int ix1 = static_cast<int>(std::round(ox1));
    int iy1 = static_cast<int>(std::round(oy1));
    int ix2 = static_cast<int>(std::round(ox2));
    int iy2 = static_cast<int>(std::round(oy2));

    return cv::Rect(cv::Point(ix1, iy1), cv::Point(std::max(ix1 + 1, ix2), std::max(iy1 + 1, iy2)));
}

static void maskToOriginal(const cv::Mat& m, const LetterboxInfo& lb, int orig_w, int orig_h, float mask_thr,
                           cv::Mat& out_bool) {
    cv::Mat m_input;
    cv::resize(m, m_input, cv::Size(lb.input_w, lb.input_h), 0, 0, cv::INTER_LINEAR);

    cv::Rect roi(lb.left, lb.top, lb.resized_w, lb.resized_h);
    cv::Mat m_unpad = m_input(roi);

    cv::Mat m_orig;
    cv::resize(m_unpad, m_orig, cv::Size(orig_w, orig_h), 0, 0, cv::INTER_LINEAR);

    cv::threshold(m_orig, out_bool, static_cast<double>(mask_thr), 255.0, cv::THRESH_BINARY);
    out_bool.convertTo(out_bool, CV_8U);
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <engine.trt|engine.engine> <input.jpg> <output_dir> [conf=0.5] [mask_thr=0.5] [rgb=1] [device=0]\n";
        return 1;
    }

    std::string engine_path = argv[1];
    std::string image_path = argv[2];
    std::string output_dir = argv[3];
    float conf_thres = (argc > 4) ? std::stof(argv[4]) : 0.5f;
    float mask_thres = (argc > 5) ? std::stof(argv[5]) : 0.5f;
    bool to_rgb = (argc > 6) ? (std::stoi(argv[6]) != 0) : true;
    int device_id = (argc > 7) ? std::stoi(argv[7]) : 0;

    cudaStream_t stream = nullptr;
    std::vector<void*> allocated_device;

    try {
        std::filesystem::create_directories(output_dir);
        std::filesystem::path save_path = std::filesystem::path(output_dir) / std::filesystem::path(image_path).filename();

        int device_count = 0;
        CUDA_CHECK(cudaGetDeviceCount(&device_count));
        if (device_count <= 0) {
            throw std::runtime_error("No CUDA devices found");
        }
        if (device_id < 0 || device_id >= device_count) {
            throw std::runtime_error("Invalid device_id. Available range: 0.." + std::to_string(device_count - 1));
        }
        CUDA_CHECK(cudaSetDevice(device_id));

        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
        std::cout << "Using CUDA device " << device_id << ": " << prop.name << "\n";

        cv::Mat bgr = cv::imread(image_path);
        if (bgr.empty()) {
            throw std::runtime_error("Failed to read image: " + image_path);
        }
        int orig_h = bgr.rows;
        int orig_w = bgr.cols;

        std::vector<char> plan = readBinaryFile(engine_path);
        if (!initLibNvInferPlugins(&gLogger, "")) {
            throw std::runtime_error("Failed to initialize TensorRT plugins");
        }

        std::unique_ptr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(gLogger)};
        if (!runtime) {
            throw std::runtime_error("Failed to create TensorRT runtime");
        }

        std::unique_ptr<nvinfer1::ICudaEngine> engine{runtime->deserializeCudaEngine(plan.data(), plan.size())};
        if (!engine) {
            throw std::runtime_error("Failed to deserialize engine");
        }

        std::unique_ptr<nvinfer1::IExecutionContext> context{engine->createExecutionContext()};
        if (!context) {
            throw std::runtime_error("Failed to create execution context");
        }

        std::string input_name;
        std::vector<std::string> output_names;

#if NV_TENSORRT_MAJOR >= 10
        for (int i = 0; i < engine->getNbIOTensors(); ++i) {
            const char* name = engine->getIOTensorName(i);
            if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
                input_name = name;
            } else {
                output_names.emplace_back(name);
            }
        }
#else
        for (int i = 0; i < engine->getNbBindings(); ++i) {
            const char* name = engine->getBindingName(i);
            if (engine->bindingIsInput(i)) {
                input_name = name;
            } else {
                output_names.emplace_back(name);
            }
        }
#endif

        if (input_name.empty() || output_names.size() < 2) {
            throw std::runtime_error("Expected 1 input and at least 2 outputs for segmentation");
        }

        std::sort(output_names.begin(), output_names.end());

        nvinfer1::Dims in_dims;
#if NV_TENSORRT_MAJOR >= 10
        in_dims = engine->getTensorShape(input_name.c_str());
#else
        int input_idx = engine->getBindingIndex(input_name.c_str());
        in_dims = engine->getBindingDimensions(input_idx);
#endif

        if (in_dims.nbDims != 4) {
            throw std::runtime_error("Expected input dims rank=4 (NCHW)");
        }

        int n = (in_dims.d[0] == -1) ? 1 : in_dims.d[0];
        int c = in_dims.d[1];
        int h = in_dims.d[2] > 0 ? in_dims.d[2] : 480;
        int w = in_dims.d[3] > 0 ? in_dims.d[3] : 640;

        if (n != 1 || c != 3) {
            throw std::runtime_error("This sample supports N=1, C=3 only");
        }

        LetterboxInfo lb{};
        cv::Mat prep = letterbox(bgr, h, w, lb, 0);
        std::vector<float> input_host;
        bgrToBlobCHW(prep, input_host, to_rgb);

        nvinfer1::Dims4 run_dims{1, 3, h, w};
#if NV_TENSORRT_MAJOR >= 10
        if (!context->setInputShape(input_name.c_str(), run_dims)) {
            throw std::runtime_error("setInputShape failed");
        }
#else
        int input_idx = engine->getBindingIndex(input_name.c_str());
        if (!context->setBindingDimensions(input_idx, run_dims)) {
            throw std::runtime_error("setBindingDimensions failed");
        }
#endif

        nvinfer1::DataType in_dtype;
#if NV_TENSORRT_MAJOR >= 10
        in_dtype = engine->getTensorDataType(input_name.c_str());
#else
        in_dtype = engine->getBindingDataType(engine->getBindingIndex(input_name.c_str()));
#endif
        if (!(in_dtype == nvinfer1::DataType::kFLOAT || in_dtype == nvinfer1::DataType::kHALF)) {
            throw std::runtime_error("Unsupported input dtype. Expected FP32 or FP16");
        }

        void* d_input = nullptr;
        int64_t in_elems = static_cast<int64_t>(n) * c * h * w;
        CUDA_CHECK(cudaMalloc(&d_input, static_cast<size_t>(in_elems) * dtypeSize(in_dtype)));
        allocated_device.push_back(d_input);

        CUDA_CHECK(cudaStreamCreate(&stream));

        std::vector<__half> input_half;
        if (in_dtype == nvinfer1::DataType::kFLOAT) {
            CUDA_CHECK(cudaMemcpyAsync(d_input, input_host.data(), static_cast<size_t>(in_elems) * sizeof(float),
                                       cudaMemcpyHostToDevice, stream));
        } else {
            input_half.resize(static_cast<size_t>(in_elems));
            for (int64_t i = 0; i < in_elems; ++i) {
                input_half[static_cast<size_t>(i)] = __float2half(input_host[static_cast<size_t>(i)]);
            }
            CUDA_CHECK(cudaMemcpyAsync(d_input, input_half.data(), static_cast<size_t>(in_elems) * sizeof(__half),
                                       cudaMemcpyHostToDevice, stream));
        }

        std::vector<TensorOutput> outputs;
        outputs.reserve(output_names.size());

#if NV_TENSORRT_MAJOR >= 10
        context->setTensorAddress(input_name.c_str(), d_input);

        for (const auto& name : output_names) {
            nvinfer1::Dims out_dims = context->getTensorShape(name.c_str());
            nvinfer1::DataType out_dtype = engine->getTensorDataType(name.c_str());
            if (!(out_dtype == nvinfer1::DataType::kFLOAT || out_dtype == nvinfer1::DataType::kHALF)) {
                throw std::runtime_error("Unsupported output dtype. Expected FP32 or FP16");
            }

            int64_t elems = volume(out_dims);
            if (elems <= 0) {
                throw std::runtime_error("Invalid output tensor shape for " + name);
            }

            void* d_out = nullptr;
            CUDA_CHECK(cudaMalloc(&d_out, static_cast<size_t>(elems) * dtypeSize(out_dtype)));
            allocated_device.push_back(d_out);

            context->setTensorAddress(name.c_str(), d_out);

            TensorOutput t{};
            t.name = name;
            t.dtype = out_dtype;
            t.dims = out_dims;
            t.shape = dimsToShape(out_dims);
            t.device_ptr = d_out;
            t.host.resize(static_cast<size_t>(elems));
            outputs.push_back(std::move(t));
        }

        if (!context->enqueueV3(stream)) {
            throw std::runtime_error("enqueueV3 failed");
        }
#else
        std::vector<void*> bindings(static_cast<size_t>(engine->getNbBindings()), nullptr);
        bindings[static_cast<size_t>(engine->getBindingIndex(input_name.c_str()))] = d_input;

        for (const auto& name : output_names) {
            int out_idx = engine->getBindingIndex(name.c_str());
            nvinfer1::Dims out_dims = context->getBindingDimensions(out_idx);
            nvinfer1::DataType out_dtype = engine->getBindingDataType(out_idx);
            if (!(out_dtype == nvinfer1::DataType::kFLOAT || out_dtype == nvinfer1::DataType::kHALF)) {
                throw std::runtime_error("Unsupported output dtype. Expected FP32 or FP16");
            }

            int64_t elems = volume(out_dims);
            if (elems <= 0) {
                throw std::runtime_error("Invalid output tensor shape for " + name);
            }

            void* d_out = nullptr;
            CUDA_CHECK(cudaMalloc(&d_out, static_cast<size_t>(elems) * dtypeSize(out_dtype)));
            allocated_device.push_back(d_out);

            bindings[static_cast<size_t>(out_idx)] = d_out;

            TensorOutput t{};
            t.name = name;
            t.dtype = out_dtype;
            t.dims = out_dims;
            t.shape = dimsToShape(out_dims);
            t.device_ptr = d_out;
            t.host.resize(static_cast<size_t>(elems));
            outputs.push_back(std::move(t));
        }

        if (!context->enqueueV2(bindings.data(), stream, nullptr)) {
            throw std::runtime_error("enqueueV2 failed");
        }
#endif

        for (auto& out : outputs) {
            int64_t elems = static_cast<int64_t>(out.host.size());
            if (out.dtype == nvinfer1::DataType::kFLOAT) {
                CUDA_CHECK(cudaMemcpyAsync(out.host.data(), out.device_ptr, static_cast<size_t>(elems) * sizeof(float),
                                           cudaMemcpyDeviceToHost, stream));
            } else {
                std::vector<__half> tmp(static_cast<size_t>(elems));
                CUDA_CHECK(cudaMemcpyAsync(tmp.data(), out.device_ptr, static_cast<size_t>(elems) * sizeof(__half),
                                           cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
                for (int64_t i = 0; i < elems; ++i) {
                    out.host[static_cast<size_t>(i)] = __half2float(tmp[static_cast<size_t>(i)]);
                }
            }
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));

        const TensorOutput& det_out = outputs[0];
        const TensorOutput& mask_out = outputs[1];

        std::vector<int> det_shape = det_out.shape;
        if (det_shape.size() == 3 && det_shape[0] == 1) {
            det_shape.erase(det_shape.begin());
        }
        if (det_shape.size() != 2) {
            throw std::runtime_error("Unsupported detection output shape rank: " + std::to_string(det_shape.size()));
        }

        int det_rows = det_shape[0];
        int det_cols = det_shape[1];
        if (det_cols < 7) {
            throw std::runtime_error("Detection output must contain box, score, cls and mask coeffs");
        }

        std::vector<int> keep_indices;
        std::vector<Detection> detections;
        std::vector<std::vector<float>> mask_coeffs;

        keep_indices.reserve(static_cast<size_t>(det_rows));
        detections.reserve(static_cast<size_t>(det_rows));
        mask_coeffs.reserve(static_cast<size_t>(det_rows));

        for (int i = 0; i < det_rows; ++i) {
            const float* p = det_out.host.data() + static_cast<size_t>(i) * det_cols;
            float score = p[4];
            if (score <= conf_thres) {
                continue;
            }

            Detection d{};
            d.box = scaleBoxToOriginal(p[0], p[1], p[2], p[3], lb, orig_w, orig_h);
            d.score = score;
            d.class_id = static_cast<int>(std::round(p[5]));
            detections.push_back(d);

            keep_indices.push_back(i);
            mask_coeffs.emplace_back(p + 6, p + det_cols);
        }

        std::vector<cv::Mat> masks_orig;
        masks_orig.reserve(detections.size());

        std::vector<int> mshape = mask_out.shape;
        if (mshape.size() == 4 && mshape[0] == 1) {
            mshape.erase(mshape.begin());
        }

        if (mshape.size() != 3) {
            throw std::runtime_error("Unsupported mask output shape rank: " + std::to_string(mshape.size()));
        }

        if (!detections.empty()) {
            int d0 = mshape[0];
            int mh = mshape[1];
            int mw = mshape[2];

            // raw mask per candidate: (max_det, mh, mw)
            if (d0 == det_rows) {
                for (int idx : keep_indices) {
                    const float* mptr = mask_out.host.data() + static_cast<size_t>(idx) * mh * mw;
                    cv::Mat m(mh, mw, CV_32F, const_cast<float*>(mptr));
                    cv::Mat m_copy = m.clone();
                    cv::Mat m_bool;
                    maskToOriginal(m_copy, lb, orig_w, orig_h, mask_thres, m_bool);
                    masks_orig.push_back(m_bool);
                }
            }
            // raw mask per kept detection: (num_kept, mh, mw)
            else if (d0 == static_cast<int>(detections.size())) {
                for (int i = 0; i < d0; ++i) {
                    const float* mptr = mask_out.host.data() + static_cast<size_t>(i) * mh * mw;
                    cv::Mat m(mh, mw, CV_32F, const_cast<float*>(mptr));
                    cv::Mat m_copy = m.clone();
                    cv::Mat m_bool;
                    maskToOriginal(m_copy, lb, orig_w, orig_h, mask_thres, m_bool);
                    masks_orig.push_back(m_bool);
                }
            }
            // proto-like output: (nm, mh, mw)
            else if (d0 == static_cast<int>(mask_coeffs[0].size())) {
                int nm = d0;
                for (size_t i = 0; i < detections.size(); ++i) {
                    std::vector<float> comb(static_cast<size_t>(mh * mw), 0.0f);
                    for (int cidx = 0; cidx < nm; ++cidx) {
                        float coeff = mask_coeffs[i][static_cast<size_t>(cidx)];
                        const float* proto = mask_out.host.data() + static_cast<size_t>(cidx) * mh * mw;
                        for (int j = 0; j < mh * mw; ++j) {
                            comb[static_cast<size_t>(j)] += coeff * proto[j];
                        }
                    }
                    for (float& v : comb) {
                        v = sigmoid(v);
                    }
                    cv::Mat m(mh, mw, CV_32F, comb.data());
                    cv::Mat m_copy = m.clone();
                    cv::Mat m_bool;
                    maskToOriginal(m_copy, lb, orig_w, orig_h, mask_thres, m_bool);
                    masks_orig.push_back(m_bool);
                }
            } else {
                throw std::runtime_error("Unsupported mask output shape combination");
            }
        }

        cv::Mat vis = bgr.clone();
        std::mt19937 rng(42);
        std::uniform_int_distribution<int> color_dist(64, 255);
        std::unordered_map<int, cv::Scalar> class_colors;

        for (size_t i = 0; i < detections.size(); ++i) {
            const auto& d = detections[i];
            if (class_colors.find(d.class_id) == class_colors.end()) {
                class_colors[d.class_id] = cv::Scalar(color_dist(rng), color_dist(rng), color_dist(rng));
            }
            cv::Scalar color = class_colors[d.class_id];

            if (i < masks_orig.size()) {
                cv::Mat overlay = vis.clone();
                overlay.setTo(color, masks_orig[i]);
                cv::addWeighted(vis, 0.65, overlay, 0.35, 0.0, vis);
            }

            cv::rectangle(vis, d.box, color, 2);
            std::string label = "cls " + std::to_string(d.class_id) + " " + cv::format("%.2f", d.score);
            int baseline = 0;
            cv::Size ts = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseline);
            int y_text = std::max(0, d.box.y - ts.height - 6);
            cv::rectangle(vis,
                          cv::Point(d.box.x, y_text),
                          cv::Point(d.box.x + ts.width + 8, y_text + ts.height + 8),
                          color,
                          cv::FILLED);
            cv::putText(vis,
                        label,
                        cv::Point(d.box.x + 4, y_text + ts.height + 2),
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.6,
                        cv::Scalar(255, 255, 255),
                        2,
                        cv::LINE_AA);
        }

        if (!cv::imwrite(save_path.string(), vis)) {
            throw std::runtime_error("Failed to write output image: " + save_path.string());
        }

        std::cout << "Detections: " << detections.size() << "\n";
        std::cout << "Saved visualization to: " << save_path.string() << "\n";

        if (stream) {
            CUDA_CHECK(cudaStreamDestroy(stream));
            stream = nullptr;
        }
        for (void* p : allocated_device) {
            CUDA_CHECK(cudaFree(p));
        }
        allocated_device.clear();

    } catch (const std::exception& e) {
        if (stream) {
            cudaStreamDestroy(stream);
        }
        for (void* p : allocated_device) {
            cudaFree(p);
        }
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }

    return 0;
}
