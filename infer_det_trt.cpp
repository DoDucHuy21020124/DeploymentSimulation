#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#include <filesystem>

class TrtLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kINFO) {
            std::cout << "[TensorRT] " << msg << '\n';
        }
    }
};

static TrtLogger gLogger;

#define CUDA_CHECK(call)                                                                                 \
    do {                                                                                                 \
        cudaError_t err = (call);                                                                        \
        if (err != cudaSuccess) {                                                                        \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err));          \
        }                                                                                                \
    } while (0)

struct Detection {
    cv::Rect box;
    float score;
    int class_id;
};

struct LetterboxInfo {
    float ratio;
    float dw;
    float dh;
    int new_w;
    int new_h;
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
    info.new_w = target_w;
    info.new_h = target_h;

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

static cv::Rect mapBoxToOriginal(float x1, float y1, float x2, float y2, const LetterboxInfo& lb, int src_w, int src_h) {
    float ox1 = (x1 - lb.dw) / lb.ratio;
    float oy1 = (y1 - lb.dh) / lb.ratio;
    float ox2 = (x2 - lb.dw) / lb.ratio;
    float oy2 = (y2 - lb.dh) / lb.ratio;

    ox1 = std::max(0.0f, std::min(ox1, static_cast<float>(src_w - 1)));
    oy1 = std::max(0.0f, std::min(oy1, static_cast<float>(src_h - 1)));
    ox2 = std::max(0.0f, std::min(ox2, static_cast<float>(src_w - 1)));
    oy2 = std::max(0.0f, std::min(oy2, static_cast<float>(src_h - 1)));

    int ix1 = static_cast<int>(std::round(ox1));
    int iy1 = static_cast<int>(std::round(oy1));
    int ix2 = static_cast<int>(std::round(ox2));
    int iy2 = static_cast<int>(std::round(oy2));

    return cv::Rect(cv::Point(ix1, iy1), cv::Point(std::max(ix1 + 1, ix2), std::max(iy1 + 1, iy2)));
}

static std::vector<Detection> decodeEnd2End(const float* out, int rows, int attrs, float conf_thres,
                                            const LetterboxInfo& lb, int src_w, int src_h) {
    std::vector<Detection> dets;
    dets.reserve(static_cast<size_t>(rows));

    for (int i = 0; i < rows; ++i) {
        const float* p = out + static_cast<size_t>(i) * attrs;
        float score = p[4];
        if (score <= conf_thres) {
            continue;
        }

        float x1 = p[0];
        float y1 = p[1];
        float x2 = p[2];
        float y2 = p[3];
        int cls = static_cast<int>(std::round(p[5]));

        Detection d;
        d.box = mapBoxToOriginal(x1, y1, x2, y2, lb, src_w, src_h);
        d.score = score;
        d.class_id = cls;
        dets.push_back(d);
    }

    return dets;
}

static std::vector<Detection> decodeYoloRaw(const float* out, int num_preds, int attrs, bool transposed,
                                            float conf_thres, const LetterboxInfo& lb, int src_w, int src_h) {
    std::vector<Detection> dets;
    dets.reserve(static_cast<size_t>(num_preds));

    const int num_cls = attrs - 4;
    if (num_cls <= 0) {
        return dets;
    }

    for (int i = 0; i < num_preds; ++i) {
        float cx, cy, w, h;
        if (transposed) {
            const float* p = out + static_cast<size_t>(i) * attrs;
            cx = p[0];
            cy = p[1];
            w = p[2];
            h = p[3];

            int best_cls = -1;
            float best_score = 0.0f;
            for (int c = 0; c < num_cls; ++c) {
                float s = p[4 + c];
                if (s > best_score) {
                    best_score = s;
                    best_cls = c;
                }
            }
            if (best_score < conf_thres) {
                continue;
            }

            float x1 = cx - w * 0.5f;
            float y1 = cy - h * 0.5f;
            float x2 = cx + w * 0.5f;
            float y2 = cy + h * 0.5f;

            Detection d;
            d.box = mapBoxToOriginal(x1, y1, x2, y2, lb, src_w, src_h);
            d.score = best_score;
            d.class_id = best_cls;
            dets.push_back(d);
        } else {
            cx = out[0 * num_preds + i];
            cy = out[1 * num_preds + i];
            w = out[2 * num_preds + i];
            h = out[3 * num_preds + i];

            int best_cls = -1;
            float best_score = 0.0f;
            for (int c = 0; c < num_cls; ++c) {
                float s = out[(4 + c) * num_preds + i];
                if (s > best_score) {
                    best_score = s;
                    best_cls = c;
                }
            }
            if (best_score < conf_thres) {
                continue;
            }

            float x1 = cx - w * 0.5f;
            float y1 = cy - h * 0.5f;
            float x2 = cx + w * 0.5f;
            float y2 = cy + h * 0.5f;

            Detection d;
            d.box = mapBoxToOriginal(x1, y1, x2, y2, lb, src_w, src_h);
            d.score = best_score;
            d.class_id = best_cls;
            dets.push_back(d);
        }
    }

    return dets;
}

static std::vector<Detection> postprocess(const std::vector<float>& out_data,
                                          const std::vector<int>& out_shape,
                                          float conf_thres,
                                          const LetterboxInfo& lb,
                                          int src_w,
                                          int src_h) {
    std::vector<Detection> dets;

    if (out_shape.size() == 3) {
        int d0 = out_shape[0], d1 = out_shape[1], d2 = out_shape[2];
        if (d0 != 1) {
            throw std::runtime_error("Only batch=1 is supported in this sample");
        }

        if (d2 >= 6 && d1 <= 1024) {
            dets = decodeEnd2End(out_data.data(), d1, d2, conf_thres, lb, src_w, src_h);
        } else if (d1 >= 6 && d2 > d1) {
            dets = decodeYoloRaw(out_data.data(), d2, d1, false, conf_thres, lb, src_w, src_h);
        } else if (d2 >= 6 && d1 > d2) {
            dets = decodeYoloRaw(out_data.data(), d1, d2, true, conf_thres, lb, src_w, src_h);
        } else {
            throw std::runtime_error("Unsupported output shape for YOLO decode");
        }
    } else if (out_shape.size() == 2) {
        int rows = out_shape[0], attrs = out_shape[1];
        if (attrs >= 6) {
            dets = decodeEnd2End(out_data.data(), rows, attrs, conf_thres, lb, src_w, src_h);
        } else {
            throw std::runtime_error("Unsupported 2D output shape");
        }
    } else {
        throw std::runtime_error("Unsupported output rank");
    }

    return dets;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <engine.trt|engine.engine> <input.jpg> <output_dir> [conf=0.25] [rgb=0] [device=0]\n";
        return 1;
    }

    std::string engine_path = argv[1];
    std::string image_path = argv[2];
    std::string output_dir = argv[3];
    float conf_thres = (argc > 4) ? std::stof(argv[4]) : 0.25f;
    bool to_rgb = (argc > 5) ? (std::stoi(argv[5]) != 0) : false;
    int device_id = (argc > 6) ? std::stoi(argv[6]) : 0;

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
        std::string output_name;
#if NV_TENSORRT_MAJOR >= 10
        for (int i = 0; i < engine->getNbIOTensors(); ++i) {
            const char* name = engine->getIOTensorName(i);
            if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
                input_name = name;
            } else {
                output_name = name;
            }
        }
#else
        for (int i = 0; i < engine->getNbBindings(); ++i) {
            const char* name = engine->getBindingName(i);
            if (engine->bindingIsInput(i)) {
                input_name = name;
            } else {
                output_name = name;
            }
        }
#endif

        if (input_name.empty() || output_name.empty()) {
            throw std::runtime_error("Failed to find input/output tensors");
        }

        nvinfer1::Dims in_dims;
#if NV_TENSORRT_MAJOR >= 10
        in_dims = engine->getTensorShape(input_name.c_str());
#else
        int input_idx = engine->getBindingIndex(input_name.c_str());
        in_dims = engine->getBindingDimensions(input_idx);
#endif

        // Expect NCHW input.
        if (in_dims.nbDims != 4) {
            throw std::runtime_error("Expected input dims rank=4 (NCHW)");
        }

        int n = (in_dims.d[0] == -1) ? 1 : in_dims.d[0];
        int c = in_dims.d[1];
        int h = in_dims.d[2];
        int w = in_dims.d[3];

        if (n != 1 || c != 3) {
            throw std::runtime_error("This sample supports N=1, C=3 only");
        }

        if (h <= 0 || w <= 0) {
            // Fallback if dynamic shape has -1 placeholders.
            h = 640;
            w = 640;
        }

        LetterboxInfo lb{};
        cv::Mat lb_img = letterbox(bgr, h, w, lb, 0);

        std::vector<float> input_host;
        bgrToBlobCHW(lb_img, input_host, to_rgb);

        nvinfer1::Dims4 run_dims{1, 3, h, w};
#if NV_TENSORRT_MAJOR >= 10
        if (!context->setInputShape(input_name.c_str(), run_dims)) {
            throw std::runtime_error("setInputShape failed");
        }
        nvinfer1::Dims out_dims = context->getTensorShape(output_name.c_str());
        nvinfer1::DataType in_dtype = engine->getTensorDataType(input_name.c_str());
        nvinfer1::DataType out_dtype = engine->getTensorDataType(output_name.c_str());
#else
        int input_idx = engine->getBindingIndex(input_name.c_str());
        int output_idx = engine->getBindingIndex(output_name.c_str());
        if (!context->setBindingDimensions(input_idx, run_dims)) {
            throw std::runtime_error("setBindingDimensions failed");
        }
        nvinfer1::Dims out_dims = context->getBindingDimensions(output_idx);
        nvinfer1::DataType in_dtype = engine->getBindingDataType(input_idx);
        nvinfer1::DataType out_dtype = engine->getBindingDataType(output_idx);
#endif

        if (!(in_dtype == nvinfer1::DataType::kFLOAT || in_dtype == nvinfer1::DataType::kHALF)) {
            throw std::runtime_error("Unsupported input dtype. Expected FP32 or FP16");
        }
        if (!(out_dtype == nvinfer1::DataType::kFLOAT || out_dtype == nvinfer1::DataType::kHALF)) {
            throw std::runtime_error("Unsupported output dtype. Expected FP32 or FP16");
        }

        int64_t in_elems = static_cast<int64_t>(n) * c * h * w;
        int64_t out_elems = volume(out_dims);
        if (out_elems <= 0) {
            throw std::runtime_error("Invalid output tensor shape");
        }

        void* d_input = nullptr;
        void* d_output = nullptr;
        CUDA_CHECK(cudaMalloc(&d_input, static_cast<size_t>(in_elems) * dtypeSize(in_dtype)));
        CUDA_CHECK(cudaMalloc(&d_output, static_cast<size_t>(out_elems) * dtypeSize(out_dtype)));

        cudaStream_t stream;
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

#if NV_TENSORRT_MAJOR >= 10
        context->setTensorAddress(input_name.c_str(), d_input);
        context->setTensorAddress(output_name.c_str(), d_output);
        if (!context->enqueueV3(stream)) {
            throw std::runtime_error("enqueueV3 failed");
        }
#else
        std::vector<void*> bindings(static_cast<size_t>(engine->getNbBindings()), nullptr);
        bindings[static_cast<size_t>(engine->getBindingIndex(input_name.c_str()))] = d_input;
        bindings[static_cast<size_t>(engine->getBindingIndex(output_name.c_str()))] = d_output;
        if (!context->enqueueV2(bindings.data(), stream, nullptr)) {
            throw std::runtime_error("enqueueV2 failed");
        }
#endif

        std::vector<float> output_host(static_cast<size_t>(out_elems));
        if (out_dtype == nvinfer1::DataType::kFLOAT) {
            CUDA_CHECK(cudaMemcpyAsync(output_host.data(), d_output, static_cast<size_t>(out_elems) * sizeof(float),
                                       cudaMemcpyDeviceToHost, stream));
        } else {
            std::vector<__half> output_half(static_cast<size_t>(out_elems));
            CUDA_CHECK(cudaMemcpyAsync(output_half.data(), d_output, static_cast<size_t>(out_elems) * sizeof(__half),
                                       cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            for (int64_t i = 0; i < out_elems; ++i) {
                output_host[static_cast<size_t>(i)] = __half2float(output_half[static_cast<size_t>(i)]);
            }
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));

        std::vector<int> out_shape;
        out_shape.reserve(static_cast<size_t>(out_dims.nbDims));
        for (int i = 0; i < out_dims.nbDims; ++i) {
            out_shape.push_back(out_dims.d[i]);
        }

        std::vector<Detection> final_dets = postprocess(output_host, out_shape, conf_thres, lb, bgr.cols, bgr.rows);

        for (const auto& d : final_dets) {
            cv::rectangle(bgr, d.box, cv::Scalar(0, 255, 255), 2);
            std::string label = "cls=" + std::to_string(d.class_id) + " " + cv::format("%.2f", d.score);
            int baseline = 0;
            cv::Size ts = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
            int ty = std::max(d.box.y, ts.height + 2);
            cv::rectangle(bgr, cv::Point(d.box.x, ty - ts.height - 2),
                          cv::Point(d.box.x + ts.width, ty + baseline - 2), cv::Scalar(0, 255, 255), cv::FILLED);
            cv::putText(bgr, label, cv::Point(d.box.x, ty - 2), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                        cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
        }

        if (!cv::imwrite(save_path.string(), bgr)) {
            throw std::runtime_error("Failed to write output image: " + save_path.string());
        }

        std::cout << "Saved result to: " << save_path.string() << "\n";
        std::cout << "Detections: " << final_dets.size() << "\n";

        CUDA_CHECK(cudaStreamDestroy(stream));
        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_output));

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }

    return 0;
}
