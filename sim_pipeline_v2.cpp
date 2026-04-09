#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cmath>
#include <csignal>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

class TrtLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TensorRT] " << msg << '\n';
        }
    }
};

static TrtLogger gLogger;
static std::atomic<bool> g_stop{false};

#define CUDA_CHECK(call)                                                                              \
    do {                                                                                              \
        cudaError_t err = (call);                                                                     \
        if (err != cudaSuccess) {                                                                     \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err));       \
        }                                                                                             \
    } while (0)

static size_t dtypeSize(nvinfer1::DataType t) {
    switch (t) {
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF: return 2;
        case nvinfer1::DataType::kINT32: return 4;
        case nvinfer1::DataType::kINT8: return 1;
#if NV_TENSORRT_MAJOR >= 10
        case nvinfer1::DataType::kINT64: return 8;
        case nvinfer1::DataType::kBOOL: return 1;
        case nvinfer1::DataType::kUINT8: return 1;
#endif
        default: throw std::runtime_error("Unsupported TensorRT dtype");
    }
}

static int64_t volume(const nvinfer1::Dims& dims) {
    int64_t v = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
        v *= dims.d[i];
    }
    return v;
}

static void warmupCudaDevice(int device_id) {
    CUDA_CHECK(cudaSetDevice(device_id));
    // Force CUDA context creation and runtime initialization early.
    CUDA_CHECK(cudaFree(nullptr));
    void* p = nullptr;
    CUDA_CHECK(cudaMalloc(&p, 1));
    CUDA_CHECK(cudaMemset(p, 0, 1));
    CUDA_CHECK(cudaFree(p));
}

template <typename T>
class BoundedQueue {
public:
    explicit BoundedQueue(size_t capacity) : cap_(std::max<size_t>(1, capacity)) {}

    bool put(const T& item, std::atomic<bool>& stop) {
        std::unique_lock<std::mutex> lk(m_);
        cv_not_full_.wait(lk, [&]() { return stop.load() || q_.size() < cap_; });
        if (stop.load()) return false;
        q_.push_back(item);
        cv_not_empty_.notify_one();
        return true;
    }

    bool get(T& out, std::atomic<bool>& stop, int timeout_ms) {
        std::unique_lock<std::mutex> lk(m_);
        if (!cv_not_empty_.wait_for(lk, std::chrono::milliseconds(timeout_ms), [&]() { return stop.load() || !q_.empty(); })) {
            return false;
        }
        if (q_.empty()) return false;
        out = q_.front();
        q_.pop_front();
        cv_not_full_.notify_one();
        return true;
    }

    size_t size() const {
        std::lock_guard<std::mutex> lk(m_);
        return q_.size();
    }

    void notifyAll() {
        cv_not_full_.notify_all();
        cv_not_empty_.notify_all();
    }

private:
    size_t cap_;
    mutable std::mutex m_;
    std::condition_variable cv_not_full_;
    std::condition_variable cv_not_empty_;
    std::deque<T> q_;
};

struct LogRecord {
    int det_buffer_size;
    int seg_buffer_size;
};

static std::vector<std::string> split(const std::string& s, char delim) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, delim)) {
        if (!tok.empty()) out.push_back(tok);
    }
    return out;
}

static std::unordered_map<std::string, std::string> parseArgs(int argc, char** argv) {
    std::unordered_map<std::string, std::string> args;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        if (k.rfind("--", 0) == 0) {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for argument: " + k);
            }
            args[k] = argv[++i];
        }
    }
    return args;
}

static std::string getArg(const std::unordered_map<std::string, std::string>& args, const std::string& key, const std::string& dflt = "") {
    auto it = args.find(key);
    return (it == args.end()) ? dflt : it->second;
}

static int getArgInt(const std::unordered_map<std::string, std::string>& args, const std::string& key, int dflt) {
    auto it = args.find(key);
    return (it == args.end()) ? dflt : std::stoi(it->second);
}

static double getArgDouble(const std::unordered_map<std::string, std::string>& args, const std::string& key, double dflt) {
    auto it = args.find(key);
    return (it == args.end()) ? dflt : std::stod(it->second);
}

static std::vector<std::filesystem::path> listImages(const std::filesystem::path& folder) {
    if (!std::filesystem::exists(folder) || !std::filesystem::is_directory(folder)) {
        throw std::runtime_error("Invalid image folder: " + folder.string());
    }

    std::vector<std::filesystem::path> out;
    for (const auto& e : std::filesystem::directory_iterator(folder)) {
        if (!e.is_regular_file()) continue;
        auto ext = e.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp" || ext == ".tif" || ext == ".tiff" || ext == ".webp") {
            out.push_back(e.path());
        }
    }
    std::sort(out.begin(), out.end());
    if (out.empty()) {
        throw std::runtime_error("No images found in folder: " + folder.string());
    }
    return out;
}

struct LetterboxInfo {
    float ratio;
    int top;
    int left;
};

static cv::Mat letterbox(const cv::Mat& image, int target_h, int target_w, LetterboxInfo& info, int pad_value = 0) {
    const int src_h = image.rows;
    const int src_w = image.cols;
    const float r = std::min(static_cast<float>(target_h) / static_cast<float>(src_h), static_cast<float>(target_w) / static_cast<float>(src_w));

    const int nh = static_cast<int>(std::round(src_h * r));
    const int nw = static_cast<int>(std::round(src_w * r));

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(nw, nh), 0, 0, cv::INTER_LINEAR);

    const int top = (target_h - nh) / 2;
    const int bottom = target_h - nh - top;
    const int left = (target_w - nw) / 2;
    const int right = target_w - nw - left;

    cv::Mat out;
    cv::copyMakeBorder(resized, out, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(pad_value, pad_value, pad_value));

    info.ratio = r;
    info.top = top;
    info.left = left;
    return out;
}

static void bgrToBlobCHW(const cv::Mat& bgr, std::vector<float>& blob) {
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    cv::Mat f;
    rgb.convertTo(f, CV_32F, 1.0 / 255.0);

    const int c = 3;
    const int h = f.rows;
    const int w = f.cols;
    blob.resize(static_cast<size_t>(c * h * w));

    std::vector<cv::Mat> chw(c);
    for (int i = 0; i < c; ++i) {
        chw[i] = cv::Mat(h, w, CV_32F, blob.data() + static_cast<size_t>(i) * h * w);
    }
    cv::split(f, chw);
}

class TrtRunner {
public:
    TrtRunner(const std::string& engine_path, int device_id) : device_id_(device_id) {
        CUDA_CHECK(cudaSetDevice(device_id_));

        std::ifstream ifs(engine_path, std::ios::binary);
        if (!ifs) throw std::runtime_error("Failed to open engine file: " + engine_path);
        ifs.seekg(0, std::ios::end);
        const size_t sz = static_cast<size_t>(ifs.tellg());
        ifs.seekg(0, std::ios::beg);
        std::vector<char> plan(sz);
        ifs.read(plan.data(), static_cast<std::streamsize>(sz));

        runtime_.reset(nvinfer1::createInferRuntime(gLogger));
        if (!runtime_) throw std::runtime_error("createInferRuntime failed");

        engine_.reset(runtime_->deserializeCudaEngine(plan.data(), plan.size()));
        if (!engine_) throw std::runtime_error("deserializeCudaEngine failed");

        context_.reset(engine_->createExecutionContext());
        if (!context_) throw std::runtime_error("createExecutionContext failed");

#if NV_TENSORRT_MAJOR >= 10
        for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
            const char* n = engine_->getIOTensorName(i);
            if (engine_->getTensorIOMode(n) == nvinfer1::TensorIOMode::kINPUT) {
                input_name_ = n;
            } else {
                output_names_.push_back(n);
            }
        }
#else
        for (int i = 0; i < engine_->getNbBindings(); ++i) {
            const char* n = engine_->getBindingName(i);
            if (engine_->bindingIsInput(i)) input_name_ = n;
            else output_names_.push_back(n);
        }
#endif
        if (input_name_.empty()) throw std::runtime_error("Input tensor not found");

#if NV_TENSORRT_MAJOR >= 10
        nvinfer1::Dims in = engine_->getTensorShape(input_name_.c_str());
#else
        const int input_idx = engine_->getBindingIndex(input_name_.c_str());
        nvinfer1::Dims in = engine_->getBindingDimensions(input_idx);
#endif

        if (in.nbDims != 4) throw std::runtime_error("Expected NCHW input dims");

        dynamic_ = (in.d[0] < 0);
        input_c_ = in.d[1] > 0 ? in.d[1] : 3;
        input_h_ = in.d[2] > 0 ? in.d[2] : 640;
        input_w_ = in.d[3] > 0 ? in.d[3] : 640;
        max_batch_ = in.d[0] > 0 ? in.d[0] : 1;

        if (dynamic_) {
            nvinfer1::Dims4 max_dims{max_batch_, input_c_, input_h_, input_w_};
#if NV_TENSORRT_MAJOR >= 10
            if (!context_->setInputShape(input_name_.c_str(), max_dims)) {
                throw std::runtime_error("setInputShape failed");
            }
#else
            const int input_idx = engine_->getBindingIndex(input_name_.c_str());
            if (!context_->setBindingDimensions(input_idx, max_dims)) {
                throw std::runtime_error("setBindingDimensions failed");
            }
#endif
        }

        CUDA_CHECK(cudaStreamCreate(&stream_));
        allocateBuffersForCurrentShape();
    }

    ~TrtRunner() {
        for (void* p : device_bufs_) {
            if (p) cudaFree(p);
        }
        if (stream_) cudaStreamDestroy(stream_);
    }

    int maxBatch() const { return std::max(1, max_batch_); }

    void infer(const std::vector<cv::Mat>& images) {
        if (images.empty()) return;
        const int bs = static_cast<int>(images.size());

        CUDA_CHECK(cudaSetDevice(device_id_));

        if (dynamic_) {
            nvinfer1::Dims4 run_dims{bs, input_c_, input_h_, input_w_};
#if NV_TENSORRT_MAJOR >= 10
            if (!context_->setInputShape(input_name_.c_str(), run_dims)) {
                throw std::runtime_error("setInputShape failed in infer");
            }
#else
            const int input_idx = engine_->getBindingIndex(input_name_.c_str());
            if (!context_->setBindingDimensions(input_idx, run_dims)) {
                throw std::runtime_error("setBindingDimensions failed in infer");
            }
#endif
        }

        const size_t elems = static_cast<size_t>(bs) * input_c_ * input_h_ * input_w_;
        std::vector<float> host(elems);

        for (int i = 0; i < bs; ++i) {
            LetterboxInfo lb{};
            cv::Mat prep = letterbox(images[static_cast<size_t>(i)], input_h_, input_w_, lb, 0);
            std::vector<float> one;
            bgrToBlobCHW(prep, one);
            std::copy(one.begin(), one.end(), host.begin() + static_cast<size_t>(i) * input_c_ * input_h_ * input_w_);
        }

        if (input_dtype_ == nvinfer1::DataType::kFLOAT) {
            CUDA_CHECK(cudaMemcpyAsync(input_dev_, host.data(), elems * sizeof(float), cudaMemcpyHostToDevice, stream_));
        } else if (input_dtype_ == nvinfer1::DataType::kHALF) {
            std::vector<__half> h(elems);
            for (size_t i = 0; i < elems; ++i) h[i] = __float2half(host[i]);
            CUDA_CHECK(cudaMemcpyAsync(input_dev_, h.data(), elems * sizeof(__half), cudaMemcpyHostToDevice, stream_));
        } else {
            throw std::runtime_error("Unsupported input dtype");
        }

#if NV_TENSORRT_MAJOR >= 10
        context_->setTensorAddress(input_name_.c_str(), input_dev_);
        for (size_t i = 0; i < output_names_.size(); ++i) {
            context_->setTensorAddress(output_names_[i].c_str(), output_devs_[i]);
        }
        if (!context_->enqueueV3(stream_)) {
            throw std::runtime_error("enqueueV3 failed");
        }
#else
        std::vector<void*> bindings(static_cast<size_t>(engine_->getNbBindings()), nullptr);
        bindings[static_cast<size_t>(engine_->getBindingIndex(input_name_.c_str()))] = input_dev_;
        for (size_t i = 0; i < output_names_.size(); ++i) {
            bindings[static_cast<size_t>(engine_->getBindingIndex(output_names_[i].c_str()))] = output_devs_[i];
        }
        if (!context_->enqueueV2(bindings.data(), stream_, nullptr)) {
            throw std::runtime_error("enqueueV2 failed");
        }
#endif

        CUDA_CHECK(cudaStreamSynchronize(stream_));
    }

private:
    void allocateBuffersForCurrentShape() {
#if NV_TENSORRT_MAJOR >= 10
        const nvinfer1::Dims in_dims = context_->getTensorShape(input_name_.c_str());
        input_dtype_ = engine_->getTensorDataType(input_name_.c_str());
#else
        const int input_idx = engine_->getBindingIndex(input_name_.c_str());
        const nvinfer1::Dims in_dims = context_->getBindingDimensions(input_idx);
        input_dtype_ = engine_->getBindingDataType(input_idx);
#endif

        const size_t in_bytes = static_cast<size_t>(volume(in_dims)) * dtypeSize(input_dtype_);
        CUDA_CHECK(cudaMalloc(&input_dev_, in_bytes));
        device_bufs_.push_back(input_dev_);

        for (const auto& out_name : output_names_) {
#if NV_TENSORRT_MAJOR >= 10
            nvinfer1::Dims d = context_->getTensorShape(out_name.c_str());
            nvinfer1::DataType t = engine_->getTensorDataType(out_name.c_str());
#else
            int out_idx = engine_->getBindingIndex(out_name.c_str());
            nvinfer1::Dims d = context_->getBindingDimensions(out_idx);
            nvinfer1::DataType t = engine_->getBindingDataType(out_idx);
#endif
            const size_t out_bytes = static_cast<size_t>(volume(d)) * dtypeSize(t);
            void* p = nullptr;
            CUDA_CHECK(cudaMalloc(&p, out_bytes));
            output_devs_.push_back(p);
            device_bufs_.push_back(p);
        }
    }

private:
    int device_id_{0};
    bool dynamic_{false};
    int input_c_{3};
    int input_h_{640};
    int input_w_{640};
    int max_batch_{1};

    cudaStream_t stream_{nullptr};

    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;

    std::string input_name_;
    std::vector<std::string> output_names_;

    nvinfer1::DataType input_dtype_{nvinfer1::DataType::kFLOAT};

    void* input_dev_{nullptr};
    std::vector<void*> output_devs_;
    std::vector<void*> device_bufs_;
};

struct Args {
    std::string det_folder;
    std::string seg_folder;
    std::string det_engine;
    std::string seg_engine;
    int det_batch_size{1};
    int seg_batch_size{1};
    double det_source_fps{10.0};
    double seg_source_fps{10.0};
    int det_buffer_capacity{64};
    int seg_buffer_capacity{64};
    int log_buffer_capacity{256};
    std::string log_jsonl{"output/sim_v2_log.jsonl"};
    int num_det_workers{6};
    int num_seg_workers{6};
    std::vector<int> gpu_ids{0, 1, 2, 3, 4, 5};
};

static Args parseConfig(int argc, char** argv) {
    auto kv = parseArgs(argc, argv);
    Args a;

    a.det_folder = getArg(kv, "--det-folder");
    a.seg_folder = getArg(kv, "--seg-folder");
    a.det_engine = getArg(kv, "--det-engine");
    a.seg_engine = getArg(kv, "--seg-engine");

    if (a.det_folder.empty() || a.seg_folder.empty() || a.det_engine.empty() || a.seg_engine.empty()) {
        throw std::runtime_error("Required args: --det-folder --seg-folder --det-engine --seg-engine");
    }

    a.det_batch_size = std::max(1, getArgInt(kv, "--det-batch-size", 1));
    a.seg_batch_size = std::max(1, getArgInt(kv, "--seg-batch-size", 1));
    a.det_source_fps = std::max(0.0, getArgDouble(kv, "--det-source-fps", 10.0));
    a.seg_source_fps = std::max(0.0, getArgDouble(kv, "--seg-source-fps", 10.0));
    a.det_buffer_capacity = std::max(1, getArgInt(kv, "--det-buffer-capacity", 64));
    a.seg_buffer_capacity = std::max(1, getArgInt(kv, "--seg-buffer-capacity", 64));
    a.log_buffer_capacity = std::max(1, getArgInt(kv, "--log-buffer-capacity", 256));
    a.log_jsonl = getArg(kv, "--log-jsonl", "output/sim_v2_log.jsonl");
    a.num_det_workers = std::max(1, getArgInt(kv, "--num-det-workers", 6));
    a.num_seg_workers = std::max(1, getArgInt(kv, "--num-seg-workers", 6));

    const std::string gpu_ids_s = getArg(kv, "--gpu-ids", "0,1,2,3,4,5");
    a.gpu_ids.clear();
    for (const auto& p : split(gpu_ids_s, ',')) {
        a.gpu_ids.push_back(std::stoi(p));
    }
    if (a.gpu_ids.empty()) throw std::runtime_error("--gpu-ids cannot be empty");

    return a;
}

static void signalHandler(int) {
    g_stop.store(true);
}

int main(int argc, char** argv) {
    try {
        if (!initLibNvInferPlugins(&gLogger, "")) {
            throw std::runtime_error("initLibNvInferPlugins failed");
        }

        std::signal(SIGINT, signalHandler);
        std::signal(SIGTERM, signalHandler);

        Args args = parseConfig(argc, argv);
        for (int gpu_id : args.gpu_ids) {
            warmupCudaDevice(gpu_id);
        }

        const auto det_seed = listImages(args.det_folder);
        const auto seg_seed = listImages(args.seg_folder);

        BoundedQueue<cv::Mat> det_queue(static_cast<size_t>(args.det_buffer_capacity));
        BoundedQueue<cv::Mat> seg_queue(static_cast<size_t>(args.seg_buffer_capacity));
        BoundedQueue<LogRecord> log_queue(static_cast<size_t>(args.log_buffer_capacity));

        std::mutex det_pop_lock;
        std::mutex seg_pop_lock;

        auto source_fn = [&](const std::vector<std::filesystem::path>& seed, BoundedQueue<cv::Mat>& q, double fps) {
            size_t idx = 0;
            const bool rate_limited = fps > 0.0;
            const double interval_s = rate_limited ? (1.0 / fps) : 0.0;
            auto next_tick = std::chrono::steady_clock::now();

            while (!g_stop.load()) {
                cv::Mat img = cv::imread(seed[idx].string());
                if (!img.empty()) {
                    if (!q.put(img, g_stop)) break;
                }
                idx = (idx + 1) % seed.size();

                if (rate_limited) {
                    next_tick += std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(interval_s));
                    std::this_thread::sleep_until(next_tick);
                }
            }
        };

        auto worker_fn = [&](const std::string& engine_path, int gpu_id, int requested_bs, BoundedQueue<cv::Mat>& q, std::mutex& pop_lock) {
            TrtRunner runner(engine_path, gpu_id);
            const int batch_size = std::max(1, std::min(requested_bs, runner.maxBatch()));

            while (!g_stop.load()) {
                std::vector<cv::Mat> batch;
                batch.reserve(static_cast<size_t>(batch_size));

                {
                    std::lock_guard<std::mutex> lk(pop_lock);
                    for (int i = 0; i < batch_size; ++i) {
                        cv::Mat img;
                        if (!q.get(img, g_stop, 200)) break;
                        batch.push_back(img);
                    }
                }

                if (batch.empty()) continue;
                runner.infer(batch);
            }
        };

        auto logger_fn = [&]() {
            const std::filesystem::path out_path(args.log_jsonl);
            if (out_path.has_parent_path()) {
                std::filesystem::create_directories(out_path.parent_path());
            }
            std::ofstream ofs(out_path, std::ios::app);
            if (!ofs) throw std::runtime_error("Failed to open log file: " + out_path.string());

            while (!g_stop.load()) {
                LogRecord r{static_cast<int>(det_queue.size()), static_cast<int>(seg_queue.size())};
                const std::string line = std::string("{\"det_buffer_size\":") + std::to_string(r.det_buffer_size) +
                                         std::string(",\"seg_buffer_size\":") + std::to_string(r.seg_buffer_size) + "}";
                std::cout << line << std::endl;
                ofs << line << '\n';
                ofs.flush();
                if (!log_queue.put(r, g_stop)) break;
                for (int i = 0; i < 10 && !g_stop.load(); ++i) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                }
            }
        };

        std::vector<std::thread> threads;
        threads.emplace_back(source_fn, det_seed, std::ref(det_queue), args.det_source_fps);
        threads.emplace_back(source_fn, seg_seed, std::ref(seg_queue), args.seg_source_fps);

        for (int i = 0; i < args.num_det_workers; ++i) {
            int gpu = args.gpu_ids[static_cast<size_t>(i) % args.gpu_ids.size()];
            threads.emplace_back(worker_fn, args.det_engine, gpu, args.det_batch_size, std::ref(det_queue), std::ref(det_pop_lock));
        }
        for (int i = 0; i < args.num_seg_workers; ++i) {
            int gpu = args.gpu_ids[static_cast<size_t>(i) % args.gpu_ids.size()];
            threads.emplace_back(worker_fn, args.seg_engine, gpu, args.seg_batch_size, std::ref(seg_queue), std::ref(seg_pop_lock));
        }

        threads.emplace_back(logger_fn);

        while (!g_stop.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }

        det_queue.notifyAll();
        seg_queue.notifyAll();
        log_queue.notifyAll();

        for (auto& t : threads) {
            if (t.joinable()) t.join();
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }

    return 0;
}
