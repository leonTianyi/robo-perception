// Model-agnostic TensorRT 10 execution backend.
//
// Loads a serialized engine, allocates device + host staging buffers for every
// I/O tensor, and runs inference via the name-based enqueueV3 API. It knows
// nothing about RF-DETR or detection — interpreting outputs (which tensor is
// "logits" vs "boxes") is the adapter's job. The Python twin lives in
// python/roboperc/runtime/tensorrt.py and mirrors this behaviour.
#pragma once

#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <NvInfer.h>
#include <cuda_runtime_api.h>

namespace roboperc::rt {

// ── error checking ───────────────────────────────────────────────────────────
#define ROBOPERC_CUDA_CHECK(call)                                              \
    do {                                                                       \
        cudaError_t _e = (call);                                              \
        if (_e != cudaSuccess) {                                              \
            std::cerr << "CUDA error " << cudaGetErrorString(_e) << " at "    \
                      << __FILE__ << ":" << __LINE__ << std::endl;            \
            std::abort();                                                      \
        }                                                                      \
    } while (0)

// ── logger ───────────────────────────────────────────────────────────────────
class Logger : public nvinfer1::ILogger {
public:
    explicit Logger(Severity max = Severity::kWARNING) : max_(max) {}
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= max_) std::cerr << "[TRT] " << msg << std::endl;
    }
private:
    Severity max_;
};

// ── one I/O tensor's buffers ──────────────────────────────────────────────────
struct Binding {
    std::string name;
    nvinfer1::Dims dims{};
    nvinfer1::DataType dtype{};
    std::size_t nbytes = 0;
    bool is_input = false;
    void* device = nullptr;
    std::vector<char> host;   // host staging buffer
};

// ── the engine runner ─────────────────────────────────────────────────────────
class TensorRTRunner {
public:
    explicit TensorRTRunner(
        const std::string& engine_path,
        nvinfer1::ILogger::Severity sev = nvinfer1::ILogger::Severity::kWARNING);
    ~TensorRTRunner();

    TensorRTRunner(const TensorRTRunner&) = delete;
    TensorRTRunner& operator=(const TensorRTRunner&) = delete;

    // Buffers / introspection.
    const Binding& input() const { return bindings_[input_idx_]; }
    const std::vector<Binding>& bindings() const { return bindings_; }
    cudaStream_t stream() const { return stream_; }

    // Copy a host CHW float buffer into the input device buffer (one-time setup
    // for device-scope timing).
    void load_input(const float* data);

    // Execute on resident device buffers only (device-scope / pure GPU compute).
    void execute_device_only();

    // Copy every output device->host (after execute_device_only).
    void fetch_outputs();

    // Full pipeline: H2D + execute + D2H of all outputs (end-to-end scope).
    void infer_end_to_end(const float* input);

    void sync();

private:
    template <class T> struct Deleter { void operator()(T* p) const { delete p; } };

    Logger logger_;
    std::unique_ptr<nvinfer1::IRuntime, Deleter<nvinfer1::IRuntime>> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine, Deleter<nvinfer1::ICudaEngine>> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext,
                    Deleter<nvinfer1::IExecutionContext>> context_;
    cudaStream_t stream_ = nullptr;
    std::vector<Binding> bindings_;
    int input_idx_ = 0;
};

// Volume of a Dims (product of extents).
std::size_t volume(const nvinfer1::Dims& d);
// Bytes per element of a TRT datatype.
std::size_t dtype_size(nvinfer1::DataType t);

}  // namespace roboperc::rt
