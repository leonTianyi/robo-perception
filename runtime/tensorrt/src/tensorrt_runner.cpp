#include "roboperc/runtime/tensorrt_runner.hpp"

#include <fstream>
#include <iterator>
#include <numeric>

namespace roboperc::rt {

std::size_t volume(const nvinfer1::Dims& d) {
    return std::accumulate(d.d, d.d + d.nbDims, std::size_t{1},
                           std::multiplies<std::size_t>());
}

std::size_t dtype_size(nvinfer1::DataType t) {
    switch (t) {
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF:  return 2;
        case nvinfer1::DataType::kINT32: return 4;
        case nvinfer1::DataType::kINT8:  return 1;
        case nvinfer1::DataType::kBOOL:  return 1;
        default:                         return 4;
    }
}

TensorRTRunner::TensorRTRunner(const std::string& engine_path,
                               nvinfer1::ILogger::Severity sev)
    : logger_(sev) {
    std::ifstream f(engine_path, std::ios::binary);
    if (!f) {
        std::cerr << "TensorRTRunner: cannot open " << engine_path << "\n";
        std::abort();
    }
    std::vector<char> blob((std::istreambuf_iterator<char>(f)),
                           std::istreambuf_iterator<char>());

    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    engine_.reset(runtime_->deserializeCudaEngine(blob.data(), blob.size()));
    if (!engine_) {
        std::cerr << "TensorRTRunner: deserialize failed for " << engine_path
                  << "\n";
        std::abort();
    }
    context_.reset(engine_->createExecutionContext());
    ROBOPERC_CUDA_CHECK(cudaStreamCreate(&stream_));

    const int n = engine_->getNbIOTensors();
    bindings_.resize(n);
    for (int i = 0; i < n; ++i) {
        Binding& b = bindings_[i];
        b.name = engine_->getIOTensorName(i);
        b.dims = engine_->getTensorShape(b.name.c_str());
        b.dtype = engine_->getTensorDataType(b.name.c_str());
        b.is_input = engine_->getTensorIOMode(b.name.c_str()) ==
                     nvinfer1::TensorIOMode::kINPUT;
        b.nbytes = volume(b.dims) * dtype_size(b.dtype);
        ROBOPERC_CUDA_CHECK(cudaMalloc(&b.device, b.nbytes));
        b.host.resize(b.nbytes);
        context_->setTensorAddress(b.name.c_str(), b.device);
        if (b.is_input) input_idx_ = i;
    }
}

TensorRTRunner::~TensorRTRunner() {
    for (auto& b : bindings_)
        if (b.device) cudaFree(b.device);
    if (stream_) cudaStreamDestroy(stream_);
}

void TensorRTRunner::load_input(const float* data) {
    const Binding& in = bindings_[input_idx_];
    ROBOPERC_CUDA_CHECK(cudaMemcpyAsync(in.device, data, in.nbytes,
                                        cudaMemcpyHostToDevice, stream_));
    ROBOPERC_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void TensorRTRunner::execute_device_only() { context_->enqueueV3(stream_); }

void TensorRTRunner::fetch_outputs() {
    for (auto& b : bindings_)
        if (!b.is_input)
            ROBOPERC_CUDA_CHECK(cudaMemcpyAsync(b.host.data(), b.device, b.nbytes,
                                                cudaMemcpyDeviceToHost, stream_));
    ROBOPERC_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void TensorRTRunner::infer_end_to_end(const float* input) {
    const Binding& in = bindings_[input_idx_];
    ROBOPERC_CUDA_CHECK(cudaMemcpyAsync(in.device, input, in.nbytes,
                                        cudaMemcpyHostToDevice, stream_));
    context_->enqueueV3(stream_);
    for (auto& b : bindings_)
        if (!b.is_input)
            ROBOPERC_CUDA_CHECK(cudaMemcpyAsync(b.host.data(), b.device, b.nbytes,
                                                cudaMemcpyDeviceToHost, stream_));
    ROBOPERC_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void TensorRTRunner::sync() {
    ROBOPERC_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

}  // namespace roboperc::rt
