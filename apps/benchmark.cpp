// benchmark — time a TensorRT engine from C++ at two scopes, write report JSON.
//
//   benchmark <engine> <image> <out.json> [warmup=200] [iters=1000]
//
//   device-scope : CUDA-event timed enqueueV3 on resident buffers (pure compute)
//   e2e-scope    : wall-clock around H2D + execute + D2H
//
// Output JSON is consumed by python/roboperc/eval/report.py. This measures the
// runtime, so it talks to TensorRTRunner directly rather than via the Detector.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>

#include "roboperc/detection/rfdetr/preprocess.hpp"
#include "roboperc/io/decode.hpp"
#include "roboperc/runtime/tensorrt_runner.hpp"

using namespace roboperc;
using roboperc::rt::TensorRTRunner;

namespace {

struct Stats {
    double mean, p50, p90, p99, stdev, mn, mx;
    int n;
};

Stats compute_stats(std::vector<double> v) {
    Stats s{};
    s.n = (int)v.size();
    std::sort(v.begin(), v.end());
    double sum = 0;
    for (double x : v) sum += x;
    s.mean = sum / v.size();
    double var = 0;
    for (double x : v) var += (x - s.mean) * (x - s.mean);
    s.stdev = std::sqrt(var / v.size());
    auto pct = [&](double p) {
        double idx = p / 100.0 * (v.size() - 1);
        size_t lo = (size_t)std::floor(idx), hi = (size_t)std::ceil(idx);
        double frac = idx - lo;
        return v[lo] * (1 - frac) + v[hi] * frac;
    };
    s.p50 = pct(50); s.p90 = pct(90); s.p99 = pct(99);
    s.mn = v.front(); s.mx = v.back();
    return s;
}

void write_stats_json(std::ofstream& f, const std::string& key, const Stats& s) {
    f << "  \"" << key << "\": {\n"
      << "    \"mean_ms\": " << s.mean << ",\n"
      << "    \"p50_ms\": "  << s.p50  << ",\n"
      << "    \"p90_ms\": "  << s.p90  << ",\n"
      << "    \"p99_ms\": "  << s.p99  << ",\n"
      << "    \"std_ms\": "  << s.stdev << ",\n"
      << "    \"min_ms\": "  << s.mn   << ",\n"
      << "    \"max_ms\": "  << s.mx   << ",\n"
      << "    \"n\": " << s.n << "\n"
      << "  }";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "usage: " << argv[0]
                  << " <engine> <image> <out.json> [warmup] [iters]\n";
        return 1;
    }
    const std::string engine_path = argv[1];
    const std::string image_path = argv[2];
    const std::string out_json = argv[3];
    const int warmup = argc > 4 ? std::stoi(argv[4]) : 200;
    const int iters = argc > 5 ? std::stoi(argv[5]) : 1000;

    cv::Mat img = io::decode_image(image_path);

    TensorRTRunner runner(engine_path);
    detection::rfdetr::PreprocCfg pc;
    const auto& in = runner.input();
    pc.input_h = in.dims.d[in.dims.nbDims - 2];
    pc.input_w = in.dims.d[in.dims.nbDims - 1];
    std::vector<float> input = detection::rfdetr::preprocess(img, pc);

    // ── device-scope: CUDA-event timed enqueueV3 on resident buffers ──────────
    runner.load_input(input.data());
    cudaEvent_t start, stop;
    ROBOPERC_CUDA_CHECK(cudaEventCreate(&start));
    ROBOPERC_CUDA_CHECK(cudaEventCreate(&stop));

    std::cout << "[cpp] device-scope warmup " << warmup << "\n";
    for (int i = 0; i < warmup; ++i) runner.execute_device_only();
    runner.sync();

    std::cout << "[cpp] device-scope timing " << iters << "\n";
    std::vector<double> dev_ms;
    dev_ms.reserve(iters);
    for (int i = 0; i < iters; ++i) {
        ROBOPERC_CUDA_CHECK(cudaEventRecord(start, runner.stream()));
        runner.execute_device_only();
        ROBOPERC_CUDA_CHECK(cudaEventRecord(stop, runner.stream()));
        ROBOPERC_CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0;
        ROBOPERC_CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        dev_ms.push_back(ms);
    }
    Stats dev = compute_stats(dev_ms);

    // ── e2e-scope: wall-clock around full H2D+exec+D2H ────────────────────────
    std::cout << "[cpp] e2e-scope warmup " << warmup / 2 << "\n";
    for (int i = 0; i < warmup / 2; ++i) runner.infer_end_to_end(input.data());

    std::cout << "[cpp] e2e-scope timing " << iters << "\n";
    std::vector<double> e2e_ms;
    e2e_ms.reserve(iters);
    for (int i = 0; i < iters; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        runner.infer_end_to_end(input.data());
        auto t1 = std::chrono::high_resolution_clock::now();
        e2e_ms.push_back(
            std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    Stats e2e = compute_stats(e2e_ms);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    const std::string tag =
        engine_path.find("fp16") != std::string::npos ? "fp16" : "fp32";

    std::ofstream f(out_json);
    f.setf(std::ios::fixed);
    f.precision(4);
    f << "{\n";
    f << "  \"label\": \"trt_cpp_" << tag << "\",\n";
    f << "  \"precision\": \"" << tag << "\",\n";
    write_stats_json(f, "latency", e2e);  f << ",\n";   // headline = e2e
    write_stats_json(f, "device_latency", dev); f << ",\n";
    f << "  \"fps\": " << (1000.0 / e2e.mean) << ",\n";
    f << "  \"device_fps\": " << (1000.0 / dev.mean) << "\n";
    f << "}\n";
    f.close();

    std::cout << "[cpp] device mean=" << dev.mean << "ms p99=" << dev.p99
              << "ms | e2e mean=" << e2e.mean << "ms p99=" << e2e.p99 << "ms\n";
    std::cout << "[cpp] wrote " << out_json << "\n";
    return 0;
}
