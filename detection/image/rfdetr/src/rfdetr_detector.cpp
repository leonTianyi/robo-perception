#include "roboperc/detection/rfdetr/rfdetr_detector.hpp"

#include <stdexcept>

#include "roboperc/core/registry.hpp"
#include "roboperc/detection/rfdetr/postprocess.hpp"

namespace roboperc::detection::rfdetr {

namespace {

// Pick out the logits / boxes output bindings by their trailing dim: the boxes
// head ends in 4 (cx,cy,w,h); the class head ends in num_classes (e.g. 91).
const rt::Binding* find_output(const rt::TensorRTRunner& r, bool want_boxes) {
    for (const auto& b : r.bindings()) {
        if (b.is_input) continue;
        const int last = b.dims.d[b.dims.nbDims - 1];
        if ((last == 4) == want_boxes) return &b;
    }
    return nullptr;
}

}  // namespace

RFDETRDetector::RFDETRDetector(const DetectorConfig& cfg) : cfg_(cfg) {
    if (cfg_.engine_path.empty())
        throw std::invalid_argument("RFDETRDetector: empty engine_path");
    runner_ = std::make_unique<rt::TensorRTRunner>(cfg_.engine_path);

    // The engine is the source of truth for input geometry; override the config.
    const auto& in = runner_->input();
    pre_.input_h = in.dims.d[in.dims.nbDims - 2];
    pre_.input_w = in.dims.d[in.dims.nbDims - 1];
}

Detections RFDETRDetector::detect(const cv::Mat& image_bgr) {
    const int orig_w = image_bgr.cols, orig_h = image_bgr.rows;

    std::vector<float> input = preprocess(image_bgr, pre_);
    runner_->infer_end_to_end(input.data());

    const rt::Binding* logits_b = find_output(*runner_, /*want_boxes=*/false);
    const rt::Binding* boxes_b  = find_output(*runner_, /*want_boxes=*/true);
    if (!logits_b || !boxes_b)
        throw std::runtime_error(
            "RFDETRDetector: could not identify logits/boxes outputs");

    const auto* logits = reinterpret_cast<const float*>(logits_b->host.data());
    const auto* boxes  = reinterpret_cast<const float*>(boxes_b->host.data());
    const int num_queries = logits_b->dims.d[logits_b->dims.nbDims - 2];
    const int num_classes = logits_b->dims.d[logits_b->dims.nbDims - 1];

    return postprocess(logits, boxes, num_queries, num_classes, orig_w, orig_h,
                       cfg_.conf_threshold);
}

}  // namespace roboperc::detection::rfdetr

// Self-register the adapter. Link this library with --whole-archive (see
// CMakeLists) so the registration TU is not dropped by the linker.
REGISTER_DETECTOR("rfdetr", roboperc::detection::rfdetr::RFDETRDetector)
