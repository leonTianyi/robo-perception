// RF-DETR detector: TensorRT engine behind the canonical Detector contract.
//
// Owns its pre/post (preprocess.hpp / postprocess.hpp) and a model-agnostic
// TensorRTRunner. Self-registers under the name "rfdetr" so it can be built via
// DetectorRegistry from a config (see apps/run_inference).
#pragma once

#include <memory>

#include "roboperc/core/config.hpp"
#include "roboperc/core/detector.hpp"
#include "roboperc/detection/rfdetr/preprocess.hpp"
#include "roboperc/runtime/tensorrt_runner.hpp"

namespace roboperc::detection::rfdetr {

class RFDETRDetector : public Detector {
public:
    explicit RFDETRDetector(const DetectorConfig& cfg);

    Detections detect(const cv::Mat& image_bgr) override;

private:
    DetectorConfig cfg_;
    std::unique_ptr<rt::TensorRTRunner> runner_;
    PreprocCfg pre_;
};

}  // namespace roboperc::detection::rfdetr
