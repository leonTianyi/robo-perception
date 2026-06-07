// The Detector contract — the single seam every 2D image detector implements.
//
// Detection is stateless and per-frame: image in, canonical Detections out. This
// is what makes everything downstream (eval, apps, bindings) model-agnostic.
// Stateful estimators (odometry/SLAM) are a PEER of this, NOT an implementation
// of it — do not force them through this interface.
#pragma once

#include <opencv2/core.hpp>

#include "roboperc/core/detections.hpp"

namespace roboperc {

class Detector {
public:
    virtual ~Detector() = default;

    // Run inference on one BGR image (as produced by io/decode) and return
    // detections in original-image pixel coordinates.
    virtual Detections detect(const cv::Mat& image_bgr) = 0;
};

}  // namespace roboperc
