// Canonical detection types — the vocabulary every detector speaks.
//
// These are deliberately model- and backend-agnostic: a TensorRT RF-DETR
// adapter, a Python PoC model behind a binding, and the eval/scoring code all
// exchange exactly these structs. Boxes are absolute pixel coordinates in the
// ORIGINAL image (top-left origin), so they are directly drawable / scorable.
#pragma once

#include <cstdint>
#include <vector>

namespace roboperc {

// One detected object. (x1,y1)-(x2,y2) is an axis-aligned box in original-image
// pixels; class_id indexes the model's label set (e.g. core/coco_labels.hpp).
struct Detection {
    float x1 = 0.f;
    float y1 = 0.f;
    float x2 = 0.f;
    float y2 = 0.f;
    float score = 0.f;
    int class_id = -1;
};

// The result of running a detector on one image.
struct Detections {
    std::vector<Detection> items;
    int image_w = 0;   // original image width  the boxes refer to
    int image_h = 0;   // original image height the boxes refer to

    std::size_t size() const { return items.size(); }
    bool empty() const { return items.empty(); }
};

}  // namespace roboperc
