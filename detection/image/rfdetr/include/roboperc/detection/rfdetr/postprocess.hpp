// RF-DETR postprocessing — decode raw head outputs into canonical Detections.
//
// Must match python/roboperc/imageproc.py (which calls THIS via the binding).
// logits: (num_queries, num_classes=91, DETR COCO indexing), boxes:
// (num_queries, 4) as normalised cx,cy,w,h in [0,1].
#pragma once

#include <cmath>

#include "roboperc/core/detections.hpp"

namespace roboperc::detection::rfdetr {

inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

inline Detections postprocess(const float* logits, const float* boxes,
                              int num_queries, int num_classes,
                              int orig_w, int orig_h, float conf_thresh) {
    Detections out;
    out.image_w = orig_w;
    out.image_h = orig_h;
    for (int q = 0; q < num_queries; ++q) {
        const float* row = logits + static_cast<std::size_t>(q) * num_classes;
        // Column 0 is background/"N/A" — never a real detection. Argmax over 1..N.
        int best_c = 1;
        float best_s = -1e30f;
        for (int c = 1; c < num_classes; ++c) {
            if (row[c] > best_s) { best_s = row[c]; best_c = c; }
        }
        const float score = sigmoid(best_s);
        if (score < conf_thresh) continue;

        const float* bx = boxes + static_cast<std::size_t>(q) * 4;
        const float cx = bx[0], cy = bx[1], w = bx[2], h = bx[3];
        out.items.push_back(Detection{
            (cx - w / 2) * orig_w, (cy - h / 2) * orig_h,
            (cx + w / 2) * orig_w, (cy + h / 2) * orig_h,
            score, best_c});
    }
    return out;
}

}  // namespace roboperc::detection::rfdetr
