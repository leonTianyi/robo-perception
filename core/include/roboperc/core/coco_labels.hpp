// COCO label set in DETR / torchvision indexing (index 0 = background/"N/A",
// 1 = person, 6 = bus, ...). Reference data, model-agnostic — shared by every
// adapter and by the Python side (mirrored in python/roboperc/coco.py).
//
// 91 entries with "N/A" placeholders for the unused COCO ids; this is the
// indexing RF-DETR emits its class logits in.
#pragma once

#include <string>
#include <vector>

namespace roboperc {

inline const std::vector<std::string>& coco_classes() {
    static const std::vector<std::string> kClasses = {
        "N/A", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
        "train", "truck", "boat", "traffic light", "fire hydrant", "N/A",
        "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
        "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "N/A",
        "backpack", "umbrella", "N/A", "N/A", "handbag", "tie", "suitcase",
        "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
        "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
        "N/A", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
        "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
        "donut", "cake", "chair", "couch", "potted plant", "bed", "N/A",
        "dining table", "N/A", "N/A", "toilet", "N/A", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster",
        "sink", "refrigerator", "N/A", "book", "clock", "vase", "scissors",
        "teddy bear", "hair drier", "toothbrush"};
    return kClasses;
}

// Safe label lookup: out-of-range ids fall back to the numeric id as a string.
inline std::string coco_label(int class_id) {
    const auto& c = coco_classes();
    if (class_id >= 0 && static_cast<std::size_t>(class_id) < c.size())
        return c[class_id];
    return std::to_string(class_id);
}

}  // namespace roboperc
