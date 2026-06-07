// Minimal configuration passed to a detector factory.
//
// Kept intentionally small and POD-like so it can be populated from a YAML file
// (apps, via a tiny parser), from C++ defaults, or from the pybind layer. Grow
// it as new detectors need new knobs — but resist turning it into a god-object;
// model-specific tuning belongs in the adapter.
#pragma once

#include <string>

namespace roboperc {

struct DetectorConfig {
    std::string engine_path;       // serialized TensorRT engine (.engine)
    float conf_threshold = 0.35f;  // score cutoff applied in postprocess
    int input_w = 560;             // network input width  (overridden by engine)
    int input_h = 560;             // network input height (overridden by engine)
};

}  // namespace roboperc
