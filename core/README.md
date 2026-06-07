# core

_Canonical types, contracts, and the registry. The vocabulary everything speaks._

`roboperc_core` is the model- and backend-agnostic foundation. Everything
downstream depends only on these types, never on a concrete model or runtime.

| File | What |
|---|---|
| `include/roboperc/core/detections.hpp` | `Detection`, `Detections` — boxes in original-image pixels |
| `include/roboperc/core/detector.hpp` | `Detector` contract: `cv::Mat -> Detections` (stateless, per-frame) |
| `include/roboperc/core/config.hpp` | `DetectorConfig` passed to factories |
| `include/roboperc/core/registry.hpp` · `src/registry.cpp` | name→factory registry + `REGISTER_DETECTOR` macro (the model-swap lever) |
| `include/roboperc/core/coco_labels.hpp` | COCO label set (DETR indexing); mirrored in `python/roboperc/coco.py` |
| `tests/test_registry.cpp` | dependency-free `ctest` check |

> **Odometry/SLAM is a peer, not a `Detector`.** When `odometry` grows real, give
> it its own contract here (`Pose`, `Trajectory`, an `Odometry` interface) — do
> not bend the stateless per-frame `Detector` interface to fit a stateful
> recursive estimator.

Build target: `roboperc_core` (STATIC). Test: `ctest -R test_registry`.
