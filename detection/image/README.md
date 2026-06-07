# detection/image

_2D detectors: rfdetr, yolo, ... Each adapter owns its pre/post-processing._

## rfdetr/ — `roboperc_detection_rfdetr`

The first concrete detector. Implements the `Detector` contract on top of a
TensorRT engine and **owns its pre/post-processing**:

| File | What |
|---|---|
| `include/.../preprocess.hpp` | resize → BGR2RGB → /255 → normalise → CHW |
| `include/.../postprocess.hpp` | sigmoid + argmax (skip bg) + box decode → canonical `Detections` |
| `include/.../rfdetr_detector.hpp` · `src/rfdetr_detector.cpp` | `RFDETRDetector : Detector`; identifies logits/boxes outputs by trailing dim; self-registers as `"rfdetr"` |

The same `preprocess`/`postprocess` are exported to Python via `roboperc._native`
(see `bindings/`), so the PyTorch / ONNX / TensorRT Python paths and this C++ path
share one implementation.

> **Adding a detector** (e.g. yolo): create `detection/image/yolo/`, implement
> the `Detector` contract with your own pre/post, `REGISTER_DETECTOR("yolo", ...)`,
> and add the subdirectory to the build. Nothing else changes — the registry +
> config select it.

Build target: `roboperc_detection_rfdetr` (STATIC; link whole-archive so the
self-registration survives — see this dir's `CMakeLists.txt`).
