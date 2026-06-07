# bindings

_pybind11 layer exposing the C++ core + contracts to Python._

Builds the module `roboperc._native` (output into `python/roboperc/`). This is the
**parity seam**: Python paths call the C++ decode + RF-DETR pre/post here instead
of reimplementing them, so cross-backend comparisons measure the model, not the
pipeline.

Exposed:

| Symbol | What |
|---|---|
| `decode_image(path)` | → BGR HWC uint8 array (the one decode path) |
| `preprocess(img, input_w, input_h)` | → NCHW float32 `(1,3,H,W)` |
| `postprocess(logits, boxes, orig_w, orig_h, conf)` | → `list[Detection]` |
| `Detection`, `Detections`, `DetectorConfig`, `Detector` | canonical types/contract |
| `make_detector(name, cfg)` · `registered_detectors()` | the registry (e.g. build `"rfdetr"`) |
| `coco_label(id)` | label lookup |

Built by CMake (`pybind11_add_module`), not by `pip`. Run a CMake build to produce
`python/roboperc/_native*.so`; `pip install -e .` then resolves both the pure
Python and the compiled module.

> **Trampolines (future):** to let a *Python* PoC model implement the C++
> `Detector` contract and be consumed by C++ code, add a `PyDetector` trampoline
> class here. Not needed yet — current Python models run on the Python side only.
