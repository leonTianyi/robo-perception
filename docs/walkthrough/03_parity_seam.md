# 3 · The parity seam (pybind11)

This is the crux of the whole design. If you understand this chapter, you
understand why the repo is shaped the way it is.

## The goal restated

Python needs to run RF-DETR three ways (PyTorch, ONNX Runtime, TensorRT) and the
C++ binary needs to run it once (TensorRT). We want **all four** to decode the image
and do pre/post-processing with the *exact same code*, so any difference in output
is attributable to the inference engine alone.

The only way to truly guarantee that is to have **one** implementation and call it
from both languages. C++ is the natural owner (it's the deployment language), so
Python reaches into C++. That bridge is `bindings/`.

## The module: `roboperc._native`

`bindings/src/module.cpp` is a pybind11 module. It exposes:

| Python symbol | Backed by |
|---|---|
| `decode_image(path)` → BGR `np.uint8` HWC | `io::decode_image` |
| `preprocess(img, input_w, input_h)` → `np.float32` `(1,3,H,W)` | `detection::rfdetr::preprocess` |
| `postprocess(logits, boxes, w, h, conf)` → `list[Detection]` | `detection::rfdetr::postprocess` |
| `Detection`, `Detections`, `DetectorConfig`, `Detector` | the `core` types/contract |
| `make_detector(name, cfg)` / `registered_detectors()` | the `core` registry |
| `coco_label(id)` | `core::coco_label` |

The tricky part is **numpy ↔ cv::Mat / pointer** conversion. Look at the small
helpers in `module.cpp`:

- `numpy_to_mat_bgr` — takes a `(H,W,3) uint8` array (forced contiguous) and
  `memcpy`s it into a `cv::Mat`. So Python's decoded image becomes a `cv::Mat` the
  C++ pre/post understands.
- `mat_to_numpy_bgr` — the reverse, for `decode_image`.
- `preprocess_py` — runs the C++ `preprocess` and reshapes the flat CHW vector into
  a `(1,3,H,W)` numpy array (the batch shape the runners feed to the engine/model).
- `postprocess_py` — infers `num_queries`/`num_classes` from the array's trailing
  two dims and calls the C++ `postprocess`, returning bound `Detection` objects.

Because `pybind11/stl.h` is included, `std::vector<Detection>` auto-converts to a
Python list, and `Detection`'s fields are exposed read-only.

## Where the `.so` goes (and why)

`bindings/CMakeLists.txt` sets:

```cmake
set_target_properties(_native PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/python/roboperc)
```

So the CMake build drops `_native.cpython-310-aarch64-linux-gnu.so` **directly into
the Python package**. That means after a build, `import roboperc._native` works from
the source tree, and `pip install -e .` ties the pure-Python and compiled parts
together. The module also whole-archives `roboperc_detection_rfdetr` so the
`"rfdetr"` registration survives (war story #2 from chapter 2).

> **Two build systems, on purpose.** CMake builds the C++ + the `.so`; `pip` installs
> the Python package. They are not coupled (no scikit-build) — simpler and more
> debuggable on a Jetson, at the cost of "remember to run cmake before pip".

## The seam in use: `python/roboperc/imageproc.py`

This is the thin Python wrapper every backend goes through:

```python
from . import _native            # the compiled bridge
def decode(path):      return _native.decode_image(str(path))
def preprocess(img):   return _native.preprocess(img, INPUT_W, INPUT_H)
def postprocess(...):  return _native.postprocess(...)
def draw_detections(...):  # pure cv2 — viz only, NOT inference-critical
```

If the `.so` isn't built, `imageproc` raises a clear, actionable error telling you
to run CMake. `draw_detections` stays pure-Python because drawing boxes is not part
of the model and doesn't need to be shared.

## Why not the other direction (Python model behind the C++ contract)?

You *could* let a Python PoC model implement the C++ `Detector` and be driven from
C++ (a pybind11 "trampoline"). That's noted as a future extension in
`bindings/README.md`, but it wasn't needed: current Python models run on the Python
side only. The seam we built (C++ pre/post → Python) is the one the parity goal
actually requires.

## The payoff, concretely

In `python/roboperc/detection/rfdetr.py`, the entire per-backend pipeline is:

```python
x = imageproc.preprocess(image_bgr)     # C++ pre
logits, boxes = self.runner.infer(x)    # the ONLY thing that varies
return imageproc.postprocess(logits, boxes, w, h, self.conf)  # C++ post
```

Swap `runner` (pytorch / onnx / tensorrt) and *nothing else changes*. That's the
comparison lever, and it's only trustworthy because pre/post are the shared native
code.

Next: [the Python edge →](04_python_edge.md)
