# 2 · C++ code tour (bottom-up)

Read the C++ tree from the inside out — `core` first, then each layer that depends
on it. Open the files as you go; this chapter is the map and the commentary.

## `core/` — the vocabulary (`roboperc_core`)

Start here. `core` defines what everything else *talks about*, and depends on
nothing but OpenCV (for the image type).

- **`include/roboperc/core/detections.hpp`** — `Detection {x1,y1,x2,y2,score,
  class_id}` and `Detections {items, image_w, image_h}`. Boxes are absolute pixels
  in the *original* image, so they're directly drawable and scorable. This is the
  lingua franca.
- **`include/roboperc/core/detector.hpp`** — the contract:
  ```cpp
  class Detector {
   public:
    virtual ~Detector() = default;
    virtual Detections detect(const cv::Mat& image_bgr) = 0;
  };
  ```
  Stateless, per-frame. *Everything* model-agnostic downstream depends only on this.
- **`include/roboperc/core/config.hpp`** — `DetectorConfig {engine_path,
  conf_threshold, input_w, input_h}`. Deliberately tiny and POD-like so it can come
  from YAML, C++ defaults, or the Python binding.
- **`include/roboperc/core/registry.hpp` + `src/registry.cpp`** — the model-swap
  lever. A `DetectorRegistry` maps a name (`"rfdetr"`) to a factory
  `DetectorConfig → unique_ptr<Detector>`. Adapters self-register via the
  `REGISTER_DETECTOR("name", Type)` macro.
- **`include/roboperc/core/coco_labels.hpp`** — the COCO label table (reference
  data, mirrored in Python).
- **`tests/test_registry.cpp`** — a dependency-free, assert-based `ctest`.

### War story #1 — the macro that broke on `::`

The first version of `REGISTER_DETECTOR` pasted the *type name* into a variable
name (`_roboperc_reg_##Type`). That works for `DummyDetector` but is **ill-formed**
for a namespaced type like `roboperc::detection::rfdetr::RFDETRDetector` (you can't
token-paste `::` into an identifier). Fix: make the variable name unique via
`__LINE__` instead. See the `ROBOPERC_CAT(_roboperc_reg_, __LINE__)` in
`registry.hpp`. Lesson: registration macros should never depend on the spelling of
the type.

## `io/decode/` — one decode (`roboperc_io_decode`)

A single function, `decode_image(path) -> cv::Mat` (BGR). Trivial, but giving it a
home enforces "everyone decodes the same way." `apps` and the bindings both call it;
nobody calls `cv::imread` directly.

## `runtime/tensorrt/` — the engine executor (`roboperc_runtime_tensorrt`)

`include/roboperc/runtime/tensorrt_runner.hpp` + `src/tensorrt_runner.cpp`.
Generalized from the PoC's `04_cpp/common/trt_utils.hpp`. It is **model-agnostic**:

- loads a serialized `.engine`, discovers every I/O tensor (TRT 10 name-based API),
  `cudaMalloc`s device buffers + host staging for each;
- `load_input()`, `execute_device_only()` (pure GPU compute, for device-scope
  timing), `infer_end_to_end()` (H2D + exec + D2H), `sync()`.

Crucially it does **not** know which output is "logits" vs "boxes" — that's the
adapter's job. The runtime runs engines; it doesn't interpret them. (The Python twin
`python/roboperc/runtime/tensorrt.py` mirrors this exactly so the two TRT paths are
comparable.)

## `detection/image/rfdetr/` — the adapter (`roboperc_detection_rfdetr`)

This is where model-specific knowledge lives, and it **owns its pre/post**:

- **`include/.../preprocess.hpp`** — resize → BGR2RGB → /255 → normalize → CHW.
- **`include/.../postprocess.hpp`** — sigmoid + argmax (skipping background class 0)
  + box decode → canonical `Detections`.
- **`include/.../rfdetr_detector.hpp` + `src/rfdetr_detector.cpp`** —
  `RFDETRDetector : Detector`. Its `detect()` = preprocess → `TensorRTRunner::
  infer_end_to_end` → postprocess. It identifies the logits vs boxes outputs by
  **trailing dim** (the boxes head ends in 4; the class head ends in 91) — robust to
  the exporter naming them `dets`/`labels`.

At the bottom of the `.cpp`:
```cpp
REGISTER_DETECTOR("rfdetr", roboperc::detection::rfdetr::RFDETRDetector)
```

### War story #2 — the self-registration that vanished

That `REGISTER_DETECTOR` line runs at static-init time, but its result isn't
referenced by any other symbol. With a normal static-library link, the linker sees
"this object exports nothing anyone uses" and **drops the whole object** — so
`"rfdetr"` silently isn't registered. The fix is to link the adapter with
**whole-archive**. Modern CMake (≥3.24) expresses this cleanly:

```cmake
target_link_libraries(run_inference PRIVATE
    "$<LINK_LIBRARY:WHOLE_ARCHIVE,roboperc_detection_rfdetr>")
```

This both forces the registration object in *and* propagates the adapter's include
dirs. (The first attempt used a raw `-Wl,--whole-archive` file path in a cache var;
it didn't propagate includes or build-ordering — the generator expression is the
right tool.) You'll see this in `apps/CMakeLists.txt` and `bindings/CMakeLists.txt`.

## `apps/` — the executables

- **`run_inference.cpp`** — the deployment entry point. Note it never names
  `RFDETRDetector` directly; it does
  `DetectorRegistry::instance().create("rfdetr", cfg)`. That's the registry paying
  off: swapping models is a string, not a recompile of call sites.
- **`benchmark.cpp`** — times the engine at two scopes (CUDA-event device scope +
  wall-clock e2e) and writes a JSON the Python `report` reads. It talks to
  `TensorRTRunner` **directly**, not through `Detector`, because it's measuring the
  *runtime*, not the pre/post. (Per the README: never time the C++ path through the
  Python binding either.)

## How they link together

```
core ◄── io/decode
  ▲         ▲
  │         │
runtime/tensorrt ◄── detection/rfdetr ◄── apps / bindings  (whole-archive)
```

Each is a `STATIC` library target with `POSITION_INDEPENDENT_CODE ON` (so the
bindings `.so` can link them). One `add_subdirectory` per dir in the root
`CMakeLists.txt`.

Next: [the parity seam →](03_parity_seam.md)
