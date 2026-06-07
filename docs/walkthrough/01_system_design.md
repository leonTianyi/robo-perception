# 1 · System design

## The problem we were solving

The PoC (`rfdetr-poc`) proved an idea: take RF-DETR, push it through
PyTorch → ONNX → TensorRT, and benchmark every stage on a Jetson AGX Orin. But it
was a **flat, numbered script pile** (`01_pytorch/`, `02_export/`, … `05_benchmark/`)
with a single `config.py` + `common.py`. Great for a spike; bad for growth:

- pre/post-processing was **reimplemented twice** (Python in `common.py`, C++ in
  `04_cpp/common/*.hpp`) and kept in sync *by hand* — a correctness landmine.
- adding a second model, or SLAM, or fusion, meant copy-paste, not extension.
- it built with a hand-written `Makefile`.

`robo-perception` is the "do it properly" home: a monorepo whose README lays out a
deliberate architecture. The task was to **land the PoC as that repo's first real
vertical slice**, reshaped to the architecture, without breaking the architecture.

## The one idea everything serves: the seam

The single most important design property:

> **There is exactly one implementation of image decode and of RF-DETR
> pre/post-processing. It is written in C++, and Python calls it.**

Why this matters. When you benchmark "PyTorch vs ONNX vs TensorRT", you want the
*only* thing that varies to be the inference engine. If each path resizes /
normalizes / decodes boxes with its own code, you're no longer comparing models —
you're comparing pipelines, and a 0.3% mAP gap might just be a different resize
interpolation. The PoC's twin implementations had exactly this risk.

So: decode lives in `io/decode`, RF-DETR pre/post lives in the adapter
`detection/image/rfdetr`, both in C++, and **`bindings/` exposes them to Python as
`roboperc._native`**. Every Python backend imports them. The C++ deployment binary
uses them directly. One source of truth.

Everything below is in service of keeping that seam clean and letting the repo grow
without rotting.

## The architecture in layers

```
core            canonical vocabulary: Detection(s), the Detector contract, registry
io/decode       one image-decode function
runtime/tensorrt model-agnostic engine executor (load .engine, run it)
detection/image/rfdetr   the RF-DETR ADAPTER: owns pre/post, implements Detector
bindings        pybind11 → roboperc._native (the seam)
apps            C++ executables (run_inference, benchmark)
python/roboperc the Python edge (everything else), mirroring the same names
```

Two layering rules make this robust:

1. **Dependencies point inward, toward `core`.** `core` knows nothing about
   TensorRT or RF-DETR. The adapter depends on `core` + `runtime`. `apps` depend on
   the adapter only through `core`'s contract. So you can add a YOLO adapter or a
   pointcloud detector without touching anything that already exists.

2. **`core` speaks only canonical types.** A `Detector` takes a `cv::Mat` and returns
   `Detections` (boxes in original-image pixels). Downstream code (drawing, eval,
   the registry, the bindings) never sees model-specific shapes — only `Detections`.

## Key design decisions and their trade-offs

| Decision | Why | Trade-off we accepted |
|---|---|---|
| **Parity via pybind11** (C++ owns pre/post; Python calls it) | makes cross-backend comparison meaningful; kills the twin-implementation drift | Python now has a **compiled dependency**; you must run a CMake build before the Python paths work. Worth it. |
| **`core` is model-agnostic; adapters own pre/post** | one new model = one new directory, zero edits elsewhere | a little boilerplate per adapter (each re-states its own pre/post), but that's *by design* — pre/post **is** part of the model |
| **Registry + `DetectorConfig`** (build a detector by name) | the "swap the model, hold everything constant" lever becomes a config string | static self-registration needs a linker trick (whole-archive — see ch.2); a known C++ wart |
| **CMake, one library target per dir** | matches NVIDIA/ROS2; lets you build/test one subsystem in isolation; target boundaries stop the monorepo rotting | more `CMakeLists.txt` files than a single Makefile |
| **All Python in one package** `python/roboperc/` | clean imports, one `pip install -e .`, mirrors the taxonomy | the top-level `export/`, `eval/`, `runtime/` dirs become "taxonomic homes" whose READMEs point into the package (a small indirection) |
| **TensorRT is C++; PyTorch/ONNX are Python-only** | TRT is the deployment runtime (must be C++); PyTorch/ONNX are prototyping/validation (Python is fine) | the three "runtime" backends live in two places; documented in `runtime/README.md` |
| **Engines are named `rfdetr__fp16__trt10__sm87.engine` + a manifest** | an `.engine` is a *non-portable cache* (bound to GPU arch + TRT + JetPack + flags). The name + `*.manifest.json` make that self-evident and reproducible | none, really — pure upside |
| **Artifacts git-ignored, never committed** | weights/onnx/engines are large and machine-specific | you must (re)build engines per machine; that's correct for TRT anyway |
| **Correctness verdict = detection-set agreement, not raw logit equality** | the native export is a *deployment-mode* graph whose logits drift slightly from the raw module; what matters is "same boxes, same classes, still fires" | a path could in principle agree on the set while a head subtly regresses — acceptable because boxes are checked sub-pixel and fp16/int8 *should* differ numerically (that delta is the measurement) |

## What was deliberately left as a stub

Following the README's "let real need pull each folder into existence":
`odometry`, `fusion`, `io/streams`, `detection/pointcloud`, and the `eval/detection`
mAP scorer are still stubs. The slice that was built is the one the PoC justified:
**image detection, end to end.**

A crucial architectural note carried over: **SLAM/odometry is a *peer* of detection,
not a `Detector`.** Detection is stateless (frame in → boxes out); odometry is a
stateful recursive estimator over time-synced streams. When it grows real, it gets
its *own* contract in `core` — it must not be bent through the `Detector` interface.

Next: [the C++ code tour →](02_cpp_tour.md)
