# 5 · Build system, tools & the end-to-end workflow

## The build system (CMake)

The PoC used a hand-written Makefile; the repo standard is CMake, for good reasons:
it's NVIDIA's and ROS2's native tongue, it gives you per-subsystem **library
targets** (build/test one piece in isolation), and it handles dependency discovery
and link ordering properly.

How it's wired:

- **Root `CMakeLists.txt`** — `project(... LANGUAGES CXX)` (note: *not* CUDA — we
  don't compile `.cu` files, we just link `cudart`, so `find_package(CUDAToolkit)`'s
  `CUDA::cudart` target is enough and avoids forcing an nvcc toolchain check). It
  finds OpenCV, CUDAToolkit, TensorRT, and (optionally) pybind11, enables `CTest`,
  then `add_subdirectory`s each subsystem. Bindings are only added if pybind11 is
  found — so the C++ deployment build works on a box without pybind11.
- **`cmake/FindTensorRT.cmake`** — TensorRT ships no CMake config on JetPack, so this
  module finds `nvinfer` + `nvonnxparser` in the aarch64 multiarch paths and exposes
  a clean `TensorRT::TensorRT` imported target. Override with `-DTensorRT_ROOT=...`.
- **Per-dir `CMakeLists.txt`** — each makes one `STATIC` library (or the apps/module),
  declares its public include dir, and links inward toward `core`. All set
  `POSITION_INDEPENDENT_CODE ON` so the pybind `.so` can link them.

Configure once, then incremental builds are fast:

```bash
cmake -S . -B build           # configure (finds deps, generates build files)
cmake --build build -j        # compile libs, apps, and roboperc._native
ctest --test-dir build        # run the core registry test
```

## The critical tools (and *why* each)

| Tool | Role here | Why this one |
|---|---|---|
| **CMake** (≥3.24; box has 4.3) | build system for all C++ + the pybind module | native to NVIDIA/ROS2; library-target isolation; needed `$<LINK_LIBRARY:WHOLE_ARCHIVE,…>` (3.24+) for self-registration |
| **TensorRT 10.3** | the deployment inference engine; ONNX→engine builder | the whole point — fastest path on Jetson; FP16/INT8 |
| **CUDA Toolkit 12.6** | `cudart` (device memory, streams, events) | underlies TRT; CUDA events give sub-µs device-scope timing |
| **OpenCV 4.8** | image decode + resize/cvtColor in pre/post + drawing | the same lib both languages link, so pixels match |
| **pybind11** | the C++↔Python seam (`roboperc._native`) | header-only, clean numpy interop, the modern default |
| **ONNX + onnxscript** | the export intermediate (PyTorch → ONNX) | the portable, inspectable graph format TRT consumes |
| **onnx-simplifier** | constant-fold/shape-infer the graph | makes TRT fusion more reliable (optional; build falls back to raw ONNX) |
| **onnxruntime-gpu** | the ONNX backend + a PyTorch<ORT<TRT data point | validates the exported graph independently of TRT |
| **cuda-python** | Python TensorRT runtime (malloc/memcpy/streams) | lets the Python TRT path mirror the C++ one for fair comparison |
| **rfdetr** | the model + its native ONNX exporter | upstream handles the DETR-specific export quirks |
| **jetson-stats / tegrastats** | GPU/RAM/power sampling during benchmarks | the only way to get Jetson power rails; one for Python, one for the C++ sidecar |
| **ctest** | the C++ test runner | zero-dependency; the registry test runs out of the box |
| **setuptools (≥64) / pip** | packages `roboperc` (editable) | `where=["python"]` layout + editable installs need ≥64 (see gotcha below) |

## The exact end-to-end workflow (what was actually run)

This is the order it was built and verified — and the order you'd reproduce it.

```bash
# 0. deps (one-time). torch/onnx/rfdetr/cuda-python from the Jetson wheel index.
bash scripts/setup_python.sh

# 1. C++ + bindings + Python package
cmake -S . -B build && cmake --build build -j     # → libs, apps, _native.so
ctest --test-dir build                            # core registry test: PASS
pip install -e .                                  # the roboperc package

# 2. PyTorch baseline → writes results/pytorch_reference.npz (the correctness ref)
python -m roboperc.poc.rfdetr_infer
#   → detects bus + 4 people on bus.jpg

# 3. export ONNX + build a TensorRT FP16 engine (+ manifest)
bash scripts/export_and_build.sh fp16
#   → artifacts/rfdetr_base.onnx (114 MB)
#   → artifacts/rfdetr__fp16__trt10__sm87.engine (59 MB) + .manifest.json

# 4. C++ deployment path
./build/apps/run_inference \
   "$(ls artifacts/rfdetr__fp16__*.engine | head -1)" data/sample.jpg out.jpg
#   → same 5 detections as PyTorch, FP16 scores within ~0.002

# 5. cross-backend comparison + report
python -m roboperc.experiments.run_pipeline
```

### What "verified" meant (the actual results)

- C++ builds clean; `ctest` passes; `import roboperc._native` lists `['rfdetr']`.
- PyTorch: bus 0.973 + 4 persons 0.90–0.95 (correct on the classic bus.jpg).
- TRT FP16 engine built in ~145 s, 59 MB.
- C++ `run_inference`: identical 5 detections.
- Correctness (Python): **PASS** for ONNX-fp32 and TRT-fp16 — boxes agree to
  ~2–6 ×10⁻⁴ (sub-pixel), same classes.
- C++ `benchmark` (short): ~7.4 ms device / ~8.2 ms e2e → ~122 FPS; `report`
  aggregates to `summary.csv`.

## The three gotchas hit (and the fixes — good lessons)

1. **`REGISTER_DETECTOR` token-paste** broke on namespaced types → use `__LINE__`
   for the unique symbol name. (ch.2, war story #1)
2. **Static self-registration dropped by the linker** → link the adapter with
   `$<LINK_LIBRARY:WHOLE_ARCHIVE,…>`. (ch.2, war story #2)
3. **Editable install silently no-op'd** because the system `setuptools` was 59.6
   (the `where=["python"]` packaging needs ≥64). Fix folded into
   `scripts/setup_python.sh`.

And one **design refinement** worth internalizing:

4. **The correctness verdict.** The first verdict compared *raw logit magnitudes*
   with a tight FP32 tolerance — and ONNX-fp32 "failed" with a 0.25 logit delta even
   though the boxes matched to 6 ×10⁻⁴ and it was the same 5 detections. The cause:
   the `rfdetr` **native exporter emits a deployment-mode graph** whose logit
   magnitudes drift slightly from the raw `nn.Module` the reference is dumped from.
   Raw-magnitude equality was the *wrong invariant*. The verdict now checks
   **detection-set agreement**: boxes within tolerance **and** the firing queries
   pick the same class **and** still clear threshold. This matches the README's
   principle: *parity lives in the harness, and fp16/int8 deltas are the measurement,
   not a bug.* See `python/roboperc/eval/detection.py::compare_to_reference`.

## Where to go next (suggested exercises)

- **Add a second detector** (e.g. a YOLO adapter): make `detection/image/yolo/`,
  implement `Detector` with its own pre/post, `REGISTER_DETECTOR("yolo", …)`, add the
  subdir to CMake. Watch how *nothing else* changes — that's the architecture paying
  off.
- **Implement `eval/detection.score_map`**: build `io/datasets` (a COCO sample
  loader), have both the Python and C++ paths dump canonical `Detections` keyed by
  sample id, and run one pycocotools COCOeval over both.
- **Build an INT8 engine**: add calibration to `export/build_engine.py`; the manifest
  and the detection-set verdict already handle "numerically different but correct".
- **Trace one image through the seam**: add prints in `bindings/src/module.cpp` and
  `python/roboperc/imageproc.py` and watch the same bytes cross the boundary.

← back to the [index](README.md)
