# robo-perception

A modular monorepo for robotics perception study on a Jetson AGX Orin: 2D/3D
detection, SLAM, and eventual sensor fusion — built C++-primary for deployment,
with Python for prototyping, experiments, and auto-annotation.

> This README is the living architecture plan. The repo began as a deliberate
> **skeleton**; the first concrete vertical slice — **RF-DETR image detection** —
> is now built (see [Implemented](#implemented--rf-detr-detection-slice)). The
> remaining folders stay stubs on purpose: let real need pull each one into
> existence rather than populating everything up front.

## Design decisions

- **Monorepo.** Shared datasets, calibration, and the eventual fusion goal make
  separate repos painful. One env story, one place to refactor.
- **C++-primary, Python for the soft edges.** The seam falls cleanly:
  - Python: training, brand-new model PoCs, experiment orchestration (Hydra),
    auto-annotation, accuracy scoring.
  - C++: inference runtime, fusion, on-robot deployment, performance benchmarks.
  - They meet at the **ONNX export boundary** and the **pybind11 bindings**.
- **One canonical vocabulary.** `core` owns the shared types (`Detections`,
  `Pose`, `Trajectory`) and contracts (`Detector`, `Odometry`). Everything
  downstream is model-agnostic because it speaks only these types.
- **Swap models via a registry + config.** Each model is a thin adapter behind a
  contract; a config selects which one. Holding dataset + pre/post constant and
  swapping `runtime` (pytorch/onnx/tensorrt) is the comparison lever.
- **Build with CMake, not Bazel.** CMake is NVIDIA's and ROS2's native tongue.
  Reproducibility comes from Docker (`l4t` base) and cross-compile toolchains.
- **ROS-agnostic core.** ROS2 enters only as thin `rclcpp` wrappers in `apps/ros2`.

## Layout — three axes: platform, capabilities, seam

```
.
# ── platform (shared infrastructure) ──
core/            canonical types · contracts (Detector, Odometry) · registry
io/
  datasets/      sample-based loaders (detection eval)
  streams/       timestamped multi-sensor replay (SLAM / rosbag)
  decode/        ONE source of truth for image/pcl decode
calibration/     intrinsics / extrinsics — used by fusion AND odometry
runtime/         NN execution ONLY: pytorch / onnx / tensorrt
export/          to_onnx · build_engine · build_manifest.json

# ── capabilities (peers) ──
detection/
  image/         rfdetr, yolo, ...   (adapter owns its pre/post)
  pointcloud/    pointpillars, ...
odometry/        SLAM: kiss-icp -> fastlio   (stateful, streaming)
fusion/          consumes detection + odometry outputs

# ── eval + harness ──
eval/
  detection/     mAP (pycocotools)
  trajectory/    ATE / RPE (evo)
  benchmark/     latency (CUDA events) / mem / power — measured in-process

# ── seam + entry points ──
bindings/        pybind11 -> python (trampolines for Python PoC models)
python/roboperc/ poc/ · experiments/ (Hydra) · annotate/
apps/            C++ executables (run_inference, benchmark)
  ros2/          rclcpp nodes (later)

# ── support ──
configs/  scripts/  cmake/  docker/
weights/  artifacts/  data/   (gitignored — DVC / download manifest)
```

Each platform/capability dir becomes its own **CMake library target** (with
`include/ src/ tests/`) when populated, so you can build/test one subsystem in
isolation. That target-level boundary is what keeps the monorepo from rotting.

> **Where the Python lives.** The platform/capability dirs above (`core`,
> `runtime/tensorrt`, `detection/image/rfdetr`, `io/decode`, `apps`) are the
> **C++** deployment tree. All Python — the soft edge — lives in one installable
> package under `python/roboperc/`, mirroring the same taxonomy internally
> (`roboperc.runtime`, `roboperc.export`, `roboperc.eval`, `roboperc.detection`,
> `roboperc.poc`, `roboperc.experiments`). The top-level `export/` and `eval/`
> dirs are taxonomic homes whose READMEs point at that package; `runtime/`'s
> PyTorch/ONNX backends live there too (only the TensorRT backend is C++).

## Implemented — RF-DETR detection slice

The first end-to-end capability: RF-DETR run via PyTorch / ONNX Runtime / TensorRT
(Python) and via a C++ TensorRT deployment path, with shared pre/post, correctness
checks, and benchmarking. (Migrated and re-architected from the standalone
`rfdetr-poc`.)

```
            C++ (canonical, deployment)              Python (edge, experiments)
io/decode ──────────────┐
detection/rfdetr pre/post├─► roboperc._native ─► used by ALL python backends
core types + contract  ──┘   (pybind11)           (pytorch · onnx · tensorrt)
runtime/tensorrt ───────────► apps/ (run_inference, benchmark)
```

The seam is the point: every Python backend calls the **same C++** decode +
pre/post through `roboperc._native`, so swapping the runtime compares *models*, not
pipelines. Concrete now: `core`, `io/decode`, `runtime/tensorrt`,
`detection/image/rfdetr`, `bindings`, `apps`, and the full `python/roboperc`
package (export, eval, experiments). Still stubs: `odometry`, `fusion`,
`io/streams`, `detection/pointcloud`, `eval/detection`'s mAP scorer.

## Build & run

Prereqs: JetPack 6.2 (CUDA 12.6, TensorRT 10.3), OpenCV 4.8.

```bash
# 0. one-time Python deps (torch, onnx, rfdetr, cuda-python, pybind11, cmake, ...)
bash scripts/setup_python.sh

# 1. build C++ libs, apps, and the roboperc._native pybind module
cmake -S . -B build && cmake --build build -j
ctest --test-dir build                  # core registry test
pip install -e .                        # the roboperc package

# 2. lock clocks for stable benchmarks
sudo nvpmodel -m 0 && sudo jetson_clocks

# 3. PyTorch baseline (also writes the correctness reference)
python -m roboperc.poc.rfdetr_infer

# 4. export ONNX + build a TensorRT engine (+ build manifest)
bash scripts/export_and_build.sh fp16

# 5. C++ deployment path
./build/apps/run_inference "$(ls artifacts/rfdetr__fp16__*.engine | head -1)" data/sample.jpg out.jpg

# 6. full cross-backend comparison + report
python -m roboperc.experiments.run_pipeline
```

## Two paradigms, kept separate

- **Detection** is stateless and per-frame: sample in, `Detections` out.
- **Odometry/SLAM** is a stateful recursive estimator over time-synchronized
  LiDAR+IMU streams. It is a **peer** of detection, not an entry in the detector
  registry. Do not force it into the `Detector` contract.

## Eval parity (Python model vs C++ inference, same dataset)

Parity lives in the **harness**, not the outputs (fp32 vs fp16/int8 *should*
differ — that delta is the measurement).

1. Share pre/post-processing + decode in C++, called from Python — otherwise you
   compare pipelines, not models.
2. Accuracy: both paths dump canonical `Detections` to a common format keyed by
   sample id; one scorer (pycocotools) reads both.
3. Performance: each runtime self-instruments in-process (CUDA events on GPU,
   `tegrastats` for mem/power). Never time the C++ path through the binding.

## Artifact policy

- Never commit weights, `.onnx`, `.engine`, or datasets to git.
- Weights: DVC, or a `weights/models.yaml` manifest + checksum-verified download.
- `.onnx`: reproducible intermediate; version the export config.
- `.engine`: **non-portable cache** (bound to GPU arch + TRT + JetPack + flags).
  Build on-device; always record how in `build_manifest.json`. Name to
  self-document, e.g. `rfdetr__fp16__trt10__sm87.engine`.

## What's next (the stubs, in pull order)

The first slice is built (see [Implemented](#implemented--rf-detr-detection-slice)).
Grow the rest only when a real need appears:

- **`eval/detection` mAP** — wire `score_map` once `io/datasets` provides a sample
  loader; both Python and C++ paths already emit canonical `Detections`.
- **`io/datasets`** — sample-based loader (shuffleable, independent samples) for
  detection eval.
- **a second detector** (e.g. yolo) — proves the registry/contract pays off; just
  add `detection/image/yolo/` and `REGISTER_DETECTOR`.
- **`io/streams` + `odometry`** — the SLAM peer (stateful, streaming); give it its
  own contract in `core`, do **not** force it through `Detector`.
- **`fusion`** — consumes detection + odometry once both are real.
- **`apps/ros2`** — thin `rclcpp` wrappers last.
