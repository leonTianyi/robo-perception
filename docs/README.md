# docs — design notes

Design rationale for the RF-DETR slice, carried over from the original
`rfdetr-poc`. The **conceptual** content (why ONNX→TensorRT works the way it does,
benchmarking methodology, environment gotchas) is unchanged and still authoritative
— start with `02_onnx_to_tensorrt.md`.

The **paths** in these notes refer to the PoC's flat layout; here is the mapping to
the integrated structure:

| PoC (docs refer to) | robo-perception |
|---|---|
| `config.py`, `common.py` | `python/roboperc/config.py`, `coco.py`, `imageproc.py` |
| `01_pytorch/` | `python/roboperc/runtime/pytorch.py`, `poc/rfdetr_infer.py` |
| `02_export/` | `python/roboperc/export/` |
| `03_tensorrt/` | `python/roboperc/export/build_engine.py`, `runtime/tensorrt.py` |
| `04_cpp/` | `core/`, `io/decode/`, `runtime/tensorrt/`, `detection/image/rfdetr/`, `apps/` (CMake) |
| `05_benchmark/` | `python/roboperc/eval/` + `apps/benchmark.cpp` |
| `models/` | `artifacts/` (onnx + engines, gitignored) |
| `results/` | `results/` (gitignored) |
| `Makefile` | CMake (`CMakeLists.txt` per subsystem) |

Two things the integration adds beyond the PoC:
- **Parity via pybind11.** The PoC reimplemented pre/post in C++ and Python; here
  there is one C++ implementation, exposed as `roboperc._native`, that every Python
  path calls. See `bindings/` and the root README's seam diagram.
- **Engine manifests.** Each `.engine` is written with a `*.manifest.json`
  recording how it was built (TRT/sm/JetPack/onnx hash) — see `export/build_engine.py`.

| Doc | Topic |
|---|---|
| `00_project_overview.md` | goals, scope, pipeline overview |
| `01_model_and_export.md` | RF-DETR, the ONNX export quirks |
| `02_onnx_to_tensorrt.md` | the conversion — **start here** |
| `03_benchmarking_methodology.md` | timing scopes, correctness tolerances |
| `04_environment_notes.md` | JetPack / CUDA / TRT / OpenCV gotchas |
| `05_results_and_benchmarks.md` | measured numbers |
| `06_team_share.md` | sharing / handoff notes |
