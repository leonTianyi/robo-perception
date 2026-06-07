# 4 · The Python edge (`python/roboperc/`)

All Python lives in one installable package that **mirrors the repo taxonomy**.
This chapter walks the modules in the order they're used in a real run.

```
roboperc/
  config.py        paths + model/benchmark knobs
  coco.py          label table (mirror of the C++ one)
  imageproc.py     the seam wrappers (ch.3) + draw
  sample.py        one-off demo image fetch
  runtime/         pytorch.py · onnx.py · tensorrt.py  (+ base.py protocol)
  detection/rfdetr.py   pipeline = runner + native pre/post
  export/          to_onnx · inspect_onnx · simplify_onnx · build_engine
  eval/            benchmark · tegrastats · report · detection (correctness/mAP)
  poc/rfdetr_infer.py   PyTorch baseline + reference dump
  experiments/run_pipeline.py   the orchestrator (replaces run_all.sh)
  annotate/        stub
```

## `config.py` — the single source of paths/knobs

Computes `REPO_ROOT` (three parents up from the file) and points at the gitignored
artifact dirs: `weights/`, `artifacts/` (onnx + engines), `data/` (samples),
`results/` (bench JSON, annotated images, the reference `.npz`). Two helpers worth
noting:

- `engine_name(precision, trt_major, sm)` → `rfdetr__fp16__trt10__sm87.engine`
- `find_engine(precision)` → globs `artifacts/` for the newest matching engine, so
  consumers locate an engine without hardcoding its full name.

## `runtime/` — the three backends

Each implements the same informal contract (`base.py`'s `Runner` protocol): take a
preprocessed `(1,3,H,W)` array, return `(logits, boxes)`.

- **`pytorch.py`** — loads RF-DETR via the `rfdetr` API and reaches the underlying
  `nn.Module` (`api.model.model`). This is the FP32 **reference**. Note the
  comparison reference is dumped from *this raw module* — relevant in ch.5's "why
  the verdict changed".
- **`onnx.py`** — an ONNX Runtime CUDA-EP session; matches outputs by trailing dim.
- **`tensorrt.py`** — the cuda-python twin of the C++ `TensorRTRunner`, including the
  device-scope hooks (`load_input_once` / `infer_device_only` / `synchronize`) so it
  can be benchmarked at the same two scopes as the C++ path.

They lazily import `torch` / `onnxruntime` / `tensorrt` *inside* `__init__`, so
importing the package (or a backend you're not using) never pulls a heavy/missing
dependency. That's why `import roboperc` works even on a box without torch.

## `detection/rfdetr.py` — the pipeline + `build_runner`

`RFDETRPipeline` is the parity twin of the C++ `RFDETRDetector` (the three-line
detect from ch.3). `build_runner("pytorch"|"onnx"|"tensorrt", ...)` constructs a
backend by name — the Python echo of the C++ registry.

## `export/` — the PyTorch → ONNX → TensorRT boundary

This is Python because export needs torch and the TRT builder.

- **`to_onnx.py`** — tries `rfdetr`'s native exporter first (handles the DETR export
  quirks), falls back to a hand-rolled `torch.onnx.export`. Writes
  `artifacts/rfdetr_base.onnx`.
- **`inspect_onnx.py`** — prints I/O, an op histogram, and a **scan for TRT-risky
  ops** (NonZero, Einsum, GridSample, …) — your first stop when a build fails.
- **`simplify_onnx.py`** — constant-folds + shape-infers via onnx-simplifier (makes
  TRT's fusion more reliable). It reads the real input tensor name from the graph
  (the native export calls it `input`, not `images`) — a small robustness fix.
- **`build_engine.py`** — parses the ONNX with the TRT `OnnxParser`, builds with the
  FP16/FP32 flag, serializes the engine **and writes `*.manifest.json`** recording
  TRT version, compute capability (`sm`), JetPack, the ONNX sha256, build time, and
  timestamp. This is the README's artifact policy made concrete: an `.engine` is a
  non-portable cache, so it self-documents how it was built.

## `eval/` — where parity is *measured*

- **`benchmark.py`** — `run_benchmark()` does warm-up + a timed loop and samples
  Jetson resources (GPU/RAM/power via jetson-stats). The in-process Python twin of
  `apps/benchmark.cpp`.
- **`tegrastats.py`** — folds a `tegrastats` sidecar log into a C++ bench JSON
  (the C++ loop can't sample jtop from inside itself).
- **`report.py`** — aggregates every `results/bench_*.json` into one sorted table +
  `summary.csv`, with speedups vs the slowest path.
- **`detection.py`** — `compare_to_reference()` (cross-path correctness) and a
  `score_map()` **stub** (pycocotools mAP, pending `io/datasets`). The verdict logic
  here is subtle — see ch.5.

## `experiments/run_pipeline.py` — the whole thing in motion

The Python replacement for the PoC's `run_all.sh`. It:

1. runs the PyTorch baseline (writes the reference `.npz` + annotated image),
2. runs ONNX Runtime (correctness vs reference + bench),
3. runs TensorRT for each precision whose engine exists (correctness + bench at
   device + e2e scope),
4. prints the aggregated report.

Backends that aren't ready (no engine, missing dep) are **skipped with a note**, not
fatal — so you can run it incrementally as you build pieces. It optionally reads
`configs/rfdetr_base.yaml` for warmup/iters/conf (Hydra-ready, but no Hydra
dependency yet).

This is the best single file to read to see how everything composes.

Next: [build, tools & workflow →](05_build_tools_and_workflow.md)
