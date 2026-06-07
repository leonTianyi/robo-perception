# export

_to_onnx, build_engine, and build_manifest.json (records JetPack/TRT/precision per engine)._

The PyTorch → ONNX → TensorRT boundary. It is **Python-implemented** (export needs
torch / the TRT builder), so the code lives in the package:

| Step | Module |
|---|---|
| `to_onnx` | `python -m roboperc.export.to_onnx` (native exporter, manual fallback) |
| `inspect_onnx` | `python -m roboperc.export.inspect_onnx` (I/O, op histogram, TRT-risk scan) |
| `simplify_onnx` | `python -m roboperc.export.simplify_onnx` (onnx-simplifier) |
| `build_engine` | `python -m roboperc.export.build_engine --fp16` → engine **+ manifest** |

Outputs land in `artifacts/` (gitignored). Engines are named self-documentingly,
e.g. `rfdetr__fp16__trt10__sm87.engine`, with a sibling `*.manifest.json`
recording TRT version, sm, JetPack, ONNX sha256, build time — because an `.engine`
is a non-portable cache bound to GPU arch + TRT + JetPack + flags.

> This top-level dir is the **taxonomic home / entry point**; the implementation is
> under `python/roboperc/export/`. A C++ on-device engine builder could live here
> later if we need to build engines without Python.
