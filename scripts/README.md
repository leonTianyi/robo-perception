# scripts

_Setup + glue scripts. Heavy logic lives in the `roboperc` package; these are thin
CLIs / orchestrators._

| Script | What |
|---|---|
| `setup_python.sh` | one-time JetPack 6.2 / aarch64 Python deps (torch, onnx, rfdetr, cuda-python, pybind11, cmake) |
| `export_and_build.sh [fp16\|fp32]` | export ONNX → build engine + manifest → `cmake --build` |
| `download_sample.py` | fetch the demo image into `data/` (wraps `roboperc.sample`) |
| `download_weights.py` | manifest + sha256 weights download (**stub** — RF-DETR self-downloads for now) |
| `compare.py` | cross-path correctness vs the PyTorch reference (wraps `roboperc.eval.detection`) |
| `init_repo.sh` | start a fresh git history (pre-existing) |
