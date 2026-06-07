# runtime

_Neural inference backends ONLY: pytorch / onnx / tensorrt. SLAM does not touch this._

The three backends split by language along the project's C++/Python seam:

| Backend | Where | Role |
|---|---|---|
| **tensorrt** | `runtime/tensorrt/` (C++ lib `roboperc_runtime_tensorrt`) | deployment path; loads `.engine`, runs `enqueueV3` |
| tensorrt (py) | `python/roboperc/runtime/tensorrt.py` | same engine via cuda-python — the Python twin |
| pytorch | `python/roboperc/runtime/pytorch.py` | FP32 baseline / correctness reference |
| onnx | `python/roboperc/runtime/onnx.py` | ONNX Runtime CUDA EP — the intermediate data point |

The C++ `TensorRTRunner` is **model-agnostic**: it loads an engine and runs it,
but does not interpret outputs. Deciding which output tensor is "logits" vs
"boxes" is the detector adapter's job (`detection/image/rfdetr`).

All Python backends share **one** pre/post-processing implementation — the C++
code in `detection/image/rfdetr`, reached through the pybind11 module
`roboperc._native`. That is what makes the cross-backend comparison a comparison
of models, not of pipelines.

Build target: `roboperc_runtime_tensorrt` (STATIC).
