# python/roboperc/poc

_Pure-Python prototype models for fast iteration, ahead of (or instead of) a C++ adapter._

| Module | What |
|---|---|
| `rfdetr_infer` | RF-DETR PyTorch baseline; writes `results/pytorch_reference.npz` (the cross-path correctness reference) + an annotated image |

Run: `python -m roboperc.poc.rfdetr_infer`.

Even prototypes use the **shared native pre/post** (`roboperc.imageproc` →
`roboperc._native`), so a PoC's numbers are directly comparable to the ONNX / TRT /
C++ paths. When a PoC graduates, it gets a C++ adapter under
`detection/image/<model>/` behind the `Detector` contract.
