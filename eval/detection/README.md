# eval/detection

_Accuracy: mAP via pycocotools, fed by canonical Detections dumps from BOTH Python and C++ paths._

Python-implemented in `python/roboperc/eval/detection.py`:

- `compare_to_reference(logits, boxes, tag)` — cheap cross-path correctness on raw
  head outputs vs the PyTorch reference. FP32 must match tightly; FP16 may differ
  numerically as long as the above-threshold detections agree. Run via
  `scripts/compare.py` or inside `experiments.run_pipeline`.
- `score_map(...)` — **stub**: mAP via pycocotools, fed by canonical detection
  dumps from both paths keyed by sample id. Needs `io/datasets` (a sample loader)
  first, per the README's eval-parity plan.

> Parity lives in the harness, not the outputs: one scorer reads both the Python
> and C++ dumps, so fp32-vs-fp16/int8 deltas are the *measurement*, not a bug.
