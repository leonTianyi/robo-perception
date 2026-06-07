"""Detection accuracy.

Two layers, matching the README's eval parity story:

1. ``compare_to_reference`` — cheap cross-path correctness on raw head outputs vs
   the PyTorch reference. FP32 paths must match tightly; FP16 may differ
   numerically as long as the *above-threshold* detections still agree.
2. ``score_map`` — mAP via pycocotools, fed by canonical detection dumps from
   BOTH the Python and C++ paths keyed by sample id. STUB until a dataset loader
   (io/datasets) lands.
"""
from __future__ import annotations

import numpy as np

from .. import config


def compare_to_reference(logits: np.ndarray, boxes: np.ndarray, tag: str,
                         ref_path=None) -> str:
    """Compare raw (logits, boxes) to the saved PyTorch reference. Returns a
    verdict string ('PASS' / 'CHECK' / 'NO-REF')."""
    ref_path = ref_path or config.REFERENCE_NPZ
    if not ref_path.exists():
        print(f"[compare:{tag}] no reference at {ref_path} — run poc.rfdetr_infer")
        return "NO-REF"

    ref = np.load(ref_path)
    rl = ref["pred_logits"].reshape(-1, ref["pred_logits"].shape[-1])  # (300,91)
    rb = ref["pred_boxes"].reshape(-1, 4)
    tl = logits.reshape(rl.shape)
    tb = boxes.reshape(rb.shape)

    # Only the above-threshold queries become detections; the rest are background
    # slots whose values are unconstrained, so an all-query max-diff is dominated
    # by noise that never matters. The meaningful check is on the detections.
    rscores = 1.0 / (1.0 + np.exp(-rl[:, 1:]))
    keep = np.where(rscores.max(axis=1) >= config.CONF_THRESHOLD)[0]

    dl_all = float(np.abs(tl - rl).max())
    dl_hi = float(np.abs(tl[keep] - rl[keep]).max()) if keep.size else 0.0
    db_hi = float(np.abs(tb[keep] - rb[keep]).max()) if keep.size else 0.0

    # Parity is about the DETECTION SET agreeing, not raw logit magnitudes. The
    # rfdetr native export is a deployment-mode graph whose class logits drift a
    # little from the raw-module reference while the geometry is preserved — and
    # fp16/int8 *should* differ numerically. So the verdict keys on:
    #   (a) boxes agree to box_tol, and
    #   (b) the firing queries pick the same class and stay above threshold.
    ref_cls = rl[keep, 1:].argmax(axis=1) + 1 if keep.size else np.array([])
    tst_cls = tl[keep, 1:].argmax(axis=1) + 1 if keep.size else np.array([])
    tscores = 1.0 / (1.0 + np.exp(-tl[keep, 1:])) if keep.size else np.array([])
    class_match = bool(np.array_equal(ref_cls, tst_cls))
    still_fire = bool(keep.size == 0 or
                      (tscores.max(axis=1) >= config.CONF_THRESHOLD).all())

    box_tol = 5e-3 if "fp16" in tag else 1e-3
    print(f"[compare:{tag}] all-query  max|Δlogits|={dl_all:.3e}  (background — info)")
    print(f"[compare:{tag}] detections max|Δlogits|={dl_hi:.3e}  max|Δboxes|={db_hi:.3e}"
          f"  ({keep.size} above {config.CONF_THRESHOLD})")
    print(f"[compare:{tag}] set: classes_match={class_match} "
          f"all_still_fire={still_fire} box_tol={box_tol}")

    verdict = "PASS" if (db_hi < box_tol and class_match and still_fire) else "CHECK"
    print(f"[compare:{tag}] verdict (detection set) -> {verdict}")
    return verdict


def score_map(*args, **kwargs):
    """mAP via pycocotools (STUB).

    Plan: both the Python and the C++ path dump canonical Detections to a common
    JSON keyed by sample id; this function loads both + the GT annotations and
    runs one pycocotools COCOeval. Needs io/datasets first.
    """
    raise NotImplementedError(
        "score_map: implement once io/datasets provides a sample loader and both "
        "paths dump canonical detections keyed by sample id.")
