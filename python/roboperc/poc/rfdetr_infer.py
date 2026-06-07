"""RF-DETR PyTorch baseline — the correctness reference for every other path.

Runs the raw nn.Module on our (native) preprocessing, saves the raw output tensors
to ``results/pytorch_reference.npz`` for cross-path comparison, and draws the
detections. Run: ``python -m roboperc.poc.rfdetr_infer``.
"""
from __future__ import annotations

import numpy as np

from .. import config, coco, imageproc, sample
from ..runtime.pytorch import PyTorchRunner


def main():
    config.ensure_dirs()
    img_path = sample.ensure_sample_image()
    img_bgr = imageproc.decode(img_path)
    orig_h, orig_w = img_bgr.shape[:2]
    print(f"[input] {img_path.name}  {orig_w}x{orig_h}")

    x = imageproc.preprocess(img_bgr)          # native, shared pre-processing
    runner = PyTorchRunner()
    logits, boxes = runner.infer(x)
    print(f"[output] pred_logits {logits.shape}  pred_boxes {boxes.shape}")

    np.savez(config.REFERENCE_NPZ, pred_logits=logits, pred_boxes=boxes,
             input=x, orig_w=orig_w, orig_h=orig_h)
    print(f"[saved] {config.REFERENCE_NPZ}")

    dets = imageproc.postprocess(logits, boxes, orig_w, orig_h)
    imageproc.draw_detections(img_bgr, dets,
                              config.RESULTS_DIR / "pytorch_detections.jpg")

    print(f"\n[detections >= {config.CONF_THRESHOLD}]")
    for d in sorted(dets, key=lambda d: -d.score):
        print(f"  {coco.label(d.class_id):15s} {d.score:.3f}  "
              f"[{d.x1:.0f},{d.y1:.0f},{d.x2:.0f},{d.y2:.0f}]")


if __name__ == "__main__":
    main()
