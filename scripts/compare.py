#!/usr/bin/env python3
"""Cross-path correctness: compare a backend's raw outputs to the PyTorch
reference. Thin CLI over roboperc.eval.detection.compare_to_reference.

  python scripts/compare.py --backend tensorrt --precision fp16
  python scripts/compare.py --backend onnx
"""
from __future__ import annotations

import argparse

from roboperc import config, imageproc, sample
from roboperc.detection.rfdetr import build_runner
from roboperc.eval.detection import compare_to_reference


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True,
                    choices=["pytorch", "onnx", "tensorrt"])
    ap.add_argument("--precision", default="fp16")
    args = ap.parse_args()

    img = imageproc.decode(sample.ensure_sample_image())
    x = imageproc.preprocess(img)
    runner = build_runner(args.backend, precision=args.precision)
    logits, boxes = runner.infer(x)
    tag = f"{args.backend}_{args.precision}"
    compare_to_reference(logits, boxes, tag)


if __name__ == "__main__":
    main()
