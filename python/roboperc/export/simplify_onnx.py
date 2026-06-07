"""Constant-fold + shape-infer the ONNX graph with onnx-simplifier, which makes
TensorRT's shape inference and fusion more reliable. Validates equivalence on
random inputs internally.

Run: ``python -m roboperc.export.simplify_onnx``.
"""
from __future__ import annotations

import sys

import onnx

from .. import config


def main():
    if not config.ONNX_PATH.exists():
        print(f"[error] {config.ONNX_PATH} not found — run export.to_onnx first")
        sys.exit(1)

    from onnxsim import simplify

    print(f"[simplify] loading {config.ONNX_PATH}")
    model = onnx.load(str(config.ONNX_PATH))
    # Input name varies by exporter (native uses "input", manual uses "images").
    in_name = model.graph.input[0].name
    overwrite = {in_name: [1, 3, config.INPUT_H, config.INPUT_W]}
    print(f"[simplify] folding with {in_name} shape {overwrite[in_name]}")
    model_sim, ok = simplify(model, overwrite_input_shapes=overwrite, check_n=3)
    if not ok:
        print("[simplify] WARNING: equivalence check did not fully validate")

    onnx.save(model_sim, str(config.ONNX_SIM_PATH))
    n0, n1 = len(model.graph.node), len(model_sim.graph.node)
    print(f"[simplify] nodes {n0} -> {n1}  ({n0 - n1} folded)")
    print(f"[saved] {config.ONNX_SIM_PATH}  "
          f"({config.ONNX_SIM_PATH.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
