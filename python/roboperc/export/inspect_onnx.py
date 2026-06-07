"""Inspect an ONNX graph before handing it to TensorRT: I/O, op histogram, a scan
for TRT-risky ops, and a checker pass. Run on both raw and simplified graphs.

Run: ``python -m roboperc.export.inspect_onnx [--path artifacts/rfdetr_base.onnx]``.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import onnx

from .. import config

# Ops that commonly trip the ONNX->TRT parse for DETR-family models. Presence is
# not necessarily fatal — it's a "look here first if the build fails".
TRT_RISKY_OPS = {
    "NonZero": "dynamic output size — needs a data-dependent shape path",
    "NonMaxSuppression": "dynamic output — keep NMS out of the graph",
    "TopK": "fine if k is static; dynamic k forces shape tensors",
    "Einsum": "TRT parser is picky about equation strings; prefer MatMul",
    "GridSample": "deformable-attention op; needs TRT >=8.6 / maybe a plugin",
    "RoiAlign": "rarely supported cleanly",
    "ScatterND": "supported but can block fusion",
    "Loop": "control flow — tracing captured something dynamic",
    "If": "control-flow branch baked into the graph",
    "ScaledDotProductAttention": "fused attention; TRT support is shape-dependent",
}


def describe_io(model):
    def shp(t):
        d = t.type.tensor_type.shape.dim
        return [(x.dim_value if x.HasField("dim_value") else (x.dim_param or "?"))
                for x in d]
    print("── inputs ──")
    for i in model.graph.input:
        print(f"  {i.name:20s} {shp(i)}")
    print("── outputs ──")
    for o in model.graph.output:
        print(f"  {o.name:20s} {shp(o)}")


def op_histogram(model):
    counts = Counter(n.op_type for n in model.graph.node)
    print(f"\n── op histogram ({len(model.graph.node)} nodes, "
          f"{len(counts)} distinct) ──")
    for op, c in counts.most_common():
        print(f"  {op:30s} {c}")
    return counts


def flag_risky(counts):
    hits = [(op, counts[op], why) for op, why in TRT_RISKY_OPS.items()
            if op in counts]
    print("\n── TRT risk scan ──")
    if not hits:
        print("  none of the watch-list ops present ✓")
    for op, c, why in hits:
        print(f"  ⚠ {op} ×{c}: {why}")
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", type=Path, default=config.ONNX_PATH)
    args = ap.parse_args()
    if not args.path.exists():
        print(f"[error] {args.path} not found — run export.to_onnx first")
        sys.exit(1)

    print(f"[inspect] {args.path}  ({args.path.stat().st_size / 1e6:.1f} MB)\n")
    model = onnx.load(str(args.path))
    print(f"ir_version={model.ir_version}  opset="
          f"{ {imp.domain or 'ai.onnx': imp.version for imp in model.opset_import} }")
    describe_io(model)
    flag_risky(op_histogram(model))

    print("\n── checker ──")
    try:
        onnx.checker.check_model(model)
        print("  onnx.checker: PASS ✓")
    except Exception as e:
        print(f"  onnx.checker: FAIL — {e}")


if __name__ == "__main__":
    main()
