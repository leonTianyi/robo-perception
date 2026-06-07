#!/usr/bin/env bash
# Full export -> TensorRT -> C++ build, end to end.
#
#   bash scripts/export_and_build.sh [fp16|fp32]
set -euo pipefail
PRECISION="${1:-fp16}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "### 1. export ONNX + inspect + simplify"
python -m roboperc.export.to_onnx
python -m roboperc.export.inspect_onnx
python -m roboperc.export.simplify_onnx || echo "(simplify skipped)"

echo "### 2. build TensorRT engine ($PRECISION) + manifest"
python -m roboperc.export.build_engine --"$PRECISION"

echo "### 3. build C++ libs / apps / bindings"
cmake -S . -B build
cmake --build build -j

echo "### done. Engine(s):"
ls -1 artifacts/*.engine 2>/dev/null || true
echo "Try:  ./build/apps/run_inference \$(ls artifacts/rfdetr__${PRECISION}__*.engine | head -1) data/sample.jpg out.jpg"
