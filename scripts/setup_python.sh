#!/usr/bin/env bash
# Install the Python deps for the RF-DETR slice on JetPack 6.2 / aarch64.
#
# Uses `pip install --user` (~/.local, no sudo). Run once: bash scripts/setup_python.sh
set -euo pipefail

PIP="pip3 install --user --no-cache-dir"

echo "=== -1  build tooling (editable install of roboperc needs setuptools>=64) ==="
$PIP -U "setuptools>=64" wheel

echo "=== 0  numpy (torch wheels need >=1.26; pin <2 — see step 6) ==="
$PIP "numpy==1.26.1"

echo "=== 1  PyTorch for JetPack 6.2 / CUDA 12.6 ==="
# jetson-ai-lab is THE community wheel index for Jetson. torch 2.8.0 / tv 0.23.0
# is the documented-working pair for JP6.2.
$PIP torch==2.8.0 torchvision==0.23.0 \
  --index-url https://pypi.jetson-ai-lab.io/jp6/cu126

echo "=== 2  ONNX toolchain ==="
$PIP onnx onnxscript
$PIP onnxruntime-gpu --index-url https://pypi.jetson-ai-lab.io/jp6/cu126
$PIP cmake                 # pip cmake (no sudo) — also drives the C++/bindings build
$PIP pybind11              # required to build roboperc._native
$PIP onnxsim || echo "[warn] onnxsim build failed — simplification step will be skipped"

echo "=== 3  RF-DETR + annotation/eval helpers ==="
$PIP rfdetr supervision pycocotools pandas tabulate pyyaml

echo "=== 4  CUDA Python bindings (MUST match the CUDA 12.6 driver) ==="
$PIP psutil "cuda-python==12.6.2.post1"

echo "=== 5  Pin numpy<2 + drop pip OpenCV (run LAST) ==="
# torch 2.8.0 is built against numpy 1.x; rfdetr deps pull numpy 2.x, so force back.
$PIP "numpy==1.26.4"
# Use the system cv2 4.8 (same OpenCV the C++ path links); remove the pip one.
pip3 uninstall -y opencv-python opencv-python-headless 2>/dev/null || true

echo ""
echo "Done. Next:"
echo "  cmake -S . -B build && cmake --build build -j   # libs, apps, roboperc._native"
echo "  pip install -e .                                # the roboperc package"
echo "  python -m roboperc.poc.rfdetr_infer"
