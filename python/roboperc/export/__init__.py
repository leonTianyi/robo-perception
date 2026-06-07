"""Export tooling — the PyTorch -> ONNX -> TensorRT boundary.

This is one of the two seams where C++ and Python meet (the other is the pybind11
bindings). Modules:
  to_onnx       - export RF-DETR to ONNX (native exporter, manual fallback)
  inspect_onnx  - I/O, op histogram, TRT-risky-op scan, checker
  simplify_onnx - constant-fold + shape-infer (onnx-simplifier)
  build_engine  - ONNX -> serialized .engine (+ build_manifest.json)
"""
