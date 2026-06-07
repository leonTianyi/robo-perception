"""Neural inference backends for the Python edge: pytorch / onnx / tensorrt.

Each backend takes a preprocessed NCHW float32 input and returns the raw RF-DETR
head outputs ``(logits, boxes)``. Pre/post-processing is shared and lives in
``roboperc.imageproc`` (C++ via the binding) — backends never reimplement it.
This is the "swap the runtime, hold everything else constant" comparison lever.
"""
