"""ONNX Runtime backend (CUDA EP) — the intermediate PyTorch < ORT < TRT point,
and a validation that the exported graph is numerically sound.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

from .. import config


class OnnxRunner:
    name = "onnx"

    def __init__(self, onnx_path: Path | None = None):
        import onnxruntime as ort

        if onnx_path is None:
            onnx_path = (config.ONNX_SIM_PATH if config.ONNX_SIM_PATH.exists()
                         else config.ONNX_PATH)
        if not Path(onnx_path).exists():
            raise FileNotFoundError(f"{onnx_path} not found — run the export stage")

        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                     if p in ort.get_available_providers()]
        self.sess = ort.InferenceSession(str(onnx_path), providers=providers)
        self.in_name = self.sess.get_inputs()[0].name
        self.providers = providers

    def infer(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        outs = self.sess.run(None, {self.in_name: np.ascontiguousarray(x)})
        logits = next(o for o in outs if o.shape[-1] != 4)
        boxes = next(o for o in outs if o.shape[-1] == 4)
        return logits, boxes
