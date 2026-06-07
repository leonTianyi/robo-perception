"""PyTorch backend — the FP32 correctness baseline every other path is compared to.

Loads RF-DETR via the rfdetr API and reaches the underlying nn.Module (the exact
graph we export to ONNX), so the PyTorch numbers match ONNX/TRT.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from .. import config


def load_raw_module():
    """Return the underlying torch nn.Module (eval, on CUDA)."""
    import torch  # noqa: F401  (import side effects / clearer error if missing)
    from rfdetr import RFDETRBase, RFDETRLarge

    Cls = RFDETRBase if config.MODEL_VARIANT == "base" else RFDETRLarge
    api = Cls(resolution=config.INPUT_H)
    # rfdetr nests the module: api.model (LWDETR wrapper) -> .model (nn.Module).
    raw = api.model.model
    raw.eval().cuda()
    return api, raw


class PyTorchRunner:
    name = "pytorch"

    def __init__(self):
        import torch

        self._torch = torch
        self.api, self.raw = load_raw_module()

    def infer(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        torch = self._torch
        with torch.no_grad():
            out = self.raw(torch.from_numpy(np.ascontiguousarray(x)).cuda())
        if isinstance(out, dict):
            logits, boxes = out["pred_logits"], out["pred_boxes"]
        else:
            logits, boxes = out[0], out[1]
        return logits.float().cpu().numpy(), boxes.float().cpu().numpy()

    def synchronize(self):
        self._torch.cuda.synchronize()
