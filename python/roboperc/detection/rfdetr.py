"""RF-DETR pipeline (Python) — the parity twin of the C++ RFDETRDetector.

Wires any runtime backend (pytorch / onnx / tensorrt) to the shared native
pre/post: decode -> preprocess -> backend.infer -> postprocess -> detections.
Because the only thing that varies between backends is ``runner``, this is the
comparison lever in concrete form.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import config, imageproc


class RFDETRPipeline:
    def __init__(self, runner, conf: float | None = None):
        self.runner = runner
        self.conf = config.CONF_THRESHOLD if conf is None else conf

    @property
    def name(self) -> str:
        return getattr(self.runner, "name", "unknown")

    def detect(self, image_bgr: np.ndarray):
        """BGR image -> list of native Detection (original-image pixels)."""
        h, w = image_bgr.shape[:2]
        x = imageproc.preprocess(image_bgr)
        logits, boxes = self.runner.infer(x)
        return imageproc.postprocess(logits, boxes, w, h, self.conf)

    def detect_file(self, path: str | Path):
        img = imageproc.decode(path)
        return img, self.detect(img)


def build_runner(backend: str, **kw):
    """Construct a runtime backend by name: pytorch | onnx | tensorrt."""
    if backend == "pytorch":
        from ..runtime.pytorch import PyTorchRunner
        return PyTorchRunner()
    if backend == "onnx":
        from ..runtime.onnx import OnnxRunner
        return OnnxRunner(kw.get("onnx_path"))
    if backend == "tensorrt":
        from ..runtime.tensorrt import TensorRTRunner
        engine = kw.get("engine_path") or config.find_engine(
            kw.get("precision", "fp16"))
        if engine is None:
            raise FileNotFoundError("no engine found — run export.build_engine")
        return TensorRTRunner(Path(engine))
    raise ValueError(f"unknown backend: {backend}")
