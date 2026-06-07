"""The parity seam, Python side.

decode / preprocess / postprocess are thin wrappers over the C++ implementation
in ``roboperc._native``. Every Python backend (pytorch / onnx / tensorrt) goes
through here, so they feed the network identical pixels and decode outputs
identically — the comparison is then a comparison of *models*, not pipelines.

``draw_detections`` is viz-only and stays pure-Python (cv2).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import config

_NATIVE_HINT = (
    "roboperc._native is not built. Build it with CMake:\n"
    "  cmake -S . -B build && cmake --build build -j\n"
    "(it is written into python/roboperc/_native*.so)."
)

try:
    from . import _native
except ImportError as e:  # pragma: no cover - environment dependent
    _native = None
    _IMPORT_ERROR = e


def _require_native():
    if _native is None:
        raise ImportError(f"{_NATIVE_HINT}\noriginal error: {_IMPORT_ERROR}")
    return _native


def decode(path: str | Path) -> np.ndarray:
    """Decode an image file to a BGR HWC uint8 array (the one decode path)."""
    return _require_native().decode_image(str(path))


def preprocess(image_bgr: np.ndarray, input_w: int | None = None,
               input_h: int | None = None) -> np.ndarray:
    """BGR HWC uint8 -> NCHW float32 (1,3,H,W), RF-DETR normalisation (C++)."""
    n = _require_native()
    return n.preprocess(image_bgr,
                        input_w or config.INPUT_W,
                        input_h or config.INPUT_H)


def postprocess(logits: np.ndarray, boxes: np.ndarray, orig_w: int, orig_h: int,
                conf: float | None = None) -> list:
    """Decode raw RF-DETR outputs -> list of native Detection (C++)."""
    n = _require_native()
    return n.postprocess(np.asarray(logits, dtype=np.float32),
                        np.asarray(boxes, dtype=np.float32),
                        orig_w, orig_h,
                        config.CONF_THRESHOLD if conf is None else conf)


def draw_detections(image_bgr: np.ndarray, detections, out_path: str | Path) -> None:
    """Draw boxes + labels and save. Accepts native Detection objects."""
    import cv2

    from .coco import label

    vis = image_bgr.copy()
    for d in detections:
        p1 = (int(d.x1), int(d.y1))
        p2 = (int(d.x2), int(d.y2))
        cv2.rectangle(vis, p1, p2, (0, 255, 0), 2)
        cv2.putText(vis, f"{label(d.class_id)} {d.score:.2f}",
                    (p1[0], max(0, p1[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 2)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)
    print(f"[saved] {out_path}  ({len(detections)} detections)")
