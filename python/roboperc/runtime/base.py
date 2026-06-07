"""The Runner contract every Python backend implements."""
from __future__ import annotations

from typing import Protocol, Tuple

import numpy as np


class Runner(Protocol):
    """A neural backend: preprocessed input in, raw (logits, boxes) out.

    Optional methods used by eval/benchmark for device-scope timing
    (``load_input_once`` / ``infer_device_only`` / ``synchronize``) are provided
    only by backends that can isolate pure GPU compute (TensorRT).
    """

    name: str

    def infer(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Run one inference; return (logits, boxes) as numpy arrays."""
        ...
