"""roboperc — the Python edge of the robo-perception monorepo.

PoC models, experiment orchestration, export tooling, auto-annotation, and
accuracy/perf scoring. Inference-critical pre/post-processing is NOT defined
here: it lives in C++ and is reached through the compiled module
``roboperc._native`` (built by CMake), so every Python backend and the C++
deployment path share one implementation.
"""

__all__ = ["config", "coco", "imageproc"]
