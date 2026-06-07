"""Central configuration for the Python edge — paths, model + benchmark knobs.

Paths point at the repo's gitignored artifact dirs (``weights/`` ``artifacts/``
``data/`` ``results/``), per the README artifact policy. Engines are named
self-documentingly (``rfdetr__fp16__trt10__sm87.engine``); use :func:`find_engine`
to resolve one by precision.
"""
from __future__ import annotations

import glob
from pathlib import Path

# python/roboperc/config.py -> repo root is three parents up.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# ── model ──────────────────────────────────────────────────────────────────────
MODEL_VARIANT = "base"      # "base" | "large"
INPUT_H = 560               # must be divisible by 14 (ViT patch size)
INPUT_W = 560
NUM_QUERIES = 300           # fixed DETR query slots
CONF_THRESHOLD = 0.35

# ── paths (all gitignored) ──────────────────────────────────────────────────────
WEIGHTS_DIR = REPO_ROOT / "weights"      # model weights / manifest
ARTIFACTS_DIR = REPO_ROOT / "artifacts"  # reproducible build outputs: .onnx / .engine
DATA_DIR = REPO_ROOT / "data"            # datasets / sample images
RESULTS_DIR = REPO_ROOT / "results"      # bench JSON, annotated images, references

ONNX_PATH = ARTIFACTS_DIR / f"rfdetr_{MODEL_VARIANT}.onnx"
ONNX_SIM_PATH = ARTIFACTS_DIR / f"rfdetr_{MODEL_VARIANT}_sim.onnx"
SAMPLE_IMAGE = DATA_DIR / "sample.jpg"
REFERENCE_NPZ = RESULTS_DIR / "pytorch_reference.npz"

# ── benchmarking ────────────────────────────────────────────────────────────────
WARMUP_ITERS = 200
BENCH_ITERS = 1000

# ── preprocessing (ImageNet statistics; mirrored in the C++ adapter) ─────────────
PIXEL_MEAN = [0.485, 0.456, 0.406]
PIXEL_STD = [0.229, 0.224, 0.225]


def engine_name(precision: str, trt_major: int, sm: str) -> str:
    """Self-documenting engine filename, e.g. rfdetr__fp16__trt10__sm87.engine."""
    return f"rfdetr__{precision}__trt{trt_major}__sm{sm}.engine"


def find_engine(precision: str) -> Path | None:
    """Resolve a built engine for a precision (newest match), or None."""
    matches = sorted(
        glob.glob(str(ARTIFACTS_DIR / f"rfdetr__{precision}__*.engine")),
        key=lambda p: Path(p).stat().st_mtime,
    )
    return Path(matches[-1]) if matches else None


def ensure_dirs() -> None:
    for d in (WEIGHTS_DIR, ARTIFACTS_DIR, DATA_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
