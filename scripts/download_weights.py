#!/usr/bin/env python3
"""Download model weights from weights/models.yaml and verify sha256 (STUB).

RF-DETR currently self-downloads its checkpoint via the `rfdetr` package on first
use, so this is not yet required. It exists to enforce the artifact policy once we
host our own / fine-tuned weights: a manifest + checksum-verified download, never
committing binaries to git.

Usage (once implemented): python scripts/download_weights.py [--name rfdetr_base]
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = REPO_ROOT / "weights" / "models.yaml"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    if not MANIFEST.exists():
        print(f"[weights] no manifest at {MANIFEST}")
        sys.exit(1)
    print(f"[weights] manifest: {MANIFEST}")
    print("[weights] STUB — RF-DETR self-downloads via the `rfdetr` package for "
          "now. Implement manifest-driven download + sha256 verify when we host "
          "our own weights.")


if __name__ == "__main__":
    main()
