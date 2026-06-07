"""One-off sample image fetch for demos/benchmarks (not a dataset loader — that is
io/datasets when it lands). Downloads a stable street scene with people + a bus.
"""
from __future__ import annotations

from pathlib import Path

from . import config

SAMPLE_URLS = [
    "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/bus.jpg",
    "http://images.cocodataset.org/val2017/000000039769.jpg",
]


def ensure_sample_image() -> Path:
    p = config.SAMPLE_IMAGE
    if p.exists() and p.stat().st_size > 0:
        return p
    p.parent.mkdir(parents=True, exist_ok=True)
    import requests

    last_err = None
    for url in SAMPLE_URLS:
        try:
            print(f"[download] {url} -> {p}")
            r = requests.get(url, timeout=30, allow_redirects=True)
            r.raise_for_status()
            p.write_bytes(r.content)
            if p.stat().st_size > 0:
                return p
        except Exception as e:  # try the next mirror
            last_err = e
            print(f"[download] failed ({e}); trying next")
    raise RuntimeError(f"could not download a sample image: {last_err}")


if __name__ == "__main__":
    print(ensure_sample_image())
