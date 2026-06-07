#!/usr/bin/env python3
"""Fetch the demo sample image into data/. Thin CLI over roboperc.sample."""
from roboperc.sample import ensure_sample_image

if __name__ == "__main__":
    print(ensure_sample_image())
