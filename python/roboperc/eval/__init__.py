"""Evaluation harness — where Python/C++ parity is measured, not assumed.

- ``benchmark`` : in-process latency (+ jetson-stats resources) per runtime.
- ``tegrastats``: fold a tegrastats sidecar log into a bench JSON (C++ path).
- ``report``    : aggregate every bench_*.json into one comparison table.
- ``detection`` : accuracy — correctness vs the PyTorch reference + mAP (stub).
"""
