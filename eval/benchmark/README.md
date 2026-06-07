# eval/benchmark

_Performance: latency (CUDA events) / mem / power (tegrastats), measured IN-PROCESS per runtime._

Two in-process implementations, one per language (never time the C++ path through
the Python binding):

| Path | Where | How |
|---|---|---|
| Python backends | `python/roboperc/eval/benchmark.py` | warm-up + timed loop, jetson-stats (jtop) resource sampling |
| C++ TensorRT | `apps/benchmark.cpp` | CUDA-event device scope + wall-clock e2e; emits the same JSON schema |
| tegrastats merge | `python/roboperc/eval/tegrastats.py` | fold a sidecar tegrastats log into a C++ bench JSON |
| aggregate | `python/roboperc/eval/report.py` | all `results/bench_*.json` → one table + `summary.csv` |

TensorRT is reported at two scopes: **device-only** (pure GPU compute, CUDA-event
timed) and **end-to-end** (H2D + execute + D2H). Lock clocks first for stable
numbers: `sudo nvpmodel -m 0 && sudo jetson_clocks`.
