# Benchmarking Methodology

How we measure, why we measure it that way, and how to read the numbers without fooling
yourself.

---

## 1. What we measure

For each of the four inference paths we report:

### Latency (per single inference, batch=1)
- **mean** — average wall-clock time over all timed iterations
- **p50** (median) — the typical case; robust to outliers
- **p99** — the tail; 1 in 100 inferences is slower than this. Critical for real-time
  systems with a frame deadline.
- **std** — spread; high std means inconsistent timing (thermal throttling, scheduling
  jitter, GC pauses)

### Throughput
- **FPS** — frames per second. For batch=1 single-stream this is `1000 / mean_latency_ms`.
  (With batching or multiple streams it can exceed that — not explored in this PoC.)

### Resource use (sampled during the timed window)
- **GPU utilisation %** — how busy the iGPU is
- **GPU memory** — peak allocation (engine + activations + I/O buffers)
- **CPU %** — host-side load (preprocessing, launch overhead)
- **RAM** — system memory (on Orin this overlaps with GPU mem due to unified memory)
- **Power (mW)** — total SoC power draw via the onboard INA3221 sensors

---

## 2. Why warm-up matters

The first N inferences are **not representative**:

| Cold-start cost | Affects |
|---|---|
| CUDA context creation | First CUDA call (~100s of ms) |
| Kernel module loading (JIT/cuModuleLoad) | First use of each kernel |
| cuDNN/cuBLAS algorithm selection caches | First few inferences |
| GPU clock ramp-up (DVFS) | First ~dozen inferences |
| Memory allocator warm-up | First allocations |

We discard the first **200 iterations** (`WARMUP_ITERS`) before timing anything. This is
deliberately generous; on a clocks-locked Orin the model stabilises within ~50, but the
extra margin costs nothing and removes doubt.

---

## 3. Why clocks are locked

DVFS (Dynamic Voltage and Frequency Scaling) makes the GPU/CPU clocks vary with load and
temperature. For benchmarking this is poison — the same code gets different numbers
depending on what ran before it.

```bash
sudo nvpmodel -m 0      # MAXN power mode: all cores, max clock ceiling
sudo jetson_clocks      # pin clocks to their maximum, disable DVFS
```

With clocks locked, run-to-run variance drops to single-digit percent and the numbers
reflect **peak achievable performance**, which is what you want for a capability PoC.

Caveat: locked clocks mean the SoC runs hot. The `power (mW)` numbers will be at the high
end. If you instead want a "typical deployment" number, benchmark in a balanced
nvpmodel mode — but then report that you did.

---

## 4. The timing loop (and a subtle GPU pitfall)

GPU kernels launch **asynchronously**. This code is WRONG:

```python
t0 = time.perf_counter()
output = model(input)          # returns immediately, kernel still running!
t1 = time.perf_counter()       # measures launch overhead, not compute
```

You must synchronise before stopping the clock:

```python
torch.cuda.synchronize()       # ensure GPU is idle before we start
t0 = time.perf_counter()
output = model(input)
torch.cuda.synchronize()       # WAIT for the kernel to actually finish
t1 = time.perf_counter()       # now t1-t0 is real compute time
```

The same applies in C++/TensorRT: after `enqueueV3(stream)` you must
`cudaStreamSynchronize(stream)` before reading the clock.

For TensorRT we additionally use **CUDA events** (`cudaEventRecord` /
`cudaEventElapsedTime`) which measure GPU-side time directly and are more precise than
host wall-clock for the device portion.

---

## 5. What's included in "latency"

We measure two scopes to separate compute from overhead:

| Scope | Includes | Why |
|---|---|---|
| **device** | Engine execution only (between two CUDA events) | Pure model compute — comparable across paths |
| **end-to-end** | preprocess + H2D copy + execute + D2H copy + postprocess | What the application actually experiences |

On Orin's unified memory the H2D/D2H copies are cheap, but preprocessing (resize +
normalise) is non-trivial on the CPU side and shows up clearly in the gap between the two
scopes. This gap is itself an interesting result — it tells you whether the bottleneck is
the model or the data plumbing.

---

## 6. Resource sampling

A background thread (Python) or sidecar process (`tegrastats`) samples resource counters
every **100 ms** during the timed window only. We then report the mean and peak.

Sources:
- **jtop / jetson-stats** Python API: structured access to GPU%, mem, power rails,
  per-core CPU, temperatures. Preferred when available.
- **tegrastats**: the lower-level NVIDIA tool. We parse its stdout as a fallback and for
  the C++ path (where we don't want a Python dependency in the timing loop).

The sampler runs in a separate thread/process so it does not perturb the inference timing.

---

## 7. Correctness verification (not a perf metric, but essential)

Speed is meaningless if the answer is wrong. Before trusting any benchmark we verify each
path produces the same detections as the PyTorch baseline:

1. Run the same input image through PyTorch → save `pred_logits`, `pred_boxes`
2. Run through ONNX Runtime → compare raw output tensors
3. Run through TRT (Python) → compare raw output tensors
4. Run through TRT (C++) → compare final detections

Tolerances:
- **FP32 path**: `max_abs_diff < 1e-3` (numerical noise from op reordering)
- **FP16 path**: `max_abs_diff < 5e-2` on logits; more importantly the **set of detected
  boxes above threshold should match** (IoU > 0.95, same classes). FP16 will not be
  bit-exact and that is fine — what matters is that detections don't move or disappear.

A useful sanity metric: count detections above `CONF_THRESHOLD` and check they agree.

---

## 8. How to read the final report

The `report.py` table will look like:

```
Path           mean(ms)  p50(ms)  p99(ms)   FPS    GPU%   Mem(MB)  Pow(W)
PyTorch FP32     ...       ...      ...      ...    ...     ...      ...
ONNX-RT FP32     ...       ...      ...      ...    ...     ...      ...
TRT-Py FP16      ...       ...      ...      ...    ...     ...      ...
TRT-C++ FP16     ...       ...      ...      ...    ...     ...      ...
```

Expected ordering (fastest → slowest): **TRT-C++ ≈ TRT-Py < ONNX-RT < PyTorch**.

What each comparison teaches you:
- **PyTorch vs ONNX-RT**: cost of the Python eager-mode framework overhead
- **ONNX-RT vs TRT**: value of ahead-of-time compilation + kernel fusion + FP16
- **TRT-Py vs TRT-C++**: the Python interpreter / binding overhead per inference
  (often surprisingly small if the GPU is the bottleneck — a key finding either way)

If TRT-C++ is barely faster than TRT-Py, that tells you the workload is GPU-bound and
Python overhead is negligible — a legitimate and useful conclusion for deciding whether a
C++ rewrite is worth it in production.
