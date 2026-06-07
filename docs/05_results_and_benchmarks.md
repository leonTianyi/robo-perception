# Results & Benchmarks

End-to-end results for the RF-DETR PoC on Jetson AGX Orin (JetPack 6.2.2, CUDA
12.6, TensorRT 10.3). This is the payoff document — what we measured, what it
means, and every problem we hit getting here.

> **Run date:** 2026-06-02. Model: RF-DETR **base** (DINOv2 ViT-S/14 backbone),
> 560×560 input, batch 1, single-stream latency mode. 200 warm-up + 1000 timed
> iterations per path.
>
> ⚠️ **Clocks were NOT locked for this run** (CPU governor was `schedutil`, not
> pinned via `jetson_clocks`). Numbers are stable and representative but should
> be treated as *indicative*. For peak/publication figures, re-run after
> `sudo nvpmodel -m 0 && sudo jetson_clocks`. The relative ordering and speedups
> are unaffected.

---

## 1. TL;DR

| Path | Precision | mean (ms) | p50 | p99 | FPS | Speedup |
|---|---|---|---|---|---|---|
| **TRT C++** | FP16 | **7.39** | 7.38 | 7.45 | **135.4** | **5.30×** |
| TRT Python | FP16 | 7.61 | 7.52 | 9.40 | 131.4 | 5.15× |
| TRT Python | FP32 | 18.70 | 18.70 | 18.83 | 53.5 | 2.09× |
| ONNX Runtime (CUDA EP) | FP32 | 29.70 | 29.68 | 30.05 | 33.7 | 1.32× |
| PyTorch (eager) | FP32 | 39.16 | 39.06 | 40.56 | 25.5 | 1.00× |

(End-to-end scope: preprocess + H2D + execute + D2H. Speedup vs the PyTorch
baseline.)

**Headline:** TensorRT FP16 takes RF-DETR from **25 FPS → 135 FPS** on Orin — a
**5.3× speedup** — while producing identical detections.

---

## 2. The central question: does a DETR-family model convert to TensorRT?

**Yes, cleanly, with no plugins.** This was the main risk of the whole project.

RF-DETR uses **deformable attention**, which exports to ONNX as the `GridSample`
operator (×3, one per relevant decoder layer). `GridSample` is historically the
op that breaks DETR→TensorRT conversions. On TensorRT 10.3 it parsed natively:

```
[TRT] Registering layer: /transformer/decoder/layers.0/cross_attn/GridSample
      for ONNX node: ...GridSample
[build] parsed OK: 3742 layers, 1 inputs, 2 outputs
[build] SUCCESS in 104.2s
```

The full op inventory (40 distinct types, 1489 ONNX nodes) parsed without a
single unsupported-operator error. The only two watch-list ops — `GridSample`
and a static-`k` `TopK` — both resolved to native TensorRT layers. **No custom
plugins, no graph surgery, no opset downgrade were needed.**

See `docs/02_onnx_to_tensorrt.md §6` for why these ops are usually risky.

---

## 3. What each comparison teaches

### PyTorch → ONNX Runtime (1.00× → 1.32×)
Dropping the Python eager-execution framework for a compiled ONNX graph on the
CUDA execution provider buys **32%**. This is pure framework-overhead removal;
both are FP32 and run the same math.

### ONNX Runtime → TensorRT FP32 (1.32× → 2.09×)
Same precision (FP32), but TensorRT's ahead-of-time **layer fusion + kernel
auto-tuning** for this exact GPU and shape adds another **1.6×**. This is the
value of TensorRT's build-phase optimisation, isolated from any precision change.

### TensorRT FP32 → FP16 (2.09× → 5.15×)
Switching the engine to FP16 gives a further **2.46×** (18.70 ms → 7.61 ms).
Orin's Ampere tensor cores run FP16 matmuls at ~2× throughput and halve memory
bandwidth pressure. This is the single biggest lever.

### TensorRT Python → C++ (5.15× → 5.30×)
The C++ runtime is only **~3% faster** than the Python TensorRT runtime
(7.39 vs 7.61 ms). That small gap is the key finding of section 4.

---

## 4. The workload is GPU-bound — Python overhead is negligible

We timed two scopes (see `docs/03_benchmarking_methodology.md §5`):

| Scope | TRT Python FP16 | TRT C++ FP16 |
|---|---|---|
| **device-only** (pure `enqueueV3`, CUDA-event timed) | 6.98 ms (143 FPS) | 6.92 ms |
| **end-to-end** (preprocess + copies + execute) | 7.61 ms | 7.39 ms |
| host-side overhead (e2e − device) | ~0.63 ms | ~0.46 ms |

Two conclusions:

1. **Device compute is ~6.9 ms regardless of language.** Both paths call the
   same TensorRT engine; the GPU doesn't care whether Python or C++ launched it.
2. **The language only affects the ~0.5 ms of host-side work** (preprocess +
   buffer copies). C++ shaves ~0.17 ms off that.

**Practical takeaway:** for this model on this hardware, a C++ rewrite buys ~3%.
If you already have a Python service, the TensorRT engine — not the language — is
where essentially all the speed is. A C++ port is worth it only if you need to
shave that last sub-millisecond or remove the Python runtime for deployment
reasons (footprint, packaging), not for throughput.

---

## 5. Correctness — every path verified against PyTorch

The PyTorch baseline is ground truth (`results/pytorch_reference.npz`). Test
image: a bus + 4 people. **All paths detect exactly the same 5 objects** with
near-identical confidence:

| Object | PyTorch | TRT Py FP16 | TRT C++ FP16 |
|---|---|---|---|
| bus | 0.974 | 0.971 | 0.971 |
| person | 0.950 | 0.950 | 0.950 |
| person | 0.942 | 0.941 | 0.941 |
| person | 0.941 | 0.941 | 0.941 |
| person | 0.903 | 0.903 | 0.903 |

**FP16 numerical accuracy** (on the queries that become detections):

```
max|Δlogits| = 0.17     max|Δboxes| = 0.0005   (normalised coords, sub-pixel)
```

### Important methodology note: don't trust the all-query max-diff

A naive max-abs-diff over **all 300 query slots** reports a scary
`max|Δlogits|=6.7, max|Δboxes|=1.08` — and even the FP32 ONNX path shows ~5.7.
This is **not** an error. RF-DETR always emits 300 fixed query slots; only ~5
exceed threshold. The other ~295 are *background slots* whose predictions are
unconstrained garbage — FP16 rounding (and even FP32 op-reorder differences)
diverge wildly there, but **nothing above threshold ever uses them**. The
meaningful metric is whether the above-threshold detections agree, and they do.
`03_tensorrt/infer_trt.py` was updated to report both and base its PASS/CHECK
verdict on the detections only.

---

## 6. Precision & artifact sizes

| Artifact | Size | Build time |
|---|---|---|
| `rfdetr_base.onnx` | 114.1 MB | — (native export) |
| `rfdetr_base_fp32.engine` | 109.5 MB | 24.9 s |
| `rfdetr_base_fp16.engine` | 58.7 MB | 104.2 s |

- FP16 halves the engine (weights stored as half-precision): 109.5 → 58.7 MB.
- FP16 builds **slower** (104 vs 25 s) — the auto-tuner evaluates more candidate
  kernels (FP16 *and* FP32 tactics) per layer.

---

## 7. Resource utilisation

Measured during the timed window. **Caveat:** two different samplers were used
and they have different coverage on this JetPack image (see §8 limitation).

| Path | GPU % | Power (W) | Notes |
|---|---|---|---|
| TRT C++ FP16 (tegrastats) | **90.7** | 37.7 | RAM 8.2 GB total system |
| TRT Python FP16 (jtop) | n/a* | 45.8 | *jtop GPU%/mem unavailable |
| TRT Python FP32 (jtop) | n/a* | 53.1 | |
| ONNX-RT FP32 (jtop) | n/a* | 49.9 | |
| PyTorch FP32 (jtop) | n/a* | 38.1 | |

- The C++ path (via `tegrastats`) shows the GPU pinned at **~91% utilisation**
  during inference — confirming the GPU-bound conclusion from §4.
- Power trend is sensible: FP32 TensorRT draws the most (53 W, most active
  compute); FP16 less (46 W). PyTorch's lower draw (38 W) reflects *under*-
  utilisation — it leaves the GPU idle between eager ops rather than being
  efficient.
- On Orin, CPU and GPU share LPDDR5, so "memory" is reported as total system RAM,
  not a discrete VRAM figure.

---

## 8. Every problem we hit (and the fix)

A faithful record — most of the effort was environment/tooling, **none** was the
model architecture itself.

| # | Problem | Root cause | Fix |
|---|---|---|---|
| 1 | `pkg-config opencv4` pointed at empty `/usr/local` | broken `.pc` prefix on the image | corrected `prefix=/usr` (`docs/04`) |
| 2 | `torch` install: "No matching distribution" | NVIDIA `jp/v62` index has no torch; old community host `*.dev` has dead DNS | install from `pypi.jetson-ai-lab.io/jp6/cu126`, pin torch 2.8.0 |
| 3 | `onnxruntime-gpu` not found | no aarch64 wheel on PyPI | pull from the Jetson `.io` index |
| 4 | `onnxsim` build failed | no aarch64 wheel; builds from source | `pip install cmake` (no sudo); made non-fatal |
| 5 | `torch` import crash: "Numpy is not available" | Jetson torch wheel built against numpy 1.x; rfdetr deps pulled numpy 2.x | pin `numpy==1.26.4` **last**; drop pip `opencv-python` |
| 6 | sample image download HTTP 308 | stale `ultralytics.com` URL, no redirect follow | redirect-following `requests` + stable URLs |
| 7 | detections mislabeled ("train"/"bicycle") | RF-DETR emits **91-class** DETR indexing (0=bg), not contiguous 80 | 91-class label table + argmax over `[1:]`, in Python **and** C++ |
| 8 | `BuilderFlag.kFP16` AttributeError | Python TRT enums drop the C++ `k` prefix | `trt.BuilderFlag.FP16` |
| 9 | `from cuda import cudart` ImportError | cuda-python moved bindings | `from cuda.bindings import runtime as cudart` |
| 10 | `cudaErrorInsufficientDriver (35)` | pip `cuda-python` 13.x targets CUDA 13; driver is 12.6 | pin `cuda-python==12.6.2.post1` |
| 11 | correctness "CHECK" false alarm | all-query max-diff dominated by background slots | verdict based on above-threshold detections |

All fixes are pinned into `setup.sh` / the code, so a fresh run is clean.

---

## 9. Reproduce

```bash
bash setup.sh                              # deps (all pins baked in)
sudo nvpmodel -m 0 && sudo jetson_clocks   # lock clocks (recommended!)

python3 01_pytorch/infer.py                # baseline + reference tensors
python3 02_export/export_onnx.py           # Roboflow native export
python3 02_export/inspect_onnx.py          # op scan / risk check
python3 03_tensorrt/build_engine.py --fp16
python3 03_tensorrt/build_engine.py --fp32
python3 03_tensorrt/infer_trt.py           # correctness check

bash 05_benchmark/run_remaining.sh         # all benchmarks + report
```

Outputs land in `results/` (`bench_*.json`, `summary.csv`, annotated `*.jpg`).

---

## 10. Conclusions

1. **RF-DETR runs on Jetson TensorRT with zero architectural friction.** The
   deformable-attention `GridSample` op — the usual DETR blocker — is native in
   TensorRT 10.3.
2. **FP16 TensorRT is the win:** 5.3× over PyTorch, 135 FPS, identical
   detections, half the disk footprint.
3. **The workload is GPU-bound;** Python vs C++ is a ~3% difference. Choose the
   language for deployment ergonomics, not speed.
4. **The real work was the toolchain,** not the model — 11 environment/version
   issues, all now documented and pinned.

### Sensible next steps (not done here)
- Lock clocks and re-run for peak figures.
- Try a higher-tier RF-DETR variant for an accuracy/latency trade. Note Nano→Large
  are all ~30–34 M params (same compact backbone; they differ by resolution +
  transformer depth); the only true heavyweight is **XL (~126 M)**.
- **INT8** with a small COCO calibration set (expect another ~1.5–2×).
- Bake NMS-free post-processing into the engine, or a CUDA preprocessing kernel
  to trim the ~0.5 ms host overhead.
- Batch / multi-stream throughput mode (this PoC measured single-stream latency).
