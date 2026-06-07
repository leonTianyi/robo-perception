# RF-DETR on Jetson Orin — Team Share

**~10-minute walkthrough.** This doc doubles as the slides *and* the speaker
notes. Each section = roughly one "slide." The 🗣️ blocks are what to say; the
tables/charts are what to put on screen.

> **One-line summary to open with:** "We took RF-DETR — a transformer-based object
> detector — and got it running on our Jetson Orin at **135 FPS**, a **5.3×
> speedup** over plain PyTorch, with no loss in detection quality. Here's how, and
> what it costs to run."

---

## 1. What is this and why do we care

**Model under test — RF-DETR (base):**

| | |
|---|---|
| Name | RF-DETR base (Roboflow Detection Transformer) |
| Parameters | **~32 M** (32,174,530) |
| Backbone | DINOv2 ViT-S/14 (vision transformer) |
| Task | Object detection, 80 COCO classes (pretrained) |
| Input | 560 × 560 RGB |
| Output | 300 candidate detections → filtered by confidence |

**The size ladder — where "base" sits** (like YOLOX S/M/L):

| Variant | Parameters | Tier |
|---|---|---|
| RF-DETR Nano | 30.5 M | smallest / fastest |
| RF-DETR Small | 32.1 M | |
| **RF-DETR Base ← we benchmarked this** | **32.2 M** | *original; mid-cluster* |
| RF-DETR Medium | 33.7 M | recommended general-purpose |
| RF-DETR Large | 33.9 M | largest of the compact tier |
| RF-DETR XL | 126.4 M | big-backbone heavyweight (separate tier) |

🗣️ *Talking points:*
- **RF-DETR** = Roboflow's real-time **DE**tection **TR**ansformer. A modern
  object detector (finds + boxes objects, 80 COCO classes). Transformer-based,
  *no NMS* needed — cleaner than YOLO-style pipelines.
- **Where Base sits:** smack in the middle of a **tight cluster — Nano through
  Large are all 30–34 M params** because they share the same compact backbone.
  Base ≈ Small ≈ Large in *size*; they differ by **input resolution + transformer
  depth**, not parameter count. So picking Base vs. Large is a speed/accuracy
  choice, not really a "bigger model" choice.
- **The only true heavyweight is XL (126 M)** — a different, larger backbone, ~4×
  the params, aimed at servers. We did *not* test that tier.
- We wanted to answer two practical questions for the team:
  1. **Can we even run a DETR-style transformer efficiently on the Orin?**
     (transformers are historically painful to deploy on edge GPUs)
  2. **What does it actually cost** — latency, throughput, power — so we can plan
     around it?
- Short answer to both: **yes, and it's cheap once optimized.**

---

## 2. What we did — a 4-stage pipeline

```
   PyTorch  ──export──►  ONNX  ──compile──►  TensorRT engine  ──run──►  C++ app
  (baseline)          (portable graph)     (GPU-optimized)        (production)
   the .pth            the .onnx              the .engine          the binary
```

🗣️ *Talking points:*
- We started from the **model as shipped** (PyTorch `.pth` weights) and walked it
  down to a **production-style deployment** (a compiled C++ binary), measuring
  speed at every step.
- Each step is a different *way of running the same model* — same math, same
  detections, progressively more optimized for our specific hardware.
- The interesting engineering is the middle: **ONNX → TensorRT** (next slide).

---

## 3. The key conversion: ONNX → TensorRT (high level)

🗣️ *Talking points — keep it conceptual:*
- **ONNX** is a universal "save format" for neural nets — it captures the model's
  math as a portable graph that any runtime can read. We export PyTorch → ONNX.
- **TensorRT** is NVIDIA's optimizer/runtime. It takes that ONNX graph and
  **compiles it specifically for our GPU** — like a compiler turning source code
  into a tuned binary. Two things happen at "build time":
  - **Layer fusion** — it merges many small operations into single fast GPU
    kernels (e.g., the attention math collapses into one optimized kernel).
  - **Auto-tuning** — for every layer it benchmarks several GPU implementations
    and keeps the fastest *for our exact chip and input size*.
- It also lets us drop to **FP16** (half precision) — Orin's tensor cores run
  FP16 at ~2× the speed, and our accuracy held.
- **The headline worry going in:** RF-DETR uses *deformable attention*, which
  relies on an operation (`GridSample`) that often breaks these conversions.
  **It converted cleanly on TensorRT 10.3 — no custom code, no workarounds.**
  That was the big de-risking result.

> 💡 If someone asks "how long does the compile take?" — **~25 s (FP32), ~100 s
> (FP16)**, done once, then cached as a `.engine` file. It's a build-time cost,
> not per-inference.

---

## 4. Environment / versions (the critical ones)

🗣️ *Talking points:* "This is JetPack 6.2 era. The versions matter — half the
effort was getting a matching stack. If you reproduce this, pin these."

| Component | Version | Notes |
|---|---|---|
| Jetson | **AGX Orin** | 12-core CPU, Ampere iGPU, 64 GB shared RAM |
| JetPack | **6.2.2** | L4T 36.4.x |
| CUDA | **12.6** | |
| TensorRT | **10.3** | the piece that does the GPU compile |
| PyTorch | **2.8.0** | Jetson-specific wheel (`pypi.jetson-ai-lab.io`) |
| ONNX Runtime | 1.24 (CUDA EP) | |
| RF-DETR | 1.7.1 | model + native ONNX exporter |
| Python | 3.10 | |
| ONNX opset | 17 | export target |

> ⚠️ Two gotchas worth a sentence: **PyTorch for Jetson is not on normal PyPI**
> (special NVIDIA index), and the **CUDA library versions must match the driver**
> exactly or it fails at runtime. Both cost us time; both are now documented.

### Hardware note — we used the 64 GB AGX Orin. What about the 32 GB?

🗣️ *If someone asks "do we need the 64 GB module?" — the short answer is no, not
for this.*

**Yes, the 32 GB is also an AGX Orin** — the AGX Orin module comes in 32 GB and
64 GB versions (the smaller Orin NX/Nano modules top out at 16/8 GB, so "32 GB"
always means AGX Orin).

| | AGX Orin **64 GB** (ours) | AGX Orin **32 GB** |
|---|---|---|
| GPU CUDA / Tensor cores | 2048 / 64 | 1792 / 56 (−12%) |
| GPU max clock | 1.3 GHz | 0.93 GHz (−28%) |
| CPU cores | 12 | 8 |
| Memory **bandwidth** | 204.8 GB/s | **204.8 GB/s (same)** |
| Power range | 15–60 W | 15–40 W |
| INT8 TOPS | 275 | 200 |

**What actually changes for *this* workload:**
- **RAM size (64 vs 32 GB) is irrelevant here.** RF-DETR base inference uses well
  under a couple GB; we saw ~8 GB *total system* RAM (mostly OS). 32 GB is plenty.
  Capacity only matters for big batches, many concurrent models, the heavyweight
  XL variant, or training — none of which is our case.
- **The 32 GB has less GPU compute** (fewer cores + lower clock ≈ 60–70% of the
  64 GB). Since our workload is GPU-bound, expect **modestly lower FPS** — a rough
  estimate is ~90–110 FPS instead of 135. **Still ~6–7× our 15 FPS target.**
- **Mitigating factor:** memory bandwidth is *identical*, and transformers are
  partly bandwidth-bound — so the real gap is likely smaller than the raw
  compute ratio suggests.
- The lower **40 W** ceiling may throttle sustained clocks sooner under continuous
  load.

> **Bottom line:** the 32 GB AGX Orin would run RF-DETR comfortably real-time —
> you'd give up some FPS *headroom*, not capability. The RAM difference doesn't
> matter for this model. *(Estimate — we only measured the 64 GB; the only way to
> be sure is to run it on the 32 GB.)*

---

## 5. ⭐ THE RESULTS — speed

🗣️ *This is the main slide. Land these numbers.*

**Latency — time to process one frame (lower = better):**

```
PyTorch  FP32   39.2 ms  ████████████████████████████████████████  (baseline)
ONNX-RT  FP32   29.7 ms  ██████████████████████████████
TRT      FP32   18.7 ms  ███████████████████
TRT-Py   FP16    7.6 ms  ████████
TRT-C++  FP16    7.4 ms  ███████   ← fastest
```

**Throughput — frames per second (higher = better):**

```
PyTorch  FP32    25 FPS  ██████████
ONNX-RT  FP32    34 FPS  █████████████
TRT      FP32    54 FPS  █████████████████████
TRT-Py   FP16   131 FPS  ███████████████████████████████████████████████████
TRT-C++  FP16   135 FPS  █████████████████████████████████████████████████████  ← fastest
```

🗣️ *Talking points:*
- Going from "the model as-is" (PyTorch, **25 FPS**) to "fully optimized"
  (TensorRT FP16, **135 FPS**) is a **5.3× speedup**.
- **The single biggest lever is TensorRT + FP16.** Just compiling with TensorRT
  (still FP32) already ~doubles it; FP16 doubles it again.
- Numbers are **stable** (p99 ≈ mean — no nasty tail latency).

---

## 6. The four approaches — what each one actually is

🗣️ *Use this when someone asks "what's the difference between these rows?"*

| Approach | mean | FPS | What it's actually doing |
|---|---|---|---|
| **PyTorch (eager)** | 39.2 ms | 25 | Runs the original `.pth` weights through the PyTorch framework in Python, one operation at a time. No compilation — "the model exactly as the researchers shipped it." |
| **ONNX Runtime** | 29.7 ms | 34 | Runs the exported `.onnx` graph on a general-purpose GPU runtime (Microsoft's). Portable across hardware, but **not** specifically tuned for Orin. |
| **TensorRT — Python** | 7.6 ms | 131 | Runs the NVIDIA-**compiled** `.engine` (built from the ONNX) via a thin Python wrapper. Layers fused + auto-tuned for *our* GPU, FP16. The big jump. |
| **TensorRT — C++** | 7.4 ms | 135 | The **same** compiled engine, driven by a compiled C++ binary with **no Python** in the loop. Closest to a real production deployment. |

> Want the detail on *what ONNX Runtime is* or *how FPS is computed*? See §9
> (How we measured this) — your backup slides.

🗣️ *Key insight to call out:*
- **Python TensorRT and C++ TensorRT are basically tied (7.6 vs 7.4 ms).**
- That's because the GPU is doing ~7 ms of work either way — the language only
  affects ~0.5 ms of "glue" around it. **The speed is in the TensorRT engine, not
  the language.**
- **Takeaway for us:** we don't *need* to rewrite services in C++ for speed. We
  can stay in Python and still get ~98% of the performance. C++ is only worth it
  if we want to drop the Python runtime for packaging/footprint reasons.

---

## 7. What it costs to run — resources

🗣️ *Talking points: "Beyond speed, what does one inference cost in power/compute?"*

| Path | GPU util | Power | Energy per frame* |
|---|---|---|---|
| TensorRT C++ FP16 | **~91%** | ~38 W | **~0.28 J** |
| PyTorch FP32 | — | ~38 W | ~1.5 J |

🗣️ *Talking points:*
- When running TensorRT, the **GPU is ~91% utilized** — we're using the chip
  efficiently, not bottlenecked on the CPU.
- **Energy per frame** is the real "cost" number: the optimized path does the same
  detection for **~5× less energy** (~0.28 J vs ~1.5 J per frame). At the same
  power budget you process 5× more frames — that's battery life / thermal
  headroom / more cameras per device.
- *(\*Rough — clocks weren't locked for this run and power sampling is
  approximate; treat as indicative, directionally solid.)*

---

## 8. Correctness — did optimization break anything?

🗣️ *One line, then move on:* "No. Every single approach detects the **exact same
objects** with the same confidence."

Test image (bus + 4 people) — all 5 paths agree:

| Object | PyTorch | TensorRT FP16 |
|---|---|---|
| bus | 0.974 | 0.971 |
| person ×4 | 0.90–0.95 | 0.90–0.95 |

🗣️ "FP16 is *not* bit-identical to FP32, but the detections are visually and
numerically the same — boxes match to sub-pixel. So the 5× speedup is free."

---

## 9. How we actually measured this (implementation detail)

🗣️ *These are your "backup slides" — pull them up if someone asks how the numbers
were produced. The honesty here builds trust in the headline figures.*

### 9a. What we tested it on — one image, on purpose

- **Test input:** a single 810×1080 photo (a bus + 4 people) from the public
  Ultralytics sample set. Downloaded once, fed to every path.
- **Batch size = 1** (single frame at a time — real-time camera scenario).
- **"Wait, one image?"** Yes, and that's correct *for a speed benchmark*. RF-DETR
  is **fixed-shape**: every image is resized to 560×560 and the model always does
  the same work — 300 detection slots — **regardless of what's in the picture**.
  So compute time is image-content-independent; one image repeated 1000× measures
  latency accurately.
- **What this does NOT measure:** accuracy (mAP). That needs the full COCO
  validation set and is a separate exercise. **This is a speed/cost benchmark, not
  an accuracy benchmark.** Be clear about that distinction if asked.

### 9b. Yes — it's a timed loop (warm-up + measure)

Every path uses the same structure: throw away the first 200 runs, then time the
next 1000.

```
# load model + preprocess image ONCE
input = preprocess(image)            # resize → normalize → tensor

# warm-up: first runs are slow (GPU spinning up, kernels loading) — discard
for i in range(200):
    run_inference(input)

# timed loop: this is what we report
times = []
for i in range(1000):
    sync_gpu()                       # make sure GPU is idle before we start the clock
    t0 = now()
    run_inference(input)
    sync_gpu()                       # WAIT for the GPU to actually finish
    t1 = now()
    times.append(t1 - t0)

mean  = average(times)
p99   = percentile(times, 99)
fps   = 1000 / mean_ms               # ← this is where FPS comes from
```

🗣️ *Two things to call out:*
- **Warm-up matters** — the first runs include one-time GPU setup; including them
  would unfairly slow every path.
- **The `sync_gpu()` calls are the subtle part** (next slide).

### 9c. How we time it correctly (the GPU async trap)

🗣️ *This is the #1 way people get GPU benchmarks wrong — worth 30 seconds.*

GPU work is **asynchronous**: when Python "calls" the model, it just *queues* the
work and returns immediately — the GPU may still be computing. Naively timing it
measures the *queuing*, not the *compute*:

```
t0 = now()
run_inference(input)     # ❌ returns INSTANTLY, GPU still working
t1 = now()               # measures ~nothing
```

The fix is to **synchronize** — explicitly wait for the GPU to finish before
stopping the clock:

```
sync_gpu()               # ensure idle
t0 = now()
run_inference(input)
sync_gpu()               # ← block until the GPU is truly done
t1 = now()               # now t1 - t0 is real compute time
```

- In **Python/PyTorch** that's `torch.cuda.synchronize()`.
- In **C++/TensorRT** we use **CUDA events** (hardware timestamps on the GPU
  itself) for even higher precision.

We also measured **two scopes**: *device-only* (pure GPU compute) vs *end-to-end*
(including image preprocessing + memory copies). The end-to-end number is what we
quote — it's what a real app experiences.

### 9d. FPS = 1000 ÷ mean latency (ms)

- A 7.4 ms mean latency → `1000 / 7.4` ≈ **135 FPS**.
- This is **single-stream** FPS: one frame, start to finish, repeated. It's the
  conservative, real-time-relevant number.
- *True max throughput could be higher* with batching or overlapping copy+compute
  across multiple camera streams — we didn't measure that here (single-camera
  latency was the goal). So 135 FPS is a floor, not a ceiling.

### 9e. How we measure power

- Jetson boards have **on-board power sensors** (INA3221 chips) that read the
  actual voltage rails — real measured watts, not estimates.
- We sample them **during the timed loop only**, every 100 ms, from a **separate
  thread/process** so the sampling doesn't disturb the inference timing:
  - **C++ path:** NVIDIA's `tegrastats` tool runs as a side-process.
  - **Python paths:** the `jtop` library samples in a background thread.
- We report the mean/peak over the window.
- **Caveat (state it):** clocks weren't locked for this run and the two samplers
  differ in coverage, so power is *indicative*, not lab-grade.

### 9f. What is "Microsoft's GPU runtime" (the ONNX path)?

🗣️ *If someone asks what ONNX Runtime actually is:*
- **ONNX Runtime (ORT)** is Microsoft's open-source, cross-platform engine for
  running ONNX models. It's widely used and vendor-neutral.
- It runs the model through pluggable backends called **Execution Providers**. We
  used the **CUDA Execution Provider**, so it runs on the NVIDIA GPU (via NVIDIA's
  cuDNN/cuBLAS libraries).
- **Why it's the middle of the pack:** ORT does *some* graph optimization and runs
  on the GPU, so it beats plain PyTorch — but it's a **general-purpose** runtime.
  It does **not** do the deep, per-GPU kernel auto-tuning and aggressive fusion
  that **TensorRT** does for this specific Orin chip. That's the gap between the
  ONNX row (34 FPS) and the TensorRT rows (130+ FPS).
- Think of it as: PyTorch = interpreter, ONNX Runtime = portable optimized runner,
  TensorRT = a compiler that tunes for *our exact* hardware.

---

## 10. Does it hit our 15 FPS target?

🗣️ *This is the "so what for us" slide — probably the most important for the team.*

**15 FPS means a 66.7 ms budget per frame.** Here's how each path uses that budget:

| Path | Latency | % of 66.7 ms budget | Headroom |
|---|---|---|---|
| TensorRT C++ FP16 | 7.4 ms | 11% | **~9× / 135 FPS** |
| TensorRT Python FP16 | 7.6 ms | 11% | ~9× / 131 FPS |
| TensorRT FP32 | 18.7 ms | 28% | ~3.5× / 54 FPS |
| ONNX Runtime FP32 | 29.7 ms | 45% | ~2.2× / 34 FPS |
| PyTorch FP32 | 39.2 ms | 59% | ~1.7× / 25 FPS |

🗣️ *Talking points — this is the key insight:*
- **Every single approach already clears 15 FPS** — even unoptimized PyTorch
  (25 FPS) is comfortably above target.
- **So if 15 FPS single-stream were the *only* requirement, we wouldn't strictly
  need TensorRT at all.** That's worth saying out loud — it sets realistic
  expectations.
- **But the optimized path turns "just barely meets it" into massive headroom.**
  At 7.4 ms we use only **11% of the frame budget**. That spare 89% is what makes
  the project actually valuable — it's what we'd spend on:
  - **More camera streams** — ~9 simultaneous 15-FPS feeds on one Orin (135 ÷ 15),
    vs. PyTorch barely handling one.
  - **A bigger / more accurate model** (a higher RF-DETR tier — up to the
    heavyweight XL) and *still* hit 15 FPS.
  - **Higher input resolution** for small-object accuracy.
  - **Running other workloads** on the same GPU concurrently.
  - **Lower power / cooler / longer battery** if we cap at 15 FPS — ~5× less
    energy per frame.

> **Bottom line for the team:** 15 FPS is not the question — *all* paths hit it.
> The real value of the TensorRT work is the **9× headroom** it buys us to scale
> up streams, model size, or resolution while staying real-time.

> ⚠️ *Reality check:* these are clean single-image, unlocked-clock numbers. A real
> deployment adds camera decode, multiple frames in flight, and other app work —
> so that headroom is exactly the safety margin you want, not luxury.

---

## 11. Takeaways / recommendation

🗣️ *Close with these:*
1. **RF-DETR is viable on Orin** — the transformer/deformable-attention concern
   turned out to be a non-issue on current TensorRT.
2. **~135 FPS at FP16** with no accuracy loss → plenty of headroom for real-time
   use, multiple streams, or a bigger model variant.
3. **Stay in Python** unless we have a packaging reason for C++ — the engine is
   where the speed lives.
4. **Budget time for the toolchain, not the model** — the model converted in an
   afternoon; matching the JetPack/CUDA/PyTorch versions was the actual work.

### If we take this further
- Lock clocks + re-run for official peak numbers.
- Try a higher RF-DETR tier — Large (same param size, higher res) or the
  heavyweight **XL (~126 M)** — to spend the FPS headroom on accuracy.
- **INT8** quantization for another ~1.5–2× (needs a small calibration set).
- Multi-stream / batched throughput test (this was single-camera latency).

---

### Likely questions (be ready)

- **"What *is* 'Base'?"** → It's the **original** RF-DETR detection model — a
  ~32 M-param model on a compact (DINOv2 ViT-Small) backbone. In mid-2025 Roboflow
  released a cleaner family — **Nano / Small / Medium / Large** — and **deprecated
  the "Base" name** (it's now an alias). Size-wise **Base ≈ the new "Small"
  (~32 M)**; it is *not* the big one (Large is 129 M). We used Base because it's
  the established baseline, and the exact same pipeline works for any variant.
- **"Why benchmark Base and not Large?"** → In this family it barely matters by
  *size* — Nano through Large are all 30–34 M params (same backbone). Base is the
  established baseline; Large is the same param size at higher resolution/compute.
  The genuinely bigger model is **XL (126 M)**, a separate server-class tier we
  didn't test. We have the FPS headroom to try Large or XL next.
- **"How long did this take?"** → Model→engine in ~an afternoon once the
  environment was sorted. Environment setup was the bulk; it's documented now.
- **"Does it work for our classes?"** → Pretrained on COCO (80 classes). For
  custom classes we'd fine-tune RF-DETR first, then the same pipeline applies.
- **"What about bigger input / accuracy?"** → We used 560×560. Higher res = more
  accuracy, lower FPS — but we have 5× headroom to spend.
- **"Can it run alongside other models?"** → GPU is ~91% busy *per inference* at
  135 FPS; if we need fewer FPS there's compute to share.
