# RF-DETR TensorRT PoC — Project Overview

## Goal

Demonstrate end-to-end optimisation of an object-detection transformer (RF-DETR) on an
NVIDIA Jetson AGX Orin, going from a plain PyTorch model all the way to a compiled
TensorRT C++ binary, with rigorous latency/throughput/resource benchmarks at every step.

---

## The four stages

```
PyTorch (Python)  ──export──►  ONNX  ──parse──►  TRT engine  ──load──►  C++ runtime
     ▲ correctness baseline         ▲ portability          ▲ peak speed
```

| Stage | File(s) | Purpose |
|---|---|---|
| 1 – PyTorch baseline | `01_pytorch/` | Ground truth for correctness; first perf number |
| 2 – ONNX export | `02_export/` | Platform-neutral serialisation of the compute graph |
| 3 – TensorRT (Python) | `03_tensorrt/` | Validate the compiled engine still matches PyTorch |
| 4 – TensorRT (C++) | `04_cpp/` | Zero-Python overhead; production-representative path |

---

## Hardware context

- **SoC**: NVIDIA Jetson AGX Orin  
- **iGPU**: Ampere, 2048 CUDA cores  
- **CPU**: 12-core Arm Cortex-A78AE  
- **Memory**: 64 GB LPDDR5 *shared* between CPU and GPU (unified memory!)  
- **JetPack**: 6.2.2 → L4T 36.4.x, CUDA 12.6, TensorRT 10.3  

The shared memory architecture matters: there is **no PCIe transfer** between host and
device — `cudaMemcpy` is an in-DRAM move, much cheaper than on a discrete GPU.  
TensorRT's memory planning can therefore be more aggressive.

Clocks are locked via `nvpmodel -m 0` (MAXN) + `jetson_clocks` so benchmark numbers are
deterministic and represent peak Orin performance.

---

## Model: RF-DETR

RF-DETR (Roboflow Detection Transformer) is a real-time DETR variant built on:
- **DINOv2** (Vision Transformer, ViT-S/14 or ViT-B/14) as the backbone
- **RT-DETR-style decoder** with cross-attention query slots
- **No NMS needed** — the Hungarian-matching training produces one detection per query

The model is pre-trained on COCO (80 classes). Input is a 3×H×W normalised tensor.
Outputs are `pred_logits (B,300,80)` and `pred_boxes (B,300,4)` where boxes are
(cx,cy,w,h) in [0,1] normalised coordinates.

We use the **base** variant (ViT-S/14 backbone) which is the right size for real-time
inference on embedded hardware.

---

## Fixed design decisions

| Decision | Choice | Reason |
|---|---|---|
| Input resolution | 560 × 560 | Divisible by ViT patch size 14; balances speed and accuracy |
| Batch size | 1 | Latency mode (single-stream); throughput mode can be explored separately |
| ONNX opset | 17 | First opset with native `ScaledDotProductAttention` op support |
| TRT precision | FP16 | Orin Ampere has full FP16 throughput; INT8 needs calibration data |
| C++ preprocessing | OpenCV CPU | Avoids a CUDA kernel dependency for this PoC |

---

## Reading order for the docs

1. `01_model_and_export.md` — how the model is structured, what ONNX is, why export is tricky
2. `02_onnx_to_tensorrt.md` — deep dive on TensorRT concepts and transformer-specific pitfalls
3. `03_benchmarking_methodology.md` — what we measure and how to read the numbers
4. `04_environment_notes.md` — real findings from this Jetson + toolchain gotchas
5. `05_results_and_benchmarks.md` — **the results**: final numbers, analysis, every bug & fix
6. `06_team_share.md` — **presenter-facing** ~10-min team walkthrough (slides + speaker notes)
