# ONNX → TensorRT: A Deep Dive

This is the most important document in the project. Read it before touching any engine
file, or when debugging a build failure.

---

## 1. What TensorRT actually does

TensorRT is **not** just a runtime. It is a two-phase system:

```
BUILD PHASE  (happens once, can take minutes)
─────────────────────────────────────────────
ONNX graph
    │  OnnxParser: translate ONNX ops → TRT INetworkDefinition
    ▼
INetworkDefinition (TRT's IR — a directed graph of ILayer objects)
    │  BuilderConfig: precision flags, workspace budget, profiles
    ▼
Builder::buildSerializedNetwork()
    │  Runs optimisation passes (see §3)
    ▼
Serialised engine (.engine file, ~tens of MB)


RUNTIME PHASE  (happens every inference call, microseconds overhead)
────────────────────────────────────────────────────────────────────
Engine bytes
    │  Runtime::deserializeCudaEngine()
    ▼
ICudaEngine  (the compiled plan, immutable)
    │  createExecutionContext()
    ▼
IExecutionContext
    │  setTensorAddress() for each I/O tensor
    │  enqueueV3(stream)
    ▼
CUDA kernels execute on GPU
```

The build phase is expensive by design — it is doing auto-tuning. You pay that cost once
and save the engine file. The runtime phase is just "launch pre-selected kernels".

---

## 2. The INetworkDefinition and layers

When OnnxParser reads your `.onnx`, it creates an `INetworkDefinition` containing one
`ILayer` per ONNX node. Not all ONNX nodes map 1:1 to a TRT layer:

- Simple ops (`MatMul`, `Add`, `Relu`): direct mapping
- Composite ops (`LayerNorm`, `GELU`, `SDPA`): TRT recognises these as multi-node
  patterns and replaces them with a single fused `IElementWiseLayer` or plugin
- **Unsupported ops**: OnnxParser emits a warning and either skips the node (breaking
  the graph) or falls back to a slow TRT "custom" fallback

If the parser encounters an op it cannot handle, you will see:

```
[E] [TRT] ModelImporter.cpp: ...While parsing node ... : No op registered for opName
```

That means you need to either:
a) Rework the export to avoid that op (preferred)
b) Write a TRT plugin

---

## 3. Optimisation passes during the build

The builder runs multiple optimisation stages. Key ones for transformer models:

### 3a. Layer fusion

TRT scans the graph for common patterns and collapses them into single, highly-optimised
kernels. Examples:

| Pattern | Fused into |
|---|---|
| Conv → BatchNorm → ReLU | Single `kCONVOLUTION` with activation |
| MatMul + Bias + (any activation) | `kFULLY_CONNECTED` |
| Q @ K.T / sqrt(d) + mask → Softmax → @ V | Flash-attention-style kernel (Ampere+) |
| LayerNorm (mean → var → norm → scale → shift) | Single `kNORMALIZATION` kernel |
| GELU (x * 0.5 * (1 + erf(x/√2))) | Single `kACTIVATION` kernel |

For a ViT model, the most impactful fusion is the **MHA fusion**. If TRT recognises the
QKV projection + attention + output projection as a single block, it uses a highly
optimised cuDNN Flash-Attention kernel instead of individual MatMuls.

Whether TRT *actually* fuses depends on:
- The exact op pattern in the ONNX graph (shapes of tensors matter)
- Whether you used eager vs fused SDPA during export (see `01_model_and_export.md`)
- TRT version (10.x has better ViT fusion than 8.x)

### 3b. Kernel auto-tuning (tactics)

For each layer, TRT benchmarks multiple CUDA kernel implementations ("tactics") and
selects the fastest one for *your specific GPU and input shape*. This is why:

- The engine file is **not portable** across GPU architectures
- Building on Orin produces a faster engine than cross-compiling on an x86 PC
- Build time is proportional to the number of layers × tactics per layer

### 3c. Memory planning

TRT analyses tensor lifetimes and assigns GPU memory ranges so that tensors that are
never alive simultaneously can share the same memory. This minimises peak VRAM.

On Orin (unified memory), this also reduces DDR bandwidth pressure.

### 3d. Constant folding

Pure-constant subgraphs (e.g., positional embeddings computed once at init time) are
evaluated during the build and their outputs are baked into the engine as constants.

---

## 4. Precision: FP32 vs FP16 vs INT8

### FP32

Default. Full accuracy, no transformations.

### FP16

Enable with `config->setFlag(BuilderFlag::kFP16)`.

TRT automatically converts supported layers to FP16. Layers that would lose too much
accuracy stay in FP32. You can inspect which layers ran in which precision with the
engine inspector (see §8).

On Orin's Ampere iGPU:
- FP16 matrix ops run at **2×** the FP32 throughput (tensor cores)
- Memory bandwidth effectively doubles since tensors are half the size
- Typical speedup for ViT models: **1.5–2×** over FP32

Watch out for: **attention score overflow**. The softmax in attention can receive very
large QK products. In FP32 this is fine; in FP16 the range is ±65504.
With 1600 tokens (560×560 / 14²), the unnormalised attention score for the largest entry
can approach `sqrt(384) ≈ 19.6 × max_token_value`. TRT handles this with KV scale
factors, but if you see NaN outputs in FP16, this is the likely cause.

### INT8

Enable with `config->setFlag(BuilderFlag::kINT8)`.

Requires a **calibration dataset** — TRT needs to observe the activation distributions
to choose optimal scale factors. Without calibration, INT8 quality degrades badly.
Not implemented in this PoC (we don't have a calibration dataset).

---

## 5. Static shapes vs dynamic shapes

### Why we use static shapes

RF-DETR's ViT backbone operates on a fixed grid of patches. Once the input resolution is
fixed (560×560), the number of tokens is fixed (1600), and every intermediate tensor has
a known shape at export time.

Static shapes let TRT:
- Select the single fastest kernel per layer (no need to handle multiple shapes)
- Pre-plan memory allocations without padding
- Avoid the overhead of the shape-inference engine at runtime

### If you needed dynamic shapes (FYI)

You would create an **optimisation profile** specifying min/opt/max shapes:

```cpp
auto profile = builder->createOptimizationProfile();
profile->setDimensions("images",
    OptProfileSelector::kMIN, Dims4{1,3,480,480});
profile->setDimensions("images",
    OptProfileSelector::kOPT, Dims4{1,3,560,560});
profile->setDimensions("images",
    OptProfileSelector::kMAX, Dims4{1,3,640,640});
config->addOptimizationProfile(profile);
```

TRT would then build separate kernel tactics for each shape "region" and switch at
runtime. This is slower to build and slightly slower at runtime.

---

## 6. Transformer-specific gotchas

### 6a. `ScaledDotProductAttention` (SDPA)

TRT 10.3 status:

| Export mode | ONNX node | TRT behaviour |
|---|---|---|
| `eager` decomposition | `MatMul + Mul + Softmax + MatMul` | Always works; TRT may fuse into MHA |
| fused SDPA export | `com.microsoft::ScaledDotProductAttention` | Parsed but fusion is model-dependent |

The safe default is eager. If you want to experiment with the fused op, set
`USE_FUSED_SDPA=True` in `02_export/export_onnx.py` and compare engine build logs.

### 6b. `LayerNorm`

ONNX exports LayerNorm as `ReduceMean → Sub → Pow → ReduceMean → Add(eps) → Sqrt → Div → Mul(γ) → Add(β)`.
TRT 10 recognises this 8-node pattern and fuses it into a single `kNORMALIZATION` layer.
If onnx-simplifier changes the pattern (e.g., adds a `Cast`), the fusion might not fire.

### 6c. Einsum / reshapes around attention

Many ViT implementations use `einsum` for QKV projections. PyTorch's ONNX exporter
lowers `einsum` to `Transpose + MatMul + Reshape`. Verify in `inspect_onnx.py` output —
you should see `Gemm` or `MatMul` nodes, not raw `Einsum` (ONNX `Einsum` opset 12
exists but TRT's parser is pickier about string-format equations).

### 6d. Positional embeddings

DINOv2 uses interpolated positional embeddings. The interpolation logic contains
`F.interpolate` which becomes `Resize` in ONNX. TRT supports `Resize` (bilinear,
nearest) at static shapes. Issues arise only if the PE grid size is computed
dynamically from the input shape — with our fixed 560×560 export, this is pre-folded
by onnx-simplifier.

### 6e. Non-contiguous view / reshape

Some PyTorch code does `.view()` on non-contiguous tensors. The ONNX exporter inserts a
`Transpose` + `Reshape` pair. Normally harmless, but occasionally this breaks TRT's
fusion pattern recognition. If you see lower-than-expected speed, check the engine
inspector for unexpectedly fragmented attention blocks.

---

## 7. The engine file

The `.engine` file is:
- A serialised `ICudaEngine` object
- **GPU-specific**: built for Orin Ampere, will crash or produce wrong results on a
  different GPU
- **TRT-version-specific**: an engine built with TRT 10.3 must be loaded with TRT 10.3
- Typically 50–200 MB for a ViT-S/14-based model
- Safe to cache and re-use as long as GPU and TRT version don't change

To verify what's inside an engine without loading it fully:

```bash
trtexec --loadEngine=models/rfdetr_base_fp16.engine --dumpLayerInfo --dumpProfile
```

---

## 8. Debugging a failed engine build

Step-by-step approach:

1. **Check the parser log first** — look for `[E]` lines, note the ONNX node name
2. **Open in Netron** — find that node in the graph, understand its inputs/outputs
3. **Run polygraphy** (if installed) for a more detailed failure report:
   ```bash
   polygraphy run rfdetr_base.onnx --trt --save-engine debug.engine -v
   ```
4. **Reduce the model** — use `onnx.utils.extract_model()` to isolate the failing
   subgraph and test it independently
5. **Check op support table** at
   https://onnx.tensorrt.com/supported_ops.html for your TRT version

Common errors and fixes:

| Error | Likely cause | Fix |
|---|---|---|
| `No implementation found for layer` | Op genuinely unsupported | Rewrite in eager mode or add plugin |
| `Assertion failed: n->inputs().size() == X` | Parser version mismatch | Update onnx-tensorrt or change opset |
| `Invalid dimensions` | Shape inference failure on dynamic dim | Fix shapes or add explicit reshape |
| `Explicit batch with implicit batch model` | Old-style ONNX with batch dim in shape | Ensure opset ≥ 11, remove `setFlag(EXPLICIT_BATCH)` in old code |

---

## 9. TRT 10 API vs older versions

If you read online TRT examples, most were written for TRT 8.x and use deprecated calls.
The key changes in TRT 10.x:

| TRT 8/9 (old) | TRT 10 (current) |
|---|---|
| `builder->createNetworkV2(1 << EXPLICIT_BATCH)` | `builder->createNetworkV2(0)` — explicit batch is always on |
| `builder->buildEngineWithConfig(*net, *cfg)` returns `ICudaEngine*` | `builder->buildSerializedNetwork(*net, *cfg)` returns `IHostMemory*` |
| `context->executeV2(bindings_array)` | `context->setTensorAddress(name, ptr)` + `context->enqueueV3(stream)` |
| `engine->getNbBindings()` | `engine->getNbIOTensors()` |
| `engine->getBindingName(i)` | `engine->getIOTensorName(i)` |
| `engine->getBindingIndex(name)` | No equivalent — iterate `getNbIOTensors()` |
| `engine->getBindingDimensions(i)` | `engine->getTensorShape(name)` |

Our C++ code uses the TRT 10 API throughout.
