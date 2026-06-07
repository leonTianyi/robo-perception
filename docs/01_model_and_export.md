# The Model and ONNX Export

## 1. RF-DETR architecture recap

```
Image (560×560×3)
      │
      ▼
┌─────────────────────────────────────────────────────────────────┐
│  DINOv2 backbone (ViT-S/14)                                      │
│                                                                  │
│  Patch embed → 40×40 = 1600 tokens of dim 384                   │
│  12× Transformer blocks (MultiHeadAttn + MLP + LayerNorm)       │
│  Output feature map: (B, 1600, 384)                             │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
               Multi-scale feature projection
               (produces P3/P4/P5 feature pyramids)
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│  Decoder (6 layers of cross + self attention)                    │
│                                                                  │
│  300 learnable object queries                                    │
│  Each query attends to backbone features via cross-attention     │
└──────────────────────────────┬───────────────────────────────────┘
                               │
                 ┌─────────────┴──────────────┐
                 ▼                            ▼
          pred_logits                   pred_boxes
          (B, 300, 80)                  (B, 300, 4)
          raw class scores            cx,cy,w,h in [0,1]
```

Key property: the 300 output slots are **fixed-size always**, regardless of how many
objects are actually in the image. At post-processing time you threshold by confidence
and keep only the high-scoring detections.

---

## 2. What is ONNX?

**ONNX (Open Neural Network Exchange)** is a standardised, language-neutral format for
representing neural network compute graphs. Think of it as "the PDF of neural nets".

An ONNX model is a protobuf file containing:
- A list of **nodes** (ops like `MatMul`, `Softmax`, `LayerNorm`)
- **Initializers** (the weight tensors, stored as raw bytes)
- A **graph** connecting nodes by named tensors (edges)
- **Meta information**: opset version, shape annotations

You can open any `.onnx` file with [Netron](https://netron.app) to visualise it — highly
recommended to understand what TensorRT is actually seeing.

### Opset versions

The ONNX spec is versioned ("opsets"). Each version adds or modifies operator definitions.
We export at **opset 17** because that is the first version to formally define the
`ScaledDotProductAttention` operator (added in opset 17), which the ViT backbone uses.

| Opset | Notable additions relevant here |
|---|---|
| 11 | Many ops gain dynamic shape support |
| 13 | `LayerNorm`, `GroupNorm` become first-class ops |
| 17 | `ScaledDotProductAttention` (SDPA) |
| 18 | `GroupQueryAttention` |

---

## 3. How `torch.onnx.export` works

PyTorch's ONNX exporter uses **tracing**:

1. Feed a sample input tensor through the model
2. Record every ATen (PyTorch core) operation executed
3. Translate the recorded ops to their ONNX equivalents

This has one important implication: **the trace captures exactly one execution path**.
If the model has Python `if/else` branches that depend on tensor values at runtime,
only one branch is captured. DETR-family models generally don't have this problem
because they use fixed-size tensors throughout.

### The SDPA problem

PyTorch's `F.scaled_dot_product_attention` is a composite op that internally dispatches
to different kernels (Flash Attention, memory-efficient, or math). When tracing for ONNX:

- With `attn_implementation="sdpa"` (default): PyTorch emits the high-level SDPA node
  in the graph. TRT 10.3 *can* handle it, but kernel selection is less predictable.
- With `attn_implementation="eager"`: PyTorch decomposes SDPA into primitive ops
  `(Q @ K.T / sqrt(d)) → softmax → @ V`. This is more verbose but universally
  supported across all ONNX consumers.

We default to the **eager decomposition** path for maximum TRT compatibility, with a
flag to try the fused SDPA op if you want to experiment.

---

## 4. Export wrapper

The rfdetr high-level API wraps the model for convenience (PIL image input, supervision
output format). For ONNX export we need to bypass that and go directly to the
`torch.nn.Module.forward()` with a plain float tensor:

```python
class ExportWrapper(torch.nn.Module):
    def __init__(self, inner_model):
        super().__init__()
        self.inner = inner_model

    def forward(self, pixel_values: torch.Tensor):
        # pixel_values: (B, 3, H, W), fp32, ImageNet-normalised
        out = self.inner(pixel_values)
        # Return as a tuple so ONNX gets two named outputs
        return out["pred_logits"], out["pred_boxes"]
```

We export with explicit `input_names` / `output_names` and **fixed** `dynamic_axes={}`.
Fixed shapes produce a simpler ONNX graph with concrete `dim_param` values, which
TensorRT can reason about without needing an optimisation profile.

---

## 5. Why post-processing stays outside the ONNX graph

We intentionally **exclude** the sigmoid + box decode + threshold filter from the ONNX
graph. Reasons:

1. `NonMaxSuppression` and `TopK` with dynamic output sizes make TRT require an explicit
   dynamic-shape profile and add shape-inference complexity.
2. The C++ benchmark becomes simpler: the engine outputs two fixed tensors, and the host
   applies cheap CPU post-processing.
3. Correctness comparison is easier: both PyTorch and TRT see the exact same raw logits.

For a production deployment you could bake post-processing into the graph (TRT has a
`batchedNMSPlugin`) — but for this PoC separating concerns is cleaner.

---

## 6. The simplification step (onnx-simplifier)

After export, `onnxsim` runs a constant-folding + shape-inference pass. This:
- Collapses chains of pure-constant reshapes, gathers, slices into their final values
- Annotates every tensor edge with a concrete shape (helps TRT's shape inference)
- Removes dead nodes

The simplifier can occasionally change the op structure enough to expose previously
hidden unsupported patterns, so `inspect_onnx.py` runs on both the raw and simplified
graphs and diffs the op lists.
