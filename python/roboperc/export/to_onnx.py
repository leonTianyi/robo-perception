"""Export RF-DETR to ONNX.

Two strategies, tried in order:
  1. rfdetr's NATIVE .export() — preferred (handles the DETR export quirks).
  2. manual torch.onnx.export fallback — educational / when native is unavailable.

Run: ``python -m roboperc.export.to_onnx [--opset 17] [--manual]``.
"""
from __future__ import annotations

import argparse
import shutil
import sys

import torch

from .. import config

# Emit decomposed (eager) attention rather than a fused SDPA op — the safe default
# for TensorRT 10.3.
USE_FUSED_SDPA = False


def export_native(opset: int) -> bool:
    try:
        from rfdetr import RFDETRBase, RFDETRLarge
    except Exception as e:
        print(f"[native] rfdetr import failed: {e}")
        return False

    Cls = RFDETRBase if config.MODEL_VARIANT == "base" else RFDETRLarge
    api = Cls(resolution=config.INPUT_H)

    out_dir = config.ARTIFACTS_DIR / "native_export"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        api.export(output_dir=str(out_dir), opset_version=opset)
    except TypeError:
        api.export(output_dir=str(out_dir))
    except Exception as e:
        print(f"[native] export() raised: {e}")
        return False

    produced = sorted(out_dir.glob("*.onnx"), key=lambda p: p.stat().st_size)
    if not produced:
        print("[native] no .onnx produced")
        return False
    shutil.copy(produced[-1], config.ONNX_PATH)  # largest = full model
    print(f"[native] exported {produced[-1].name} -> {config.ONNX_PATH}")
    return True


def export_manual(opset: int) -> bool:
    from ..runtime.pytorch import load_raw_module

    print(f"[manual] loading model (fused_sdpa={USE_FUSED_SDPA})")
    if not USE_FUSED_SDPA:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)

    _, raw = load_raw_module()
    raw.eval()

    class ExportWrapper(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, pixel_values):
            out = self.inner(pixel_values)
            if isinstance(out, dict):
                return out["pred_logits"], out["pred_boxes"]
            return out[0], out[1]

    wrapper = ExportWrapper(raw).eval().cuda()
    dummy = torch.randn(1, 3, config.INPUT_H, config.INPUT_W, device="cuda")
    print(f"[manual] exporting opset {opset} (static shapes) -> {config.ONNX_PATH}")
    torch.onnx.export(
        wrapper, dummy, str(config.ONNX_PATH),
        input_names=["images"], output_names=["pred_logits", "pred_boxes"],
        opset_version=opset, do_constant_folding=True, dynamic_axes=None)
    return config.ONNX_PATH.exists()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--manual", action="store_true",
                    help="skip native export, force the manual path")
    args = ap.parse_args()

    config.ensure_dirs()
    ok = False
    if not args.manual:
        ok = export_native(args.opset)
    if not ok:
        print("[export] falling back to manual torch.onnx.export")
        ok = export_manual(args.opset)
    if not ok:
        print("[export] FAILED")
        sys.exit(1)

    mb = config.ONNX_PATH.stat().st_size / 1e6
    print(f"\n[done] {config.ONNX_PATH}  ({mb:.1f} MB)")
    print("Next: python -m roboperc.export.inspect_onnx")


if __name__ == "__main__":
    main()
