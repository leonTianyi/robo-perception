"""Build a serialized TensorRT engine from the ONNX graph — the conversion the
whole capability revolves around.

Writes the engine under a self-documenting name (per the README artifact policy),
e.g. ``rfdetr__fp16__trt10__sm87.engine``, plus a sibling ``build_manifest.json``
recording exactly how it was built (an .engine is a non-portable cache bound to
GPU arch + TRT + JetPack + flags).

Run: ``python -m roboperc.export.build_engine [--fp16|--fp32] [--onnx PATH]``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import tensorrt as trt

from .. import config

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sm() -> str:
    """Compute capability as 'NN' (e.g. '87' for Orin), best-effort."""
    try:
        from cuda.bindings import runtime as cudart
    except ImportError:
        try:
            from cuda import cudart
        except ImportError:
            return "unknown"
    try:
        err, props = cudart.cudaGetDeviceProperties(0)
        return f"{props.major}{props.minor}"
    except Exception:
        return "unknown"


def _jetpack() -> str:
    p = Path("/etc/nv_tegra_release")
    if p.exists():
        return p.read_text(errors="ignore").splitlines()[0].strip()
    return "unknown"


def build(onnx_path: Path, precision: str, workspace_gb: float = 4.0) -> Path:
    fp16 = precision == "fp16"
    trt_major = int(trt.__version__.split(".")[0])
    sm = _sm()
    engine_path = config.ARTIFACTS_DIR / config.engine_name(precision, trt_major, sm)

    print(f"[build] TensorRT {trt.__version__}  precision={precision}  sm={sm}")
    print(f"[build] onnx   : {onnx_path}")
    print(f"[build] engine : {engine_path}")

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(0)   # TRT 10: explicit batch always on
    parser = trt.OnnxParser(network, TRT_LOGGER)

    print("[build] parsing ONNX...")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            print(f"[build] PARSE FAILED — {parser.num_errors} error(s):")
            for i in range(parser.num_errors):
                print(f"   [{i}] {parser.get_error(i)}")
            sys.exit(1)
    print(f"[build] parsed OK: {network.num_layers} layers, "
          f"{network.num_inputs} inputs, {network.num_outputs} outputs")
    for i in range(network.num_inputs):
        t = network.get_input(i)
        print(f"   input  {i}: {t.name} {t.shape} {t.dtype}")
    for i in range(network.num_outputs):
        t = network.get_output(i)
        print(f"   output {i}: {t.name} {t.shape} {t.dtype}")

    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,
                              int(workspace_gb * (1 << 30)))
    if fp16:
        if not builder.platform_has_fast_fp16:
            print("[build] WARNING: platform reports no fast FP16")
        cfg.set_flag(trt.BuilderFlag.FP16)

    # Static shapes -> no optimisation profile needed unless a dim is dynamic.
    if any(-1 in tuple(network.get_input(i).shape)
           for i in range(network.num_inputs)):
        print("[build] dynamic input dim detected — adding opt profile")
        profile = builder.create_optimization_profile()
        for i in range(network.num_inputs):
            t = network.get_input(i)
            shape = [config.INPUT_H if d == -1 else d for d in t.shape]
            profile.set_shape(t.name, shape, shape, shape)
        cfg.add_optimization_profile(profile)

    print("[build] building serialized engine (this can take minutes)...")
    t0 = time.perf_counter()
    serialized = builder.build_serialized_network(network, cfg)
    build_s = time.perf_counter() - t0
    if serialized is None:
        print("[build] BUILD FAILED (build_serialized_network returned None)")
        sys.exit(1)

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(serialized)

    manifest = {
        "model": "rfdetr",
        "variant": config.MODEL_VARIANT,
        "precision": precision,
        "input_shape": [1, 3, config.INPUT_H, config.INPUT_W],
        "engine": engine_path.name,
        "onnx": str(onnx_path.relative_to(config.REPO_ROOT))
                if onnx_path.is_relative_to(config.REPO_ROOT) else str(onnx_path),
        "onnx_sha256": _sha256(onnx_path),
        "tensorrt_version": trt.__version__,
        "sm": sm,
        "jetpack": _jetpack(),
        "workspace_gb": workspace_gb,
        "build_seconds": round(build_s, 1),
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = engine_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2))

    mb = engine_path.stat().st_size / 1e6
    print(f"\n[build] SUCCESS in {build_s:.1f}s  ->  {engine_path}  ({mb:.1f} MB)")
    print(f"[build] manifest: {manifest_path}")
    print(f"\nInspect with: trtexec --loadEngine={engine_path} "
          "--dumpLayerInfo --dumpProfile")
    return engine_path


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--fp16", action="store_true")
    g.add_argument("--fp32", action="store_true")
    ap.add_argument("--onnx", type=Path, default=None,
                    help="defaults to simplified onnx if present, else raw")
    ap.add_argument("--workspace", type=float, default=4.0)
    args = ap.parse_args()

    precision = "fp32" if args.fp32 else "fp16"
    onnx_path = args.onnx or (config.ONNX_SIM_PATH if config.ONNX_SIM_PATH.exists()
                              else config.ONNX_PATH)
    if not onnx_path.exists():
        print(f"[error] {onnx_path} not found — run the export stage first")
        sys.exit(1)
    build(onnx_path, precision, workspace_gb=args.workspace)


if __name__ == "__main__":
    main()
