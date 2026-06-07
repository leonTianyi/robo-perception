"""Run every available RF-DETR backend on the sample image, checking correctness
against the PyTorch reference and benchmarking each, then print the report.

This is the comparison lever in action: same dataset, same (native) pre/post,
swap the runtime. Backends that aren't ready (no engine / onnx / deps) are
skipped with a note rather than failing the run.

Run: ``python -m roboperc.experiments.run_pipeline [--config configs/rfdetr_base.yaml]``.
Assumes clocks are locked for stable numbers: ``sudo nvpmodel -m 0 && sudo jetson_clocks``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .. import config, imageproc, sample
from ..eval import benchmark as bench
from ..eval import detection as acc
from ..eval import report


def _load_cfg(path: Path | None):
    """Optionally override config knobs from a YAML file (warmup/iters/conf)."""
    if path is None or not path.exists():
        return
    try:
        import yaml
    except ImportError:
        print("[cfg] pyyaml not installed — using built-in defaults")
        return
    data = yaml.safe_load(path.read_text()) or {}
    bm = data.get("benchmark", {})
    config.WARMUP_ITERS = bm.get("warmup", config.WARMUP_ITERS)
    config.BENCH_ITERS = bm.get("iters", config.BENCH_ITERS)
    config.CONF_THRESHOLD = data.get("model", {}).get("conf", config.CONF_THRESHOLD)
    print(f"[cfg] loaded {path}: warmup={config.WARMUP_ITERS} "
          f"iters={config.BENCH_ITERS} conf={config.CONF_THRESHOLD}")


def _bench_runner(runner, x, label, precision, *, device_scope=False):
    """Benchmark a runner's infer() (end-to-end); save JSON for the report."""
    sync = getattr(runner, "synchronize", None)
    if device_scope and hasattr(runner, "infer_device_only"):
        runner.load_input_once(x)
        dev = bench.run_benchmark(runner.infer_device_only,
                                  warmup=config.WARMUP_ITERS,
                                  iters=config.BENCH_ITERS, sync_fn=runner.synchronize,
                                  label=f"{label}_device")
        e2e = bench.run_benchmark(lambda: runner.infer(x),
                                  warmup=config.WARMUP_ITERS // 2,
                                  iters=config.BENCH_ITERS, sync_fn=runner.synchronize,
                                  sample_resources=False, label=f"{label}_e2e")
        out = {"label": label, "precision": precision,
               "device_scope": dev, "e2e_scope": e2e,
               "latency": e2e["latency"], "fps": e2e["fps"],
               "resources": dev["resources"]}
    else:
        r = bench.run_benchmark(lambda: runner.infer(x),
                                warmup=config.WARMUP_ITERS // 2,
                                iters=config.BENCH_ITERS, sync_fn=sync, label=label)
        r["precision"] = precision
        out = r
    bench.save_result(out, config.RESULTS_DIR / f"bench_{label}.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path,
                    default=config.REPO_ROOT / "configs" / "rfdetr_base.yaml")
    ap.add_argument("--skip-bench", action="store_true",
                    help="run inference + correctness only, no timing loops")
    args = ap.parse_args()

    _load_cfg(args.config)
    config.ensure_dirs()

    img_bgr = imageproc.decode(sample.ensure_sample_image())
    orig_h, orig_w = img_bgr.shape[:2]
    x = imageproc.preprocess(img_bgr)

    # ── 1. PyTorch baseline: reference + bench ────────────────────────────────
    print("\n=== PyTorch (baseline / reference) ===")
    try:
        from ..poc.rfdetr_infer import main as pt_main
        pt_main()  # writes reference npz + annotated image
        if not args.skip_bench:
            from ..runtime.pytorch import PyTorchRunner
            _bench_runner(PyTorchRunner(), x, "pytorch_fp32", "fp32")
    except Exception as e:
        print(f"[pytorch] skipped: {e}")

    # ── 2. ONNX Runtime (intermediate) ────────────────────────────────────────
    print("\n=== ONNX Runtime ===")
    try:
        from ..runtime.onnx import OnnxRunner
        ort = OnnxRunner()
        logits, boxes = ort.infer(x)
        acc.compare_to_reference(logits, boxes, "onnxrt_fp32")
        imageproc.draw_detections(
            img_bgr, imageproc.postprocess(logits, boxes, orig_w, orig_h),
            config.RESULTS_DIR / "onnxrt_detections.jpg")
        if not args.skip_bench:
            _bench_runner(ort, x, "onnxrt_fp32", "fp32")
    except Exception as e:
        print(f"[onnx] skipped: {e}")

    # ── 3. TensorRT (deployment) — both precisions if their engines exist ─────
    from ..runtime.tensorrt import TensorRTRunner
    for precision in ("fp16", "fp32"):
        engine = config.find_engine(precision)
        print(f"\n=== TensorRT {precision} ===")
        if engine is None:
            print(f"[trt:{precision}] no engine — run "
                  f"`python -m roboperc.export.build_engine --{precision}`")
            continue
        try:
            runner = TensorRTRunner(engine)
            logits, boxes = runner.infer(x)
            acc.compare_to_reference(logits, boxes, f"trt_{precision}")
            imageproc.draw_detections(
                img_bgr, imageproc.postprocess(logits, boxes, orig_w, orig_h),
                config.RESULTS_DIR / f"trt_{precision}_detections.jpg")
            if not args.skip_bench:
                _bench_runner(runner, x, f"trt_python_{precision}", precision,
                              device_scope=True)
            runner.close()
        except Exception as e:
            print(f"[trt:{precision}] skipped: {e}")

    # ── 4. Report ──────────────────────────────────────────────────────────────
    print("\n=== Report ===")
    report.main()


if __name__ == "__main__":
    main()
