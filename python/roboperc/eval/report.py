"""Aggregate every results/bench_*.json into one comparison table + CSV.

Run: ``python -m roboperc.eval.report``.
"""
from __future__ import annotations

import csv
import glob
import json
from pathlib import Path

from .. import config


def load_all():
    rows = []
    for fp in sorted(glob.glob(str(config.RESULTS_DIR / "bench_*.json"))):
        with open(fp) as f:
            d = json.load(f)
        lat = d.get("latency", {})
        res = d.get("resources", {})
        rows.append({
            "path": d.get("label", Path(fp).stem),
            "precision": d.get("precision", "-"),
            "mean_ms": lat.get("mean_ms", float("nan")),
            "p50_ms": lat.get("p50_ms", float("nan")),
            "p99_ms": lat.get("p99_ms", float("nan")),
            "fps": d.get("fps", float("nan")),
            "gpu_pct": res.get("gpu_util_mean", float("nan")),
            "mem_mb": res.get("ram_used_mb_max", float("nan")),
            "cpu_pct": res.get("cpu_util_mean", float("nan")),
            "power_w": (res.get("power_mw_mean", float("nan")) / 1000.0
                        if res.get("power_mw_mean") else float("nan")),
        })
    return rows


def _fmt(x, w, p=2):
    try:
        if x != x:  # NaN
            return f"{'-':>{w}}"
        return f"{x:>{w}.{p}f}"
    except Exception:
        return f"{str(x):>{w}}"


def main():
    rows = load_all()
    if not rows:
        print("[report] no results/bench_*.json found — run the benchmarks first")
        return
    rows.sort(key=lambda r: r["mean_ms"] if r["mean_ms"] == r["mean_ms"] else 1e9)

    hdr = (f"{'path':22s} {'prec':5s} {'mean':>8s} {'p50':>8s} {'p99':>8s} "
           f"{'FPS':>7s} {'GPU%':>6s} {'Mem(MB)':>8s} {'CPU%':>6s} {'Pow(W)':>7s}")
    print("\n" + "=" * len(hdr))
    print("robo-perception — RF-DETR benchmark summary")
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['path']:22s} {r['precision']:5s} "
              f"{_fmt(r['mean_ms'],8)} {_fmt(r['p50_ms'],8)} {_fmt(r['p99_ms'],8)} "
              f"{_fmt(r['fps'],7,1)} {_fmt(r['gpu_pct'],6,1)} "
              f"{_fmt(r['mem_mb'],8,0)} {_fmt(r['cpu_pct'],6,1)} "
              f"{_fmt(r['power_w'],7,2)}")
    print("=" * len(hdr))

    base = max(rows, key=lambda r: r["mean_ms"] if r["mean_ms"] == r["mean_ms"]
               else 0)
    print(f"\nspeedup vs {base['path']} ({base['mean_ms']:.2f} ms):")
    for r in rows:
        if r["mean_ms"] == r["mean_ms"] and r["mean_ms"] > 0:
            print(f"  {r['path']:22s} {base['mean_ms']/r['mean_ms']:.2f}x")

    csv_path = config.RESULTS_DIR / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[saved] {csv_path}")


if __name__ == "__main__":
    main()
