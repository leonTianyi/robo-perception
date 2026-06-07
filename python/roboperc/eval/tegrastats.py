"""Fold a tegrastats sidecar log into a bench JSON.

The C++ benchmark can't easily sample jtop from inside its timing loop, so we run
`tegrastats` alongside it (see scripts/export_and_build.sh / experiments) and merge
the numbers afterwards. Run: ``python -m roboperc.eval.tegrastats <log> <json>``.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

RAM_RE = re.compile(r"RAM (\d+)/(\d+)MB")
GR3D_RE = re.compile(r"GR3D_FREQ (\d+)%")
VDDIN_RE = re.compile(r"VDD_IN (\d+)mW")
RAIL_RE = re.compile(r"(VDD_\w+) (\d+)mW")


def parse(log_path: Path):
    ram, gpu, power = [], [], []
    for line in log_path.read_text(errors="ignore").splitlines():
        if (m := RAM_RE.search(line)):
            ram.append(float(m.group(1)))
        if (m := GR3D_RE.search(line)):
            gpu.append(float(m.group(1)))
        if (m := VDDIN_RE.search(line)):
            power.append(float(m.group(1)))
        elif "VDD_" in line:
            rails = RAIL_RE.findall(line)
            if rails:
                power.append(sum(float(v) for _, v in rails))
    return ram, gpu, power


def inject(log_path: Path, json_path: Path):
    if not log_path.exists():
        print(f"[tegra] {log_path} missing, skipping")
        return
    ram, gpu, power = parse(log_path)
    m = lambda x: float(np.mean(x)) if x else 0.0
    mx = lambda x: float(np.max(x)) if x else 0.0
    res = {"gpu_util_mean": m(gpu), "gpu_util_max": mx(gpu),
           "ram_used_mb_mean": m(ram), "ram_used_mb_max": mx(ram),
           "cpu_util_mean": 0.0, "power_mw_mean": m(power),
           "power_mw_max": mx(power), "n_samples": len(gpu),
           "source": "tegrastats"}
    data = json.loads(json_path.read_text()) if json_path.exists() else {}
    data["resources"] = res
    json_path.write_text(json.dumps(data, indent=2))
    print(f"[tegra] injected into {json_path.name}: gpu={res['gpu_util_mean']:.0f}% "
          f"power={res['power_mw_mean']:.0f}mW ram={res['ram_used_mb_max']:.0f}MB "
          f"({res['n_samples']} samples)")


def main():
    if len(sys.argv) < 3:
        print("usage: python -m roboperc.eval.tegrastats <tegra.log> <bench.json>")
        sys.exit(1)
    inject(Path(sys.argv[1]), Path(sys.argv[2]))


if __name__ == "__main__":
    main()
