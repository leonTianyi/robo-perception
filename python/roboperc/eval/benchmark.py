"""In-process latency + Jetson resource sampling. Used by every Python path so the
numbers are produced identically (mirrors apps/benchmark.cpp on the C++ side).
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np


@dataclass
class LatencyStats:
    mean_ms: float
    p50_ms: float
    p90_ms: float
    p99_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    n: int

    @classmethod
    def from_samples(cls, samples_ms) -> "LatencyStats":
        a = np.asarray(samples_ms, dtype=np.float64)
        return cls(float(a.mean()), float(np.percentile(a, 50)),
                   float(np.percentile(a, 90)), float(np.percentile(a, 99)),
                   float(a.std()), float(a.min()), float(a.max()), int(a.size))

    @property
    def fps(self) -> float:
        return 1000.0 / self.mean_ms if self.mean_ms > 0 else 0.0


@dataclass
class ResourceStats:
    gpu_util_mean: float = 0.0
    gpu_util_max: float = 0.0
    ram_used_mb_mean: float = 0.0
    ram_used_mb_max: float = 0.0
    cpu_util_mean: float = 0.0
    power_mw_mean: float = 0.0
    power_mw_max: float = 0.0
    n_samples: int = 0
    source: str = "none"


class ResourceSampler:
    """Background thread polling jtop. No-op if jetson-stats is unavailable."""

    def __init__(self, interval_s: float = 0.1):
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._gpu, self._ram, self._cpu, self._pow = [], [], [], []
        self._jtop = None
        self._source = "none"

    def _open_jtop(self):
        try:
            from jtop import jtop
            j = jtop()
            j.start()
            self._jtop = j
            self._source = "jtop"
        except Exception as e:
            print(f"[ResourceSampler] jtop unavailable ({e}); resource stats off")
            self._jtop = None

    def _sample_once(self):
        j = self._jtop
        if j is None or not j.ok():
            return
        try:
            gpu = j.gpu
            if isinstance(gpu, dict):
                first = next(iter(gpu.values()))
                load = first.get("status", {}).get("load", first.get("load", 0))
                self._gpu.append(float(load))
            mem = j.memory
            if isinstance(mem, dict) and "RAM" in mem:
                self._ram.append(float(mem["RAM"].get("used", 0)) / 1024.0)
            cpu = j.cpu
            if isinstance(cpu, dict) and "cpu" in cpu:
                loads = [c.get("user", 0) + c.get("system", 0)
                         for c in cpu["cpu"] if isinstance(c, dict)]
                if loads:
                    self._cpu.append(float(np.mean(loads)))
            pwr = j.power
            if isinstance(pwr, dict) and "tot" in pwr:
                self._pow.append(float(pwr["tot"].get("power", 0)))
        except Exception:
            pass

    def _run(self):
        self._open_jtop()
        while not self._stop.is_set():
            self._sample_once()
            time.sleep(self.interval_s)
        if self._jtop is not None:
            try:
                self._jtop.close()
            except Exception:
                pass

    def __enter__(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        time.sleep(0.5)
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def result(self) -> ResourceStats:
        m = lambda x: float(np.mean(x)) if x else 0.0
        mx = lambda x: float(np.max(x)) if x else 0.0
        return ResourceStats(
            gpu_util_mean=m(self._gpu), gpu_util_max=mx(self._gpu),
            ram_used_mb_mean=m(self._ram), ram_used_mb_max=mx(self._ram),
            cpu_util_mean=m(self._cpu),
            power_mw_mean=m(self._pow), power_mw_max=mx(self._pow),
            n_samples=max(len(self._gpu), len(self._pow)), source=self._source)


class _nullcontext:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def run_benchmark(infer_fn: Callable[[], None], *, warmup: int, iters: int,
                  sync_fn: Optional[Callable[[], None]] = None,
                  sample_resources: bool = True, label: str = "") -> dict:
    """Warm-up then a timed loop; return latency + resource stats."""
    if sync_fn is None:
        sync_fn = lambda: None

    print(f"[{label}] warm-up: {warmup} iters")
    for _ in range(warmup):
        infer_fn()
        sync_fn()

    print(f"[{label}] timing: {iters} iters")
    samples = np.empty(iters, dtype=np.float64)
    sampler = ResourceSampler() if sample_resources else None
    with (sampler if sampler is not None else _nullcontext()):
        for i in range(iters):
            sync_fn()
            t0 = time.perf_counter()
            infer_fn()
            sync_fn()
            samples[i] = (time.perf_counter() - t0) * 1000.0

    lat = LatencyStats.from_samples(samples)
    res = sampler.result() if sampler is not None else ResourceStats()
    print(f"[{label}] mean={lat.mean_ms:.3f}ms  p50={lat.p50_ms:.3f}ms  "
          f"p99={lat.p99_ms:.3f}ms  fps={lat.fps:.1f}")
    return {"label": label, "latency": asdict(lat), "fps": lat.fps,
            "resources": asdict(res)}


def save_result(result: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[saved] {path}")
