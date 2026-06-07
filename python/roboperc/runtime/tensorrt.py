"""TensorRT backend (Python) — cuda-python twin of the C++ TensorRTRunner.

Loads a serialized engine, allocates device buffers for every I/O tensor, and runs
inference via enqueueV3 + setTensorAddress. Exposes device-scope hooks so
eval/benchmark can time pure GPU compute, exactly like apps/benchmark does in C++.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import tensorrt as trt

# cuda-python moved the runtime bindings under cuda.bindings in newer versions.
try:
    from cuda.bindings import runtime as cudart
except ImportError:  # pragma: no cover
    from cuda import cudart


def _check(err, msg=""):
    if isinstance(err, tuple):
        err = err[0]
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA error {err} {msg}")


class TensorRTRunner:
    name = "tensorrt"

    def __init__(self, engine_path: Path, logger_severity=trt.Logger.WARNING):
        self.logger = trt.Logger(logger_severity)
        trt.init_libnvinfer_plugins(self.logger, "")
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"failed to deserialize {engine_path}")
        self.context = self.engine.create_execution_context()
        _, self.stream = cudart.cudaStreamCreate()

        self.inputs, self.outputs = [], []
        self.bindings = {}
        for i in range(self.engine.num_io_tensors):
            n = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(n))
            dtype = trt.nptype(self.engine.get_tensor_dtype(n))
            nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
            err, dptr = cudart.cudaMalloc(nbytes)
            _check(err, f"malloc {n}")
            self.bindings[n] = dict(host=np.empty(shape, dtype=dtype),
                                    device=int(dptr), shape=shape, dtype=dtype,
                                    nbytes=nbytes)
            (self.inputs if self.engine.get_tensor_mode(n) ==
             trt.TensorIOMode.INPUT else self.outputs).append(n)
            self.context.set_tensor_address(n, int(dptr))

    @property
    def input_name(self) -> str:
        return self.inputs[0]

    # ── full inference ────────────────────────────────────────────────────────
    def infer(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        inp = self.bindings[self.input_name]
        x = np.ascontiguousarray(x, dtype=inp["dtype"])
        _check(cudart.cudaMemcpyAsync(
            inp["device"], x.ctypes.data, inp["nbytes"],
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream), "H2D")
        self.context.execute_async_v3(self.stream)
        for n in self.outputs:
            o = self.bindings[n]
            _check(cudart.cudaMemcpyAsync(
                o["host"].ctypes.data, o["device"], o["nbytes"],
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, self.stream), "D2H")
        _check(cudart.cudaStreamSynchronize(self.stream), "sync")
        return self.output_by_role()

    def output_by_role(self) -> Tuple[np.ndarray, np.ndarray]:
        """(logits, boxes) matched by last-dim size (==4 -> boxes)."""
        logits = boxes = None
        for n in self.outputs:
            arr = self.bindings[n]["host"]
            if arr.shape[-1] == 4:
                boxes = arr
            else:
                logits = arr
        return logits, boxes

    # ── device-scope hooks (pure GPU compute, for benchmarking) ─────────────────
    def load_input_once(self, x: np.ndarray):
        inp = self.bindings[self.input_name]
        x = np.ascontiguousarray(x, dtype=inp["dtype"])
        _check(cudart.cudaMemcpyAsync(
            inp["device"], x.ctypes.data, inp["nbytes"],
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream), "H2D")
        self.synchronize()

    def infer_device_only(self):
        self.context.execute_async_v3(self.stream)

    def synchronize(self):
        _check(cudart.cudaStreamSynchronize(self.stream), "sync")

    def close(self):
        for e in self.bindings.values():
            cudart.cudaFree(e["device"])
        cudart.cudaStreamDestroy(self.stream)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
