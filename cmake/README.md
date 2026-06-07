# cmake

_Toolchain files (aarch64 cross-compile), FindTensorRT.cmake._

| File | What |
|---|---|
| `FindTensorRT.cmake` | locates `nvinfer` + `nvonnxparser` (JetPack multiarch paths); exposes the `TensorRT::TensorRT` imported target |

CUDA comes from CMake's built-in `find_package(CUDAToolkit)` (`CUDA::cudart`),
OpenCV from `find_package(OpenCV)`. Override TensorRT discovery with
`-DTensorRT_ROOT=...` if it isn't in the default JetPack location.

> **Cross-compile (future):** add `aarch64-toolchain.cmake` here and configure with
> `-DCMAKE_TOOLCHAIN_FILE=cmake/aarch64-toolchain.cmake`. Not needed while building
> on-device.
