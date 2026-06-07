cmake_minimum_required(VERSION 3.22)
project(robo_perception LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

# ── dependencies ──────────────────────────────────────────────────────────────
find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)         # gives CUDA::cudart (no CUDA lang needed)
find_package(TensorRT REQUIRED)            # cmake/FindTensorRT.cmake
find_package(pybind11 CONFIG QUIET)        # optional: only the bindings need it

# ── tests on by default; disable with -DBUILD_TESTING=OFF ─────────────────────
include(CTest)

# ── subsystems (each is its own library target) ───────────────────────────────
# Concrete first slice: the RF-DETR image-detection capability + its deps.
# odometry / fusion / io/streams / detection/pointcloud stay stubs until needed.
add_subdirectory(core)
add_subdirectory(io/decode)
add_subdirectory(runtime/tensorrt)
add_subdirectory(detection/image/rfdetr)
add_subdirectory(apps)

if(pybind11_FOUND)
    add_subdirectory(bindings)
else()
    message(STATUS
        "pybind11 not found — skipping roboperc._native. "
        "Install it (pip install pybind11) and re-run cmake to build the Python bridge.")
endif()

# ── summary ───────────────────────────────────────────────────────────────────
message(STATUS "robo-perception configured:")
message(STATUS "  OpenCV     ${OpenCV_VERSION}")
message(STATUS "  CUDA       ${CUDAToolkit_VERSION}")
message(STATUS "  TensorRT   include=${TensorRT_INCLUDE_DIR}")
message(STATUS "  pybind11   ${pybind11_FOUND}")
