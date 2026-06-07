# roboperc_runtime_tensorrt — model-agnostic TensorRT 10 execution backend.

add_library(roboperc_runtime_tensorrt STATIC
    src/tensorrt_runner.cpp
)

target_include_directories(roboperc_runtime_tensorrt PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(roboperc_runtime_tensorrt PUBLIC
    TensorRT::TensorRT
    CUDA::cudart
)

set_target_properties(roboperc_runtime_tensorrt PROPERTIES
    POSITION_INDEPENDENT_CODE ON)
