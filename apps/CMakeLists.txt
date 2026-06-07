# apps — C++ executables. ros2/ nodes come later (see apps/ros2/README.md).

# run_inference builds a detector via the registry, so it must whole-archive the
# rfdetr adapter (otherwise its self-registration object is dropped).
add_executable(run_inference run_inference.cpp)
target_link_libraries(run_inference PRIVATE
    roboperc_core
    roboperc_io_decode
    roboperc_runtime_tensorrt
    "$<LINK_LIBRARY:WHOLE_ARCHIVE,roboperc_detection_rfdetr>"
)

# benchmark talks to the runtime directly (no registry needed), but still uses the
# rfdetr preprocess header — a normal link suffices.
add_executable(benchmark benchmark.cpp)
target_link_libraries(benchmark PRIVATE
    roboperc_io_decode
    roboperc_detection_rfdetr
    roboperc_runtime_tensorrt
)
