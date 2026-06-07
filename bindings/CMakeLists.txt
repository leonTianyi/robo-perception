# roboperc._native — pybind11 module. Built into python/roboperc/ so it imports
# as `roboperc._native` from the source tree (alongside `pip install -e .`).

pybind11_add_module(_native src/module.cpp)

target_link_libraries(_native PRIVATE
    roboperc_core
    roboperc_io_decode
    roboperc_runtime_tensorrt
    # whole-archive so the rfdetr self-registration survives; also propagates the
    # adapter's include dirs (needed for the rfdetr pre/post headers).
    "$<LINK_LIBRARY:WHOLE_ARCHIVE,roboperc_detection_rfdetr>"
)

# Drop the .so directly into the Python package.
set_target_properties(_native PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/python/roboperc)
