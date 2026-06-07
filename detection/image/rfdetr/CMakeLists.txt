# roboperc_detection_rfdetr — RF-DETR adapter (owns its pre/post, self-registers).

add_library(roboperc_detection_rfdetr STATIC
    src/rfdetr_detector.cpp
)

target_include_directories(roboperc_detection_rfdetr PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(roboperc_detection_rfdetr PUBLIC
    roboperc_core
    roboperc_runtime_tensorrt
)

set_target_properties(roboperc_detection_rfdetr PROPERTIES
    POSITION_INDEPENDENT_CODE ON)

# NOTE: the adapter self-registers in a static initializer with no externally
# referenced symbol, so consumers must whole-archive this library or the object
# is dropped. Link it via $<LINK_LIBRARY:WHOLE_ARCHIVE,roboperc_detection_rfdetr>
# (see bindings/ and apps/ CMakeLists) — that also propagates its include dirs.
