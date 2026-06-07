# roboperc_io_decode — the single image-decode source of truth.

add_library(roboperc_io_decode STATIC
    src/decode.cpp
)

target_include_directories(roboperc_io_decode PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(roboperc_io_decode PUBLIC ${OpenCV_LIBS})
target_include_directories(roboperc_io_decode PUBLIC ${OpenCV_INCLUDE_DIRS})

set_target_properties(roboperc_io_decode PROPERTIES POSITION_INDEPENDENT_CODE ON)
