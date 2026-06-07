# roboperc_core — canonical types, the Detector contract, and the registry.
# Header-mostly; registry.cpp is the only compiled unit.

add_library(roboperc_core STATIC
    src/registry.cpp
)

target_include_directories(roboperc_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# detector.hpp exposes cv::Mat in the contract, so OpenCV headers are public.
target_link_libraries(roboperc_core PUBLIC ${OpenCV_LIBS})
target_include_directories(roboperc_core PUBLIC ${OpenCV_INCLUDE_DIRS})

set_target_properties(roboperc_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# ── tests ────────────────────────────────────────────────────────────────────
if(BUILD_TESTING)
    add_executable(test_registry tests/test_registry.cpp)
    target_link_libraries(test_registry PRIVATE roboperc_core)
    add_test(NAME test_registry COMMAND test_registry)
endif()
