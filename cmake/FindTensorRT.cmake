# FindTensorRT.cmake — locate TensorRT and expose an imported target.
#
# On JetPack the headers live in the aarch64 multiarch include dir and the libs
# in /usr/lib/aarch64-linux-gnu (TRT 10.3 here). Override with -DTensorRT_ROOT.
#
# Provides:
#   TensorRT_FOUND, TensorRT_INCLUDE_DIR, TensorRT_LIBRARIES
#   target  TensorRT::TensorRT  (links nvinfer + nvonnxparser, sets includes)

set(_trt_hints
    ${TensorRT_ROOT}
    /usr
    /usr/local/tensorrt
)

find_path(TensorRT_INCLUDE_DIR
    NAMES NvInfer.h
    HINTS ${_trt_hints}
    PATH_SUFFIXES include include/aarch64-linux-gnu aarch64-linux-gnu
)

find_library(TensorRT_nvinfer_LIBRARY
    NAMES nvinfer
    HINTS ${_trt_hints}
    PATH_SUFFIXES lib lib64 lib/aarch64-linux-gnu aarch64-linux-gnu
)

find_library(TensorRT_nvonnxparser_LIBRARY
    NAMES nvonnxparser
    HINTS ${_trt_hints}
    PATH_SUFFIXES lib lib64 lib/aarch64-linux-gnu aarch64-linux-gnu
)

# Parse the version from NvInferVersion.h when available (informational).
if(TensorRT_INCLUDE_DIR AND EXISTS "${TensorRT_INCLUDE_DIR}/NvInferVersion.h")
    file(STRINGS "${TensorRT_INCLUDE_DIR}/NvInferVersion.h" _trt_major
         REGEX "define NV_TENSORRT_MAJOR")
    string(REGEX MATCH "[0-9]+" TensorRT_VERSION_MAJOR "${_trt_major}")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TensorRT
    REQUIRED_VARS TensorRT_INCLUDE_DIR TensorRT_nvinfer_LIBRARY
                  TensorRT_nvonnxparser_LIBRARY
    VERSION_VAR TensorRT_VERSION_MAJOR)

if(TensorRT_FOUND)
    set(TensorRT_LIBRARIES ${TensorRT_nvinfer_LIBRARY}
                           ${TensorRT_nvonnxparser_LIBRARY})
    if(NOT TARGET TensorRT::TensorRT)
        add_library(TensorRT::TensorRT INTERFACE IMPORTED)
        set_target_properties(TensorRT::TensorRT PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIR}"
            INTERFACE_LINK_LIBRARIES "${TensorRT_LIBRARIES}")
    endif()
endif()

mark_as_advanced(TensorRT_INCLUDE_DIR TensorRT_nvinfer_LIBRARY
                 TensorRT_nvonnxparser_LIBRARY)
