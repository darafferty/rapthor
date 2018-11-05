# This module tries to find CUDA driver libaries on your system
#
# Once done this will define
#  CUDA_FOUND        - system has CUDA
#  CUDA_INCLUDE_DIR  - the CUDA include directory
#  CUDA_CUDA_LIBRARY - link these to use CUDA driver api
#  CUDA_FFT_LIBRARY  - link these to use cuFFT

find_package(PackageHandleStandardArgs)

# CUDA runtime library
find_library(
    CUDA_CUDART_LIBRARY
    NAMES cudart
    HINTS ENV LD_LIBRARY_PATH
)

# CUDA driver library (could be a stub)
find_library(
    CUDA_CUDA_LIBRARY
    NAMES cuda
    HINTS ENV LD_LIBRARY_PATH
    PATH_SUFFIXES stubs
)

# CUDA FFT library
find_library(
    CUDA_FFT_LIBRARY
    NAMES cufft
    HINTS ENV LD_LIBRARY_PATH
)

# CUDA NVTX
find_library(
    CUDA_NVTX_LIBRARY
    NAMES nvToolsExt
    HINTS ENV LD_LIBRARY_PATH
)

# Find CUDA include directory
get_filename_component(CUDA_LIB_DIR ${CUDA_CUDART_LIBRARY} PATH)
get_filename_component(_CUDA_INCLUDE_DIR ${CUDA_LIB_DIR}/../include ABSOLUTE)
find_path(CUDA_INCLUDE_DIR cuda.h PATH ${_CUDA_INCLUDE_DIR})

find_package_handle_standard_args(CUDA DEFAULT_MSG CUDA_CUDA_LIBRARY CUDA_INCLUDE_DIR)
