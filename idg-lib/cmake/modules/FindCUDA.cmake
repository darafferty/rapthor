# This module tries to find CUDA libaries on your system
#
# Once done this will define
#  CUDA_FOUND          - system has CUDA
#  CUDA_INCLUDE_DIR    - the CUDA include directory
#  CUDA_LIB_DIR        - the CUDA library directory
#  CUDA_CUDART_LIBRARY - link these to use CUDA runtime api
#  CUDA_CUDA_LIBRARY   - link these to use CUDA driver api
#  CUDA_FFT_LIBRARY    - link these to use cuFFT
#  CUDA_NVTX_LIBRARY   - link these to use NVTX

find_package(PackageHandleStandardArgs)

# CUDA runtime library
find_library(
    CUDA_CUDART_LIBRARY
    NAMES cudart
    HINTS ${CUDA_ROOT_DIR}
    HINTS ENV LD_LIBRARY_PATH
    PATH_SUFFIXES lib64
)

# CUDA library directory
get_filename_component(CUDA_LIB_DIR ${CUDA_CUDART_LIBRARY} PATH)

# Make sure CUDA_ROOT_DIR is set
if("${CUDA_ROOT_DIR}" STREQUAL "")
    get_filename_component(CUDA_ROOT_DIR ${CUDA_LIB_DIR}/.. ABSOLUTE)
endif()

# Find CUDA include directory
find_path(
    CUDA_INCLUDE_DIR
    NAMES cuda.h
    HINTS ${CUDA_ROOT_DIR}
    PATH_SUFFIXES include
)

# CUDA driver library stub
find_library(
    CUDA_CUDA_LIBRARY
    NAMES cuda
    HINTS ${CUDA_ROOT_DIR}
    PATH_SUFFIXES lib64/stubs
)

# CUDA FFT library
find_library(
    CUDA_FFT_LIBRARY
    NAMES cufft
    HINTS ${CUDA_ROOT_DIR}
    PATH_SUFFIXES lib64
)

# CUDA NVTX
find_library(
    CUDA_NVTX_LIBRARY
    NAMES nvToolsExt
    HINTS ${CUDA_ROOT_DIR}
    PATH_SUFFIXES lib64
)

# CUDA compiler
find_program(
    CUDA_NVCC
    NAMES nvcc
    HINTS ${CUDA_ROOT_DIR}
    PATH_SUFFIXES bin
)

find_package_handle_standard_args(CUDA DEFAULT_MSG CUDA_CUDA_LIBRARY CUDA_INCLUDE_DIR)
