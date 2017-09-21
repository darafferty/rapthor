# This module tries to find CUDA driver libaries on your system
#
# Once done this will define
#  CUDA_DRIVER_LIBRARIES    - link these to use CUDA

find_package(PackageHandleStandardArgs)
find_library(CUDA_DRIVER_LIBRARY
    NAMES cuda
    HINTS ENV LD_LIBRARY_PATH
    HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64/stubs)
