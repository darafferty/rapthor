# This module tries to find CUDA driver libaries on your system
#
# Once done this will define
#  CUDA_DRIVER_FOUND        - system has CUDA
#  CUDA_DRIVER_LIBRARIES    - link these to use CUDA

FIND_PACKAGE(PackageHandleStandardArgs)
FIND_LIBRARY(CUDA_DRIVER_LIBRARIES cuda ENV LD_LIBRARY_PATH)
