# This module tries to find Nvidia Management library on your system
#
# Once done this will define
#  NVML_FOUND        - system has NVML
#  NVML_LIBRARIES    - link these to use NVML

FIND_PACKAGE(PackageHandleStandardArgs)
FIND_LIBRARY(NVML_LIBRARY nvidia-ml ENV LD_LIBRARY_PATH)
