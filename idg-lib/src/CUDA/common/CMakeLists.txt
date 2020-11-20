# Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later

project(cuda-common)

# Set headers
include_directories (
    ${CUDA_INCLUDE_DIR}
)

# Set headers
set (
    ${PROJECT_NAME}_headers
    CUDA.h
    CU.h
    CUFFT.h
    InstanceCUDA.h
    PowerRecord.h
)

# Set sources
set (
    ${PROJECT_NAME}_sources
    CUDA.cpp
    CU.cpp
    CUFFT.cpp
    InstanceCUDA.cpp
    PowerRecord.cpp
)

# Set build target
add_library(
    ${PROJECT_NAME} OBJECT ${${PROJECT_NAME}_headers}
                           ${${PROJECT_NAME}_sources}
)
if (NOT BUILD_STATIC_LIBS)
set_target_properties (${PROJECT_NAME} PROPERTIES
        COMPILE_FLAGS "-fPIC"
        SOVERSION 0
)
endif()

# set install
install(
    FILES
    kernels/KernelDegridder.cu
    kernels/KernelFFT.cu
    kernels/KernelGridder.cu
    kernels/KernelScaler.cu
    kernels/KernelAdder.cu
    kernels/KernelSplitter.cu
    kernels/KernelCalibrate.cu
    kernels/math.cu
    kernels/Types.h
    DESTINATION
    lib/idg-cuda
)

install(
   FILES
   ${${PROJECT_NAME}_headers}
   DESTINATION
   include/CUDA/common
)
