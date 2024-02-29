# Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later

project(hybrid-cuda-generic-optimized)

# Set headers
include_directories(${IDG_CUDA_INCLUDE_DIR} ${CUDAToolkit_INCLUDE_DIRS}
                    ${FFTW3_INCLUDE_DIR})

# Set headers
set(${PROJECT_NAME}_headers GenericOptimized.h GenericOptimizedC.h)

# Set sources
set(${PROJECT_NAME}_sources GenericOptimized.cpp GenericOptimizedC.cpp
                            routines/Imaging.cpp routines/Calibration.cpp)

# Set build target
add_library(${PROJECT_NAME} OBJECT ${${PROJECT_NAME}_headers}
                                   ${${PROJECT_NAME}_sources})

# Set install
install(FILES ${${PROJECT_NAME}_headers}
        DESTINATION include/Hybrid/CUDA/GenericOptimized)
