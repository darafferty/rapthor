# Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later

project(cpu-optimized)

# sources and header files
set(${PROJECT_NAME}_headers Optimized.h OptimizedC.h)

set(${PROJECT_NAME}_sources Optimized.cpp OptimizedC.cpp OptimizedKernels.cpp)

# create library
add_library(${PROJECT_NAME} OBJECT ${${PROJECT_NAME}_headers}
                                   ${${PROJECT_NAME}_sources})

if(BUILD_WITH_MKL)
  target_include_directories(${PROJECT_NAME} PRIVATE ${MKL_INCLUDE_DIRS})
else()
  target_include_directories(${PROJECT_NAME} PRIVATE ${FFTW3_INCLUDE_DIR})
endif()

# Copy header files
install(FILES ${${PROJECT_NAME}_headers} DESTINATION include/CPU/Optimized)

# Add kernels
add_subdirectory(kernels)
