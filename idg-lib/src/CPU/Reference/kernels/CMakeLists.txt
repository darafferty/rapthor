# Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later

project(cpu-reference-kernels)

# Include directories
include_directories(${FFTW3_INCLUDE_DIR})

# Build type options
if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
  add_compile_options("-Wall")
  add_compile_options("-DDEBUG")
else()
  add_compile_options("-O3")
endif()

# Compiler options
if(${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
  add_compile_options("-qopenmp")
  add_compile_options("-mkl")
else()
  add_compile_options("-fopenmp")
  if(BUILD_WITH_MKL)
    link_libraries(${MKL_LIBRARIES})
  else()
    link_libraries(${FFTW3_LIBRARIES})
  endif()
endif()

# Miscellaneous options
add_compile_options("-DNR_CORRELATIONS=4")
add_library(
  ${PROJECT_NAME} OBJECT
  Kernels.h
  KernelGridder.cpp
  KernelDegridder.cpp
  KernelAdder.cpp
  KernelSplitter.cpp
  KernelFFT.cpp
  KernelAverageBeam.cpp)
