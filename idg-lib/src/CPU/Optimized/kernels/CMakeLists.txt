# Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later

project(cpu-optimized-kernels)

# Build type options
add_compile_options("-Wall")
add_compile_options("-DDEBUG")
add_compile_options("-O3")
add_compile_options("-ffast-math")

# Compiler options
if(${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
  add_compile_options("-x${INSTRUCTION_SET}")
  add_compile_options("-DUSE_VML")
  add_compile_options("-mkl")
  link_libraries(${MKL_LIBRARIES})
else()
  if(PORTABLE)
    message(
      WARNING
        "Building Optimized CPU kernels non-natively may negatively impact performance."
    )
  else()
    # Enable optimizations for the current CPU architecture
    if(POWER_ARCHITECTURE)
      # e.g. Power 8, Power 9
      include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../external/)
      add_compile_options("-mcpu=native")
    else()
      # x86_64
      add_compile_options("-march=native")
    endif()
  endif()
  add_compile_options("-fopenmp")
  if(BUILD_WITH_MKL)
    add_compile_options("-DUSE_VML")
    link_libraries(${MKL_LIBRARIES})
    include_directories(${MKL_INCLUDE_DIRS})
  else()
    link_libraries(${FFTW3_LIBRARIES})
    include_directories(${FFTW3_INCLUDE_DIR})
  endif()
endif()

if(${USE_LOOKUP_TABLE})
  add_compile_options("-DUSE_LOOKUP")
endif()
if(${USE_PHASOR_EXTRAPOLATION})
  add_compile_options("-DUSE_EXTRAPOLATE")
endif()

# Add library
add_library(
  ${PROJECT_NAME} OBJECT
  Lookup.cpp
  KernelGridder.cpp
  KernelDegridder.cpp
  KernelAdder.cpp
  KernelSplitter.cpp
  KernelFFT.cpp
  KernelCalibrate.cpp
  KernelAdderWStack.cpp
  KernelSplitterWStack.cpp
  KernelWtiling.cpp)
