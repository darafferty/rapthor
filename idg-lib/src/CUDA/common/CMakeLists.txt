# Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later

project(cuda-common)

# Set headers
include_directories(${CUDAToolkit_INCLUDE_DIRS})

# Set headers
set(${PROJECT_NAME}_headers CUDA.h CU.h CUFFT.h InstanceCUDA.h PowerRecord.h)

# Set sources
set(${PROJECT_NAME}_sources
    CUDA.cpp
    CU.cpp
    CUFFT.cpp
    InstanceCUDA.cpp
    PowerRecord.cpp
    routines/Beam.cpp
    routines/WTiling.cpp)

# Set compile options
if(${USE_PHASOR_EXTRAPOLATION})
  add_compile_options("-DUSE_EXTRAPOLATE")
endif()

# Set build target
add_library(${PROJECT_NAME} OBJECT ${${PROJECT_NAME}_headers}
                                   ${${PROJECT_NAME}_sources})
if(NOT BUILD_STATIC_LIBS)
  set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-fPIC"
                                                   SOVERSION 0)
endif()

install(
  FILES kernels/KernelDegridder.cuh
        kernels/KernelDegridder.cu
        kernels/KernelGridder.cuh
        kernels/KernelGridder.cu
        kernels/KernelScaler.cu
        kernels/KernelAdder.cu
        kernels/KernelSplitter.cu
        kernels/KernelCalibrate.cu
        kernels/KernelAverageBeam.cu
        kernels/KernelFFTShift.cu
        kernels/KernelWtiling.cu
        kernels/math.cu
        kernels/Types.h
  COMPONENT gpu
  DESTINATION lib/idg-cuda)

# kernels/Types.h includes some common files, which need to be installed, too.
install(
  FILES ../../common/Index.h ../../common/KernelTypes.h ../../common/Math.h
  COMPONENT gpu
  DESTINATION lib/idg-cuda/common)

# Make kernel findable in build tree
execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory
                        ${CMAKE_BINARY_DIR}/lib)
execute_process(
  COMMAND ${CMAKE_COMMAND} -E create_symlink
          ${CMAKE_CURRENT_SOURCE_DIR}/kernels ${CMAKE_BINARY_DIR}/lib/idg-cuda)

install(FILES ${${PROJECT_NAME}_headers} DESTINATION include/CUDA/common)
