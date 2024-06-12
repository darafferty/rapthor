# Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later

project(cuda-common)

# Set headers
include_directories(${CUDAToolkit_INCLUDE_DIRS})

# Set headers
set(${PROJECT_NAME}_headers CUDA.h InstanceCUDA.h PowerRecord.h)

# Set sources
set(${PROJECT_NAME}_sources
    gpu_utils.cpp
    CUDA.cpp
    InstanceCUDA.cpp
    PowerRecord.cpp
    routines/Beam.cpp
    routines/WTiling.cpp
    KernelFactory.cpp
    kernels/Kernel.cpp
    kernels/KernelAdder.cpp
    kernels/KernelAverageBeam.cpp
    kernels/KernelCalibrate.cpp
    kernels/KernelDegridder.cpp
    kernels/KernelFFT.cpp
    kernels/KernelFFTShift.cpp
    kernels/KernelGridder.cpp
    kernels/KernelScaler.cpp
    kernels/KernelSplitter.cpp
    kernels/KernelWTiling.cpp)

# Set compile options
if(${USE_PHASOR_EXTRAPOLATION})
  add_compile_options("-DUSE_EXTRAPOLATE")
endif()

# Set build target
add_library(${PROJECT_NAME} OBJECT ${${PROJECT_NAME}_headers}
                                   ${${PROJECT_NAME}_sources})

find_package(
  Boost
  COMPONENTS system
  REQUIRED)

target_link_libraries(${PROJECT_NAME} PRIVATE Boost::system xtensor)

install(
  FILES kernels/KernelDegridder.cu
        kernels/KernelGridder.cu
        kernels/KernelScaler.cu
        kernels/KernelAdder.cu
        kernels/KernelSplitter.cu
        kernels/KernelCalibrate_index.cuh
        kernels/KernelCalibrate_lmnp.cu
        kernels/KernelCalibrate_sums.cu
        kernels/KernelCalibrate_gradient.cu
        kernels/KernelCalibrate_hessian.cu
        kernels/KernelAverageBeam.cu
        kernels/KernelFFTShift.cu
        kernels/KernelWTiling_copy.cu
        kernels/KernelWTiling_phasor.cu
        kernels/KernelWTiling_subgrids_from_wtiles.cu
        kernels/KernelWTiling_subgrids_to_wtiles.cu
        kernels/KernelWTiling_wtiles_from_grid.cu
        kernels/KernelWTiling_wtiles_from_patch.cu
        kernels/KernelWTiling_wtiles_to_grid.cu
        kernels/KernelWTiling_wtiles_to_patch.cu
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
