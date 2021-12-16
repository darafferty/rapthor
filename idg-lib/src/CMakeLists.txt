# Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later

project(idg)

set(IDG_CUDA_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/CUDA/common)
set(IDG_OpenCL_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/OpenCL/common)

include_directories(external)

# add subfolders
add_subdirectory(common)
add_subdirectory(external)
add_subdirectory(fft)
if(BUILD_LIB_CPU)
  add_subdirectory(CPU)
endif()

add_subdirectory(Hybrid)

if(BUILD_LIB_CUDA)
  add_subdirectory(CUDA)
endif()
if(BUILD_LIB_OPENCL)
  add_subdirectory(OpenCL)
endif()
if(BUILD_WITH_PYTHON)
  add_subdirectory(python)
endif()

#########################################################
# enable rpath for all of the libidg-*.so files         #
#########################################################
list(APPEND CMAKE_INSTALL_RPATH ${CUDA_LIB_DIR})
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

#########################################################
# build libidg-external, containing all external code   #
#########################################################
add_library(idg-external $<TARGET_OBJECTS:external>)

set_target_properties(idg-external PROPERTIES PUBLIC_HEADER idg-external.h
                                              SOVERSION 0)

install(
  TARGETS idg-external
  COMPONENT base
  EXPORT idg-targets
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib/static
  PUBLIC_HEADER DESTINATION include)

#########################################################
# build libidg-common, containing all common code       #
# Note: used to access common functionality from python #
#########################################################
add_library(idg-common $<TARGET_OBJECTS:common>)

set(LINK_LIBRARIES idg-external)

if(HAVE_POWERSENSOR)
  set(LINK_LIBRARIES ${LINK_LIBRARIES} ${POWERSENSOR_LIBRARY})
endif()

target_link_libraries(idg-common ${LINK_LIBRARIES})

set_target_properties(idg-common PROPERTIES PUBLIC_HEADER idg-common.h SOVERSION
                                                                       0)

install(
  TARGETS idg-common
  COMPONENT base
  EXPORT idg-targets
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib/static
  PUBLIC_HEADER DESTINATION include)

#########################################################
# build libidg-fft, that wraps calls to FFTW or MKL;    #
# used in the CPU code, the example init code, API code #
#########################################################
add_library(idg-fft $<TARGET_OBJECTS:fft>)

set(LINK_LIBRARIES ${FFTW3_LIBRARIES})

target_link_libraries(idg-fft ${LINK_LIBRARIES})

set_target_properties(idg-fft PROPERTIES PUBLIC_HEADER idg-fft.h SOVERSION 0)

install(
  TARGETS idg-fft
  COMPONENT base
  EXPORT idg-targets
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib/static
  PUBLIC_HEADER DESTINATION include)

########################################################
# build libidg-cpu, containing all needed for CPU code #
########################################################
if(BUILD_LIB_CPU)
  add_library(
    idg-cpu
    $<TARGET_OBJECTS:common>
    $<TARGET_OBJECTS:cpu-common>
    $<TARGET_OBJECTS:cpu-reference>
    $<TARGET_OBJECTS:cpu-optimized>
    $<TARGET_OBJECTS:cpu-reference-kernels>
    $<TARGET_OBJECTS:cpu-optimized-kernels>)

  set(LINK_LIBRARIES idg-common ${FFTW3_LIBRARIES})
  if(BUILD_WITH_MKL)
    set(LINK_LIBRARIES ${LINK_LIBRARIES} ${MKL_LIBRARIES})
  endif()
  target_link_libraries(idg-cpu ${LINK_LIBRARIES})

  set_target_properties(idg-cpu PROPERTIES PUBLIC_HEADER idg-cpu.h SOVERSION 0)

  install(
    TARGETS idg-cpu
    COMPONENT cpu
    EXPORT idg-targets
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib/static
    PUBLIC_HEADER DESTINATION include)

endif()

##########################################################
# build libidg-cuda, containing all needed for CUDA code #
##########################################################
if(BUILD_LIB_CUDA)

  add_library(
    idg-cuda $<TARGET_OBJECTS:common> $<TARGET_OBJECTS:cuda-common>
             $<TARGET_OBJECTS:cuda-generic> $<TARGET_OBJECTS:cuda-unified>)

  set(LINK_LIBRARIES idg-common CUDA::cuda_driver CUDA::cufft CUDA::nvToolsExt)

  target_link_libraries(idg-cuda ${LINK_LIBRARIES})

  set_target_properties(idg-cuda PROPERTIES PUBLIC_HEADER idg-cuda.h SOVERSION
                                                                     0)

  install(
    TARGETS idg-cuda
    COMPONENT gpu
    EXPORT idg-targets
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib/static
    PUBLIC_HEADER DESTINATION include)

endif(BUILD_LIB_CUDA)

##############################################################
# build libidg-hybrid-cuda: containing all for CPU/CUDA code #
##############################################################
if(BUILD_LIB_CPU AND BUILD_LIB_CUDA)

  add_library(idg-hybrid-cuda $<TARGET_OBJECTS:hybrid-cuda-generic-optimized>)

  set(LINK_LIBRARIES idg-cpu idg-cuda)

  target_link_libraries(idg-hybrid-cuda ${LINK_LIBRARIES})

  set_target_properties(idg-hybrid-cuda
                        PROPERTIES PUBLIC_HEADER idg-hybrid-cuda.h SOVERSION 0)

  install(
    TARGETS idg-hybrid-cuda
    COMPONENT hybrid
    EXPORT idg-targets
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib/static
    PUBLIC_HEADER DESTINATION include)

endif()

###################################################
# build libidg-opencl, containing all common code #
###################################################
if(BUILD_LIB_OPENCL)

  add_library(idg-opencl $<TARGET_OBJECTS:opencl-common>
                         $<TARGET_OBJECTS:opencl-generic>)

  set(LINK_LIBRARIES idg-common ${OPENCL_LIBRARIES} ${CLFFT_LIBRARIES})

  target_link_libraries(idg-opencl ${LINK_LIBRARIES})

  set_target_properties(idg-opencl PROPERTIES PUBLIC_HEADER idg-opencl.h
                                              SOVERSION 0)

  install(
    TARGETS idg-opencl
    COMPONENT gpu
    EXPORT idg-targets
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib/static
    PUBLIC_HEADER DESTINATION include)

endif(BUILD_LIB_OPENCL)

################################################
# build libidg, containing all common code     #
################################################

add_library(idg idg.cpp)

set(LINK_LIBRARIES idg-fft)
if(BUILD_LIB_CPU)
  set(LINK_LIBRARIES ${LINK_LIBRARIES} idg-cpu)
endif()
if(BUILD_LIB_CUDA)
  set(LINK_LIBRARIES ${LINK_LIBRARIES} idg-cuda)
endif()
if(BUILD_LIB_CPU AND BUILD_LIB_CUDA)
  set(LINK_LIBRARIES ${LINK_LIBRARIES} idg-hybrid-cuda)
endif()
if(BUILD_LIB_OPENCL)
  set(LINK_LIBRARIES ${LINK_LIBRARIES} idg-opencl)
endif()

target_link_libraries(idg ${LINK_LIBRARIES})

set_target_properties(idg PROPERTIES PUBLIC_HEADER idg.h SOVERSION 0)

install(
  TARGETS idg
  EXPORT idg-targets
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib/static
  PUBLIC_HEADER DESTINATION include)
