project(idg)

set (IDG_CUDA_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/CUDA/common)
set (IDG_OpenCL_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/OpenCL/common)

include_directories(external)

# add subfolders
add_subdirectory(common)
add_subdirectory(fft)
if (BUILD_LIB_CPU)
    add_subdirectory(CPU)
endif()

add_subdirectory(Hybrid)

if (BUILD_LIB_CUDA)
    add_subdirectory(CUDA)
endif()
if (BUILD_LIB_OPENCL)
    add_subdirectory(OpenCL)
endif()
if (BUILD_WITH_PYTHON)
    add_subdirectory(python)
endif()



#########################################################
# build libidg-common, containing all common code       #
# Note: used to access common functionality from python #
#########################################################
add_library(
    idg-common
    $<TARGET_OBJECTS:common>
)

if(HAVE_POWERSENSOR)
    target_link_libraries(
        idg-common
        ${POWERSENSOR_LIBRARY})
endif()

set_target_properties(
    idg-common PROPERTIES
    PUBLIC_HEADER idg-common.h
)

install(
    TARGETS idg-common
    EXPORT idg-targets
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib/static
    PUBLIC_HEADER DESTINATION include
)



#########################################################
# build libidg-fft, that wraps calls to FFTW or MKL;    #
# used in the CPU code, the example init code, API code #
#########################################################
add_library(idg-fft
    $<TARGET_OBJECTS:fft>
)

set(
    LINK_LIBRARIES
    ${FFTW3_LIBRARIES}
)

target_link_libraries(
    idg-fft
    ${LINK_LIBRARIES}
)

set_target_properties(
    idg-fft PROPERTIES
    PUBLIC_HEADER idg-fft.h
)

install(
    TARGETS idg-fft
    EXPORT idg-targets
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib/static
    PUBLIC_HEADER DESTINATION include
)


########################################################
# build libidg-cpu, containing all needed for CPU code #
########################################################
if (BUILD_LIB_CPU)
add_library(idg-cpu
  $<TARGET_OBJECTS:cpu-common>
  $<TARGET_OBJECTS:cpu-reference>
  $<TARGET_OBJECTS:cpu-optimized>
)


set(LINK_LIBRARIES idg-common ${CMAKE_DL_LIBS})
if(BUILD_WITH_MKL)
    set(LINK_LIBRARIES ${LINK_LIBRARIES} ${MKL_LIBRARIES})
endif()
target_link_libraries (
  idg-cpu
  ${LINK_LIBRARIES}
)

set_target_properties(idg-cpu PROPERTIES
                      PUBLIC_HEADER idg-cpu.h)

install(
  TARGETS idg-cpu
  EXPORT idg-targets
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib/static
  PUBLIC_HEADER DESTINATION include
)


endif()


##########################################################
# build libidg-cuda, containing all needed for CUDA code #
##########################################################
if (BUILD_LIB_CUDA)

add_library (
    idg-cuda
    $<TARGET_OBJECTS:cuda-common>
    $<TARGET_OBJECTS:cuda-generic>
    $<TARGET_OBJECTS:cuda-unified>
)

set(
    LINK_LIBRARIES
    idg-common
    ${CUDA_CUDA_LIBRARY} ${CUDA_FFT_LIBRARY} ${CUDA_NVTX_LIBRARY}
)

target_link_libraries (
    idg-cuda
    ${LINK_LIBRARIES}
)

set_target_properties(
    idg-cuda PROPERTIES
    PUBLIC_HEADER idg-cuda.h
)

install(
    TARGETS idg-cuda
    EXPORT idg-targets
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib/static
    PUBLIC_HEADER DESTINATION include
)

endif (BUILD_LIB_CUDA)


##############################################################
# build libidg-hybrid-cuda: containing all for CPU/CUDA code #
##############################################################
if (BUILD_LIB_CPU AND BUILD_LIB_CUDA)

add_library (
    idg-hybrid-cuda
    $<TARGET_OBJECTS:hybrid-cuda-generic-optimized>
    $<TARGET_OBJECTS:hybrid-cuda-unified-optimized>
)

set (
    LINK_LIBRARIES
    idg-cpu
    idg-cuda
)

target_link_libraries (
    idg-hybrid-cuda
    ${LINK_LIBRARIES}
)

set_target_properties (
    idg-hybrid-cuda PROPERTIES
    PUBLIC_HEADER idg-hybrid-cuda.h
)

install(
    TARGETS idg-hybrid-cuda
    EXPORT idg-targets
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib/static
    PUBLIC_HEADER DESTINATION include
)

endif()


###################################################
# build libidg-opencl, containing all common code #
###################################################
if(BUILD_LIB_OPENCL)

add_library(
    idg-opencl
    $<TARGET_OBJECTS:opencl-common>
    $<TARGET_OBJECTS:opencl-generic>
)

set(
    LINK_LIBRARIES
    idg-common
    ${OPENCL_LIBRARIES}
    ${CLFFT_LIBRARIES}
)

target_link_libraries(
    idg-opencl
    ${LINK_LIBRARIES}
)

set_target_properties(
    idg-opencl PROPERTIES
    PUBLIC_HEADER idg-opencl.h
)

install(
    TARGETS idg-opencl
    EXPORT idg-targets
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib/static
    PUBLIC_HEADER DESTINATION include
)

endif(BUILD_LIB_OPENCL)



################################################
# build libidg, containing all common code     #
################################################

add_library(idg
    idg.cpp
)

set(LINK_LIBRARIES idg-fft)
if (BUILD_LIB_CPU)
  set(LINK_LIBRARIES ${LINK_LIBRARIES} idg-cpu)
endif()
if (BUILD_LIB_CUDA)
  set(LINK_LIBRARIES ${LINK_LIBRARIES} idg-cuda idg-hybrid-cuda)
endif()
if (BUILD_LIB_OPENCL)
  set(LINK_LIBRARIES ${LINK_LIBRARIES} idg-opencl)
endif()

target_link_libraries (
  idg
  ${LINK_LIBRARIES}
)

set_target_properties(idg PROPERTIES
                      PUBLIC_HEADER idg.h)

install(
    TARGETS idg
  EXPORT idg-targets
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib/static
  PUBLIC_HEADER DESTINATION include
)
