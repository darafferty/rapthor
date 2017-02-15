project(idg)

# add subfolders
add_subdirectory(common)
add_subdirectory(powersensor)

if (BUILD_LIB_CPU)
    add_subdirectory(fft)
endif()

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
add_library(idg-common
  $<TARGET_OBJECTS:common>
)

set_target_properties(idg-common PROPERTIES
                      PUBLIC_HEADER idg-common.h)

install(TARGETS idg-common
  EXPORT IDGTargets
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib/static
  PUBLIC_HEADER DESTINATION include
)



#########################################################
# build libidg-powersensor, containing all code used to #
# to measure runtime and power consumption of kernels   #
#########################################################
add_library(idg-powersensor
  $<TARGET_OBJECTS:powersensor>
)

set (LINK_LIRBRARIES ${NUMA_LIBRARY} ${LIKWID_LIBRARIES} ${NVML_LIBRARY})
target_link_libraries(idg-powersensor ${LINK_LIRBRARIES})

set_target_properties(idg-powersensor PROPERTIES
                      PUBLIC_HEADER idg-powersensor.h)

install(TARGETS idg-powersensor
  EXPORT IDGTargets
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

set (LINK_LIRBRARIES ${FFTW3F_LIBRARY} ${FFTW3_LIBRARY})

# HACK TO MAKE PYTHON BINDING WORKS WITH MKL (needed?)
# if (BUILD_WITH_PYTHON)
#  set (LINK_LIRBRARIES ${LINK_LIRBRARIES} mkl_avx2 mkl_def)
# endif()

target_link_libraries(
  idg-fft
  ${LINK_LIRBRARIES}
  )

set_target_properties(idg-fft PROPERTIES
                      PUBLIC_HEADER idg-fft.h)

install(TARGETS idg-fft
  EXPORT IDGTargets
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
  $<TARGET_OBJECTS:cpu-reference2>
  $<TARGET_OBJECTS:cpu-optimized>
  $<TARGET_OBJECTS:cpu-knl>
)


set(LINK_LIRBRARIES idg-common idg-powersensor ${CMAKE_DL_LIBS})
target_link_libraries (
  idg-cpu
  ${LINK_LIRBRARIES}
)

set_target_properties(idg-cpu PROPERTIES
                      PUBLIC_HEADER idg-cpu.h)

install(TARGETS idg-cpu
  EXPORT IDGTargets
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
)

set(
    LINK_LIRBRARIES
    idg-common idg-powersensor
    ${CUDA_DRIVER_LIBRARY} ${CUDA_FFT_LIBRARY}
)

target_link_libraries (
    idg-cuda
    ${LINK_LIRBRARIES}
)

set_target_properties(
    idg-cuda PROPERTIES
    PUBLIC_HEADER idg-cuda.h
)

install(
    TARGETS idg-cuda
    EXPORT IDGTargets
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
    $<TARGET_OBJECTS:cpu-common>
    $<TARGET_OBJECTS:cpu-optimized>
    $<TARGET_OBJECTS:cuda-common>
    $<TARGET_OBJECTS:cuda-generic>
    $<TARGET_OBJECTS:hybrid-cuda-common>
)
#
#set (
#    LINK_LIRBRARIES
#    idg-common idg-powersensor
#    ${CUDA_DRIVER_LIBRARY} ${CUDA_FFT_LIBRARY}
#)
#
#target_link_libraries (
#    idg-hybrid-cuda
#    ${LINK_LIRBRARIES}
#)
#
#set_target_properties (
#    idg-hybrid-cuda PROPERTIES
#    PUBLIC_HEADER idg-hybrid-cuda.h
#)
#
#install(
#    TARGETS idg-hybrid-cuda
#    EXPORT IDGTargets
#    RUNTIME DESTINATION bin
#    LIBRARY DESTINATION lib
#    ARCHIVE DESTINATION lib/static
#    PUBLIC_HEADER DESTINATION include
#)

endif()


##################################################################
# build libidg-hybrid-opencl: containing all for CPU/OpenCL code #
##################################################################
if (BUILD_LIB_CPU AND BUILD_LIB_OPENCL)
#add_library(idg-hybrid-opencl
#  $<TARGET_OBJECTS:cpu-common>
#  $<TARGET_OBJECTS:cpu-reference>
#  $<TARGET_OBJECTS:cpu-optimized>
#  $<TARGET_OBJECTS:opencl-common>
#  $<TARGET_OBJECTS:opencl-generic>
#  $<TARGET_OBJECTS:hybrid-opencl>
#)
#
#set(LINK_LIRBRARIES idg-common idg-powersensor ${CMAKE_DL_LIBS} ${OPENCL_LIBRARIES} ${CLFFT_LIBRARIES})
#
#target_link_libraries (
#  idg-hybrid-opencl
#  ${LINK_LIRBRARIES}
#)
#
#set_target_properties(idg-hybrid-opencl PROPERTIES
#                      PUBLIC_HEADER idg-hybrid-opencl.h)
#
#install(TARGETS idg-hybrid-opencl
#  EXPORT IDGTargets
#  RUNTIME DESTINATION bin
#  LIBRARY DESTINATION lib
#  ARCHIVE DESTINATION lib/static
#  PUBLIC_HEADER DESTINATION include
#)
endif()


###################################################
# build libidg-opencl, containing all common code #
###################################################
if (BUILD_LIB_OPENCL)
add_library(idg-opencl
  $<TARGET_OBJECTS:opencl-common>
  $<TARGET_OBJECTS:opencl-generic>
)

set(LINK_LIRBRARIES idg-common idg-powersensor ${CMAKE_DL_LIBS} ${OPENCL_LIBRARIES} ${CLFFT_LIBRARIES})

target_link_libraries(
  idg-opencl
  ${LINK_LIRBRARIES}
)

set_target_properties(idg-opencl PROPERTIES
                      PUBLIC_HEADER idg-opencl.h)

install(TARGETS idg-opencl
  EXPORT IDGTargets
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

set(LINK_LIRBRARIES idg-fft ${CMAKE_DL_LIBS})

if (BUILD_LIB_CPU)
  set(LINK_LIRBRARIES ${LINK_LIRBRARIES} idg-cpu)
endif()
if (BUILD_LIB_CUDA)
  set(LINK_LIRBRARIES ${LINK_LIRBRARIES} idg-cuda)
endif()
if (BUILD_LIB_OPENCL)
  set(LINK_LIRBRARIES ${LINK_LIRBRARIES} idg-opencl)
endif()

target_link_libraries (
  idg
  ${LINK_LIRBRARIES}
)

set_target_properties(idg PROPERTIES
                      PUBLIC_HEADER idg.h)

install(TARGETS idg
  EXPORT IDGTargets
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib/static
  PUBLIC_HEADER DESTINATION include
)
