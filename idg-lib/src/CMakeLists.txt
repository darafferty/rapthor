project(idg)

# add subfolders
add_subdirectory(common)
add_subdirectory(api)

if (BUILD_WITH_EXAMPLES OR BUILD_WITH_TESTS)
add_subdirectory(utility)
endif ()

if (BUILD_LIB_CPU OR BUILD_WITH_EXAMPLES OR BUILD_WITH_TESTS)
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
    add_subdirectory(python-api)
endif()
if (BUILD_WITH_BOOST_PYTHON)
    add_subdirectory(boost-python)
endif()



#########################################################
# build libidg-common, containing all common code       #
# Note: used to access common functionality from python #
#########################################################
add_library(idg-common
  $<TARGET_OBJECTS:common>
)

if (BUILD_WITH_LIKWID_POWER_SENSOR)
    set (LINK_LIRBRARIES ${NUMA_LIBRARY} ${LIKWID_LIBRARIES})
    target_link_libraries(idg-common ${LINK_LIRBRARIES})
endif()

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
# build libidg-fft, that wraps calls to FFTW or MKL;    #
# used in the CPU code, the example init code, API code #
#########################################################
add_library(idg-fft
  $<TARGET_OBJECTS:fft>
)

set (LINK_LIRBRARIES ${FFTW3F_LIBRARY} ${FFTW3_LIBRARY})

# HACK TO MAKE PYTHON BINDING WORKS WITH MKL
if (BUILD_WITH_PYTHON)
  set (LINK_LIRBRARIES ${LINK_LIRBRARIES} mkl_avx2 mkl_def)
endif()

if (BUILD_WITH_LIKWID_POWER_SENSOR)
    set (LINK_LIRBRARIES ${LINK_LIRBRARIES} ${NUMA_LIBRARY} ${LIKWID_LIBRARIES})
endif()

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


#########################################################
# build libidg-utility: init example data, ...
#########################################################
add_library(idg-utility
  $<TARGET_OBJECTS:utility-initialize>
  $<TARGET_OBJECTS:utility-external>
)

target_link_libraries(
  idg-utility
  idg-fft
  )

set_target_properties(idg-utility PROPERTIES
                      PUBLIC_HEADER idg-utility.h)

install(TARGETS idg-utility
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
  $<TARGET_OBJECTS:cpu-reference>
  $<TARGET_OBJECTS:cpu-haswellep>
)


set(LINK_LIRBRARIES idg-common ${CMAKE_DL_LIBS})
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
add_library(idg-cuda
  $<TARGET_OBJECTS:cuda-common>
  $<TARGET_OBJECTS:cuda-jetson>
  $<TARGET_OBJECTS:cuda-generic>
)

set(LINK_LIRBRARIES idg-common ${CUDA_CUFFT_LIBRARIES} ${CUDA_DRIVER_LIBRARIES})
target_link_libraries (
  idg-cuda
  ${LINK_LIRBRARIES}
)

set_target_properties(idg-cuda PROPERTIES
                      PUBLIC_HEADER idg-cuda.h)

install(TARGETS idg-cuda
  EXPORT IDGTargets
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib/static
  PUBLIC_HEADER DESTINATION include
)

endif(BUILD_LIB_CUDA)


##############################################################
# build libidg-hybrid-cuda: containing all for CPU/CUDA code #
##############################################################
if (BUILD_LIB_CPU AND BUILD_LIB_CUDA)
add_library(idg-hybrid-cuda
  $<TARGET_OBJECTS:cpu-common>
  $<TARGET_OBJECTS:cpu-reference>
  $<TARGET_OBJECTS:cpu-haswellep>
  $<TARGET_OBJECTS:cuda-common>
  $<TARGET_OBJECTS:hybrid-cuda>
  $<TARGET_OBJECTS:cuda-generic>
)

set(LINK_LIRBRARIES idg-common ${CMAKE_DL_LIBS} ${CUDA_CUFFT_LIBRARIES} ${CUDA_DRIVER_LIBRARIES})

target_link_libraries (
  idg-hybrid-cuda
  ${LINK_LIRBRARIES}
)

set_target_properties(idg-hybrid-cuda PROPERTIES
                      PUBLIC_HEADER idg-hybrid-cuda.h)

install(TARGETS idg-hybrid-cuda
  EXPORT IDGTargets
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib/static
  PUBLIC_HEADER DESTINATION include
)
endif()



##################################################################
# build libidg-hybrid-opencl: containing all for CPU/OpenCL code #
##############################################################
if (BUILD_LIB_CPU AND BUILD_LIB_OPENCL)
add_library(idg-hybrid-opencl
  $<TARGET_OBJECTS:cpu-common>
  $<TARGET_OBJECTS:cpu-reference>
  $<TARGET_OBJECTS:cpu-haswellep>
  $<TARGET_OBJECTS:opencl-common>
  $<TARGET_OBJECTS:opencl-reference>
  $<TARGET_OBJECTS:hybrid-opencl-reference>
)

set(LINK_LIRBRARIES idg-common ${CMAKE_DL_LIBS} ${OPENCL_LIBRARIES} ${CLFFT_LIBRARIES})

target_link_libraries (
  idg-hybrid-opencl
  ${LINK_LIRBRARIES}
)

set_target_properties(idg-hybrid-opencl PROPERTIES
                      PUBLIC_HEADER idg-hybrid-opencl.h)

install(TARGETS idg-hybrid-opencl
  EXPORT IDGTargets
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib/static
  PUBLIC_HEADER DESTINATION include
)
endif()



###################################################
# build libidg-opencl, containing all common code #
###################################################
if (BUILD_LIB_OPENCL)
add_library(idg-opencl
  $<TARGET_OBJECTS:opencl-common>
  $<TARGET_OBJECTS:opencl-reference>
)

set(LINK_LIRBRARIES idg-common ${CMAKE_DL_LIBS} ${OPENCL_LIBRARIES} ${CLFFT_LIBRARIES})

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
# build libidg-knc, containing all common code #
################################################
if (BUILD_LIB_KNC)
add_library(idg-knc
  $<TARGET_OBJECTS:hybrid-knc>
)

set(LINK_LIRBRARIES idg-common mkl_avx2 mkl_def)

target_link_libraries(
    idg-knc
    ${LINK_LIRBRARIES}
)

set_target_properties(idg-knc PROPERTIES
                      PUBLIC_HEADER idg-knc.h)

add_library(idg-knc-nocommon
  $<TARGET_OBJECTS:hybrid-knc>
)

install(TARGETS idg-knc idg-knc-nocommon
  EXPORT IDGTargets
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib/static
  PUBLIC_HEADER DESTINATION include
)
endif(BUILD_LIB_KNC)



################################################
# build libidg, containing all common code     #
################################################

add_library(idg
  $<TARGET_OBJECTS:idg-api>
)

set(LINK_LIRBRARIES idg-fft ${CMAKE_DL_LIBS})

if (BUILD_LIB_CPU)
  set(LINK_LIRBRARIES ${LINK_LIRBRARIES} idg-cpu)
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
