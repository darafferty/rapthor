project(idg)

# add subfolders
add_subdirectory(common)

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

install(TARGETS idg-common
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib/static
)

install(
    FILES
    idg-common.h
    DESTINATION
    include
)



########################################################
# build libidg-cpu, containing all needed for CPU code #
########################################################
if (BUILD_LIB_CPU)
add_library(idg-cpu
  $<TARGET_OBJECTS:common>
  $<TARGET_OBJECTS:cpu-common>
  $<TARGET_OBJECTS:cpu-reference>
  $<TARGET_OBJECTS:cpu-haswellep>
  $<TARGET_OBJECTS:cpu-sandybridgeep>
)

install(TARGETS idg-cpu
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib/static
)

install(
    FILES
    idg-cpu.h
    DESTINATION
    include
)
endif()


##########################################################
# build libidg-cuda, containing all needed for CUDA code #
##########################################################
if (BUILD_LIB_CUDA)
add_library(idg-cuda
  $<TARGET_OBJECTS:common>
  $<TARGET_OBJECTS:cuda-common>
  $<TARGET_OBJECTS:cuda-kepler>
  $<TARGET_OBJECTS:cuda-maxwell>
)

install(TARGETS idg-cuda
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib/static
)

set(LINK_LIRBRARIES ${CUDA_CUFFT_LIBRARIES} ${CUDA_DRIVER_LIBRARIES})

target_link_libraries (
  idg-cuda
  ${LINK_LIRBRARIES}
)

install(
    FILES
    idg-cuda.h
    DESTINATION
    include
)
endif(BUILD_LIB_CUDA)


##############################################################
# build libidg-hybrid-cuda: containing all for CPU/CUDA code #
##############################################################
if (BUILD_LIB_CPU AND BUILD_LIB_CUDA)
add_library(idg-hybrid-cuda
  $<TARGET_OBJECTS:common>
  $<TARGET_OBJECTS:cpu-common>
  $<TARGET_OBJECTS:cpu-reference>
  $<TARGET_OBJECTS:cpu-haswellep>
  $<TARGET_OBJECTS:cpu-sandybridgeep>
  $<TARGET_OBJECTS:cuda-common>
  $<TARGET_OBJECTS:cuda-kepler>
  $<TARGET_OBJECTS:cuda-maxwell>
  $<TARGET_OBJECTS:hybrid-cuda-maxwellhaswellep>
)

set(LINK_LIRBRARIES ${CMAKE_DL_LIBS} ${CUDA_CUFFT_LIBRARIES} ${CUDA_DRIVER_LIBRARIES})

target_link_libraries (
  idg-hybrid-cuda
  ${LINK_LIRBRARIES}
)

install(TARGETS idg-hybrid-cuda
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib/static
)

install(
    FILES
    idg-hybrid-cuda.h
    DESTINATION
    include
)
endif()



###################################################
# build libidg-opencl, containing all common code #
###################################################
if (BUILD_LIB_OPENCL)
add_library(idg-opencl
  $<TARGET_OBJECTS:common>
  $<TARGET_OBJECTS:opencl-common>
  $<TARGET_OBJECTS:opencl-reference>
)

install(TARGETS idg-opencl
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib/static
)

install(
    FILES
    idg-opencl.h
    DESTINATION
    include
)
endif(BUILD_LIB_OPENCL)



################################################
# build libidg-knc, containing all common code #
################################################
if (BUILD_LIB_KNC)

add_library(idg-knc
  $<TARGET_OBJECTS:common>
  $<TARGET_OBJECTS:hybrid-knc>
)

install(TARGETS idg-knc
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib/static
)

install(
    FILES
    idg-knc.h
    DESTINATION
    include
)
endif(BUILD_LIB_KNC)
