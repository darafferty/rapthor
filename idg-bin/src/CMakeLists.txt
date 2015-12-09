project(idg)

# add subfolders
add_subdirectory(common)

if (BUILD_WITH_CPU)
    add_subdirectory(CPU)
endif()

add_subdirectory(Hybrid)

if (BUILD_WITH_CUDA)
    add_subdirectory(CUDA)
endif()
if (BUILD_WITH_OPENCL)
    add_subdirectory(OpenCL)
endif()
if (BUILD_WITH_PYTHON)
    add_subdirectory(python)
endif()
if (BUILD_WITH_BOOST_PYTHON)
    add_subdirectory(boost-python)
endif()



###################################################
# build libidg-common, containing all common code #
###################################################
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
if (BUILD_WITH_CPU)
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
if (BUILD_WITH_CUDA)
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
endif(BUILD_WITH_CUDA)


##############################################################
# build libidg-hybrid-cuda: containing all for CPU/CUDA code #
##############################################################
if (BUILD_WITH_CUDA)
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

install(TARGETS idg-hybrid-cuda
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib/static
)
endif(BUILD_WITH_CUDA)
