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

if (BUILD_WITH_CPU)
add_library(${PROJECT_NAME}
  $<TARGET_OBJECTS:common-objects>
  $<TARGET_OBJECTS:idg-cpu-common>
  $<TARGET_OBJECTS:idg-cpu-reference>
  $<TARGET_OBJECTS:idg-cpu-haswellep>
  $<TARGET_OBJECTS:idg-cpu-sandybridgeep>
)

install(TARGETS ${PROJECT_NAME}
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib/static
)
endif()
