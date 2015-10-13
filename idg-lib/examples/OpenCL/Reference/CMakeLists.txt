project(opencl-reference.x)

# Set external header files
include_directories(
  ${_OPENCL_CPP_INCLUDE_DIRS}
  ${_CLFFT_CPP_INCLUDE_DIRS}
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/examples/common/external
  ${CMAKE_SOURCE_DIR}/examples/common/utility
  ${CMAKE_SOURCE_DIR}/src/OpenCL/common
  ${CMAKE_SOURCE_DIR}/src/common
  ${CMAKE_SOURCE_DIR}/examples/OpenCL/common
 )

set (${PROJECT_NAME}_sources 
  main.cpp
)

# Set build target
add_executable (${PROJECT_NAME} ${${PROJECT_NAME}_sources})

# link
link_directories (${LIBRARY_OUTPUT_PATH}) 

target_link_libraries (
  ${PROJECT_NAME} 
  idg-examples-common
  idg-opencl-reference
  idg-opencl-common
  idg-common
  ${OPENCL_LIBRARIES}
  ${CLFFT_LIBRARIES}
)
