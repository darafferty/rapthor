project(idg-opencl-common)

# external header files needed
include_directories (
  ${_OPENCL_CPP_INCLUDE_DIRS}
  ${CMAKE_SOURCE_DIR}/src/common
)

# sources and header files
set (${PROJECT_NAME}_headers
  OpenCL.h
  Kernels.h
)

set (${PROJECT_NAME}_sources
  OpenCL.cpp
  Kernels.cpp
)

# create library
add_library(
  ${PROJECT_NAME} ${${PROJECT_NAME}_headers}
                  ${${PROJECT_NAME}_sources}
)

# linker
target_link_libraries(${PROJECT_NAME} idg-opencl-common)
