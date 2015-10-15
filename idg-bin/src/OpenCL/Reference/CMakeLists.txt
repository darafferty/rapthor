project(idg-opencl-reference)

# external header files needed
include_directories (
  ${_OPENCL_CPP_INCLUDE_DIRS}
  ${_CLFFT_CPP_INCLUDE_DIRS}
  ${CMAKE_SOURCE_DIR}/src/common
  ${CMAKE_SOURCE_DIR}/src/OpenCL/common
)

# sources and header files
set (${PROJECT_NAME}_headers
  Reference.h
)

set (${PROJECT_NAME}_sources
  Reference.cpp
)

# create library
add_library(
  ${PROJECT_NAME} ${${PROJECT_NAME}_headers}
                  ${${PROJECT_NAME}_sources}
)

# linker
target_link_libraries(${PROJECT_NAME} idg-opencl-reference)
