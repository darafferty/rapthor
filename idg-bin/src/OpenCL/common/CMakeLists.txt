project(idg-opencl-common)

# external header files needed
include_directories (
  ${_OPENCL_CPP_INCLUDE_DIRS}
  ${_CLFFT_CPP_INCLUDE_DIRS}
  ${CMAKE_SOURCE_DIR}/src/common
  ${CMAKE_SOURCE_DIR}/src//external
)

# sources and header files
set (${PROJECT_NAME}_headers
  OpenCL.h
  Kernels.h
  PerformanceCounter.h
)

set (${PROJECT_NAME}_sources
  OpenCL.cpp
  Kernels.cpp
  PerformanceCounter.cpp
)

# create library
add_library(
  ${PROJECT_NAME} ${${PROJECT_NAME}_headers}
                  ${${PROJECT_NAME}_sources}
)

# Set installation directory
install(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
