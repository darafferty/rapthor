project(hybrid-knc)

# external header files needed
include_directories (
  ${CMAKE_SOURCE_DIR}/src
)

# sources and header files
set (${PROJECT_NAME}_headers
  KNC.h
  Kernels.h
)

set (${PROJECT_NAME}_sources
  KNC.cpp
  Kernels.cpp
  kernels/KernelGridder.cpp
  kernels/KernelDegridder.cpp
  kernels/KernelFFT.cpp
  kernels/KernelAdder.cpp
  kernels/KernelSplitter.cpp
)

# create library
add_library(
  ${PROJECT_NAME} OBJECT ${${PROJECT_NAME}_headers}
                         ${${PROJECT_NAME}_sources}
)
if (NOT BUILD_STATIC_LIBS)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-fPIC")
endif()

# Copy header files
install(
    FILES
    ${${PROJECT_NAME}_headers}
    DESTINATION
    include/Hybrid/KNC
)
