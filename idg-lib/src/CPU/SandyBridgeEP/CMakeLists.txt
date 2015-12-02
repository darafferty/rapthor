project(idg-cpu-sandybridgeep)

# external header files needed
include_directories (
  ${CMAKE_SOURCE_DIR}/src/common
  ${CMAKE_SOURCE_DIR}/src/CPU/common
)

# sources and header files
set (${PROJECT_NAME}_headers
  SandyBridgeEP.h
)

set (${PROJECT_NAME}_sources
  SandyBridgeEP.cpp
)

# create library
add_library(
  ${PROJECT_NAME} OBJECT ${${PROJECT_NAME}_headers}
                         ${${PROJECT_NAME}_sources}
)
if (NOT BUILD_STATIC_LIBS)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-fPIC")
endif()

# Copy kernel source files
install(
    FILES
    kernels/KernelAdder.cpp
    kernels/KernelDegridder.cpp
    kernels/KernelFFT.cpp
    kernels/KernelGridder.cpp
    kernels/KernelSplitter.cpp
    DESTINATION
    lib/kernels/CPU/SandyBridgeEP
)

# Copy header files
install(
    FILES
    idg.h
    SandyBridgeEP.h
    DESTINATION
    include/CPU/SandyBridgeEP
)
