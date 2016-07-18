project(cpu-optimized)

# sources and header files
set (${PROJECT_NAME}_headers
  Optimized.h
)

set (${PROJECT_NAME}_sources
  Optimized.cpp
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
    ${CMAKE_SOURCE_DIR}/src/common/Types.h
    kernels/Math.h
    DESTINATION
    lib/kernels/CPU/Optimized
)

# Copy header files
install(
    FILES
    ${${PROJECT_NAME}_headers}
    DESTINATION
    include/CPU/Optimized
)
