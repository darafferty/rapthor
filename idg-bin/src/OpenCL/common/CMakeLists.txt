project(opencl-common)

# sources and header files
set (${PROJECT_NAME}_headers
    OpenCL.h
    Kernels.h
    PerformanceCounter.h
    Util.h
    DeviceInstance.h
)

set (${PROJECT_NAME}_sources
    OpenCL.cpp
    Kernels.cpp
    PerformanceCounter.cpp
    Util.cpp
    DeviceInstance.cpp
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
    kernels/KernelGridder.cl
    kernels/KernelDegridder.cl
    kernels/KernelAdder.cl
    kernels/KernelSplitter.cl
    kernels/KernelScaler.cl
    kernels/math.cl
    kernels/Types.cl
    DESTINATION
    lib/kernels/OpenCL
)

# Copy header files
install(
    FILES
    ${${PROJECT_NAME}_headers}
    DESTINATION
    include/OpenCL/common
)
