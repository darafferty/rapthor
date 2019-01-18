project(opencl-common)

# sources and header files
set (${PROJECT_NAME}_headers
    OpenCL.h
    Util.h
    InstanceOpenCL.h
    PowerRecord.h
)

set (${PROJECT_NAME}_sources
    OpenCL.cpp
    Util.cpp
    InstanceOpenCL.cpp
    PowerRecord.cpp
)

# create library
add_library(
    ${PROJECT_NAME} OBJECT ${${PROJECT_NAME}_headers}
                           ${${PROJECT_NAME}_sources}
)
if (NOT BUILD_STATIC_LIBS)
    set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-fPIC" SOVERSION 0)
endif()
if (${CMAKE_VERSION} VERSION_GREATER 2.8.12)
target_compile_options(${PROJECT_NAME} PRIVATE -Wno-deprecated-declarations)
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
    kernels/types.cl
    DESTINATION
    lib/idg-opencl
)

# Copy header files
install(
    FILES
    ${${PROJECT_NAME}_headers}
    DESTINATION
    include/OpenCL/common
)
