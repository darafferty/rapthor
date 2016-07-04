project(cuda-common)

# external header files needed
include_directories (
    ${CUDA_INCLUDE_DIRS}
)

# sources and header files
set (${PROJECT_NAME}_headers
    CUDA.h
    CU.h
    CUFFT.h
    DeviceInstance.h
    PowerRecord.h
)

set (${PROJECT_NAME}_sources
    CUDA.cpp
    CU.cpp
    CUFFT.cpp
    Kernels.cpp
    DeviceInstance.cpp
    PowerRecord.cpp
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
    kernels/KernelDegridder.cu
    kernels/KernelFFT.cu
    kernels/KernelGridder.cu
    kernels/KernelScaler.cu
    kernels/KernelAdder.cu
    kernels/KernelSplitter.cu
    kernels/math.cu
    kernels/Types.h
    DESTINATION
    lib/kernels/CUDA
)

# Copy header files
install(
   FILES
   ${${PROJECT_NAME}_headers}
   DESTINATION
   include/CUDA/common
)
