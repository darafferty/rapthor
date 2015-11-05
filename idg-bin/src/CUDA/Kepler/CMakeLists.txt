project(idg-cuda-kepler)

# external header files needed
include_directories (
    ${CUDA_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/src/common
    ${CMAKE_SOURCE_DIR}/src/CUDA/common
)

# sources and header files
set (${PROJECT_NAME}_headers
    Kepler.h
    ${CMAKE_SOURCE_DIR}/src/CUDA/common/Kernels.h
)

set (${PROJECT_NAME}_sources
    Kepler.cpp
    Kernels.cpp
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

# Copy kernel source files
install(
    FILES
    kernels/KernelAdder.cpp
    kernels/KernelDegridder.cpp
    kernels/KernelFFT.cpp
    kernels/KernelGridder.cpp
    kernels/KernelSplitter.cpp
    DESTINATION
    lib/kernels/GPU/Kepler
)

# Copy header files
install(
    FILES
    idg.h
    Reference.h
    DESTINATION
    include/GPU/Kepler
)
