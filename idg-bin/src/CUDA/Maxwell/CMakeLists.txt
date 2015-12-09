project(cuda-maxwell)

# external header files needed
include_directories (
    ${CUDA_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/src/common
    ${CMAKE_SOURCE_DIR}/src/CUDA/common
)

# sources and header files
set (${PROJECT_NAME}_headers
    Maxwell.h
    KernelsMaxwell.h
)

set (${PROJECT_NAME}_sources
    Maxwell.cpp
)

# create library
add_library(
  ${PROJECT_NAME} OBJECT ${${PROJECT_NAME}_headers}
                         ${${PROJECT_NAME}_sources}
)
if (NOT BUILD_STATIC_LIBS)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-fPIC")
endif()


# Set installation directory
# install(TARGETS ${PROJECT_NAME}
#    RUNTIME DESTINATION bin
#    LIBRARY DESTINATION lib
#    ARCHIVE DESTINATION lib
# )

# Copy kernel source files
# install(
#    FILES
#    kernels/KernelDegridder.cu
#    kernels/KernelGridder.cu
#    kernels/KernelFFT.cu
#    kernels/math.cu
#    kernels/Types.h
#    DESTINATION
#    lib/kernels/GPU/Maxwell
#)

# Copy header files
#install(
#    FILES
#    DESTINATION
#    include/GPU/Maxwell
#)
