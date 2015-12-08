project(idg-cuda-common)

# external header files needed
include_directories (
    ${CUDA_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/src/common
)

# sources and header files
set (${PROJECT_NAME}_headers
    CU.h
    CUFFT.h
    CUDA.h
)

set (${PROJECT_NAME}_sources
    CU.cpp
    CUFFT.cpp
    CUDA.cpp
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

# Copy header files
install(
    FILES
    DESTINATION
    include/CUDA
)
