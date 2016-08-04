project(hybrid-cuda)

# Set headers
include_directories (
    ${CMAKE_SOURCE_DIR}/src/CUDA/common
    ${CUDA_INCLUDE_DIRS}
)

# Set headers
set (
    ${PROJECT_NAME}_headers
    HybridCUDA.h
)

# Set sources
set (
    ${PROJECT_NAME}_sources
    HybridCUDA.cpp
)

# Set build target
add_library(
    ${PROJECT_NAME} OBJECT ${${PROJECT_NAME}_headers}
                           ${${PROJECT_NAME}_sources}
)
if (NOT BUILD_STATIC_LIBS)
set_target_properties (
    ${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-fPIC"
)
endif()

# Set install
install (
   FILES
   ${${PROJECT_NAME}_headers}
   DESTINATION
   include/Hybrid/CUDA
)
