project(hybrid-cuda-generic-optimized)

# Set headers
include_directories (
    ${IDG_CUDA_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIR}
)

# Set headers
set (
    ${PROJECT_NAME}_headers
    GenericOptimized.h
)

# Set sources
set (
    ${PROJECT_NAME}_sources
    GenericOptimized.cpp
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
   include/Hybrid/CUDA/GenericOptimized
)
