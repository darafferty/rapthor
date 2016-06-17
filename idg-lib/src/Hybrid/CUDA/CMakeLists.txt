project(hybrid-maxwell)

# external header files needed
include_directories (
    ${CMAKE_SOURCE_DIR}/src/CUDA/common
    ${CMAKE_SOURCE_DIR}/src/CUDA/Generic
    ${CUDA_INCLUDE_DIRS}
)

# sources and header files
set (${PROJECT_NAME}_headers
    HybridCUDA.h
)

set (${PROJECT_NAME}_sources
    HybridCUDA.cpp
)

# create library
add_library(
    ${PROJECT_NAME} OBJECT ${${PROJECT_NAME}_headers}
                           ${${PROJECT_NAME}_sources}
)
if (NOT BUILD_STATIC_LIBS)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-fPIC")
endif()
