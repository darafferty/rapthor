project(hybrid-opencl)

# external header files needed
include_directories (
    ${OPENCL_INCLUDE_DIRS}
    ${CLFFT_INCLUDE_DIRS}
)

# sources and header files
set (${PROJECT_NAME}_headers
    HybridOpenCL.h
)

set (${PROJECT_NAME}_sources
    HybridOpenCL.cpp
)

# create library
add_library(
    ${PROJECT_NAME} OBJECT ${${PROJECT_NAME}_headers}
                           ${${PROJECT_NAME}_sources}
)
if (NOT BUILD_STATIC_LIBS)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-fPIC")
endif()
