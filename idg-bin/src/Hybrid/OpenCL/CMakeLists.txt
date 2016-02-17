project(hybrid-opencl-reference)

# external header files needed
include_directories (
    ${_OPENCL_CPP_INCLUDE_DIRS}
    ${_CLFFT_CPP_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/src/external
)

# sources and header files
set (${PROJECT_NAME}_headers
    OpenCL.h
)

set (${PROJECT_NAME}_sources
    OpenCL.cpp
)

# create library
add_library(
    ${PROJECT_NAME} OBJECT ${${PROJECT_NAME}_headers}
                           ${${PROJECT_NAME}_sources}
)
if (NOT BUILD_STATIC_LIBS)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-fPIC")
endif()
