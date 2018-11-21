project(opencl)

# ignore warnings in OpenCL header file
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-ignored-attributes")

# resolve dependencies
find_package(OpenCL)
find_package(clFFT)

# external header files needed
include_directories (
    ${OPENCL_INCLUDE_DIRS}
    ${CLFFT_INCLUDE_DIRS}
)

# add subfolders
add_subdirectory(common)
add_subdirectory(Generic)
