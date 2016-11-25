project(opencl)

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
