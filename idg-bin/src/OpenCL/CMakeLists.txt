project(opencl)

# external header files needed
include_directories (
    ${OPENCL_INCLUDE_DIRS}
    ${CLFFT_INCLUDE_DIRS}
)

# add subfolders
add_subdirectory(common)
add_subdirectory(Generic)
