project(examples-cuda)

# external header files
include_directories(
    ${FFTW3_INCLUDE_DIR}
)

# add subfolders
add_subdirectory(Jetson)
add_subdirectory(Generic)
