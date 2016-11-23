project(cpu)

# external header files
include_directories(
    ${FFTW3_INCLUDE_DIR}
)

# add subfolders
add_subdirectory(common)
add_subdirectory(Reference)
add_subdirectory(Optimized)
add_subdirectory(KNL)

