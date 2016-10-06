project(examples-api)

# add subfolders
add_subdirectory(Optimized)
if (BUILD_LIB_CUDA)
    add_subdirectory(CUDA)
endif()
