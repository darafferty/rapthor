project(hybrid)

# add subfolders
if (BUILD_LIB_CPU AND BUILD_LIB_CUDA)
    add_subdirectory(CUDA)
endif()
