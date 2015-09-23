project(src)

# add subfolders
add_subdirectory(common)
add_subdirectory(CPU)
add_subdirectory(Hybrid)
add_subdirectory(python)

if (BUILD_WITH_CUDA)
    add_subdirectory(CUDA)
endif()
if (BUILD_WITH_OPENCL)
    add_subdirectory(OpenCL)
endif()
