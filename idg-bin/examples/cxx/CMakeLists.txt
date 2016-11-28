project(cxx)

# add subfolders
add_subdirectory(CPU)
add_subdirectory(Hybrid)
if (BUILD_LIB_CUDA)
add_subdirectory(CUDA)
endif ()
if (BUILD_LIB_OPENCL)
add_subdirectory(OpenCL)
endif ()
