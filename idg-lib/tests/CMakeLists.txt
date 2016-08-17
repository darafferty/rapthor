project(tests)

# add subfolders
if (BUILD_LIB_CPU)
    add_subdirectory(CPU)
    add_subdirectory(api)        # TODO: merge with other code
endif()
add_subdirectory(Hybrid)
if (BUILD_LIB_CUDA)
    add_subdirectory(CUDA)
endif()
if (BUILD_LIB_OPENCL)
    add_subdirectory(OpenCL)
endif()
if (BUILD_WITH_PYTHON)
    add_subdirectory(python)
    # add_subdirectory(python-api)  # TODO: merge into python
endif()
