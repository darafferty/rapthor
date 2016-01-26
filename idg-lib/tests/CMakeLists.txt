project(tests)

# add subfolders
if (BUILD_LIB_CPU)
    add_subdirectory(CPU)
endif()
add_subdirectory(Hybrid)
if (BUILD_WITH_PYTHON)
    add_subdirectory(python)
endif()
