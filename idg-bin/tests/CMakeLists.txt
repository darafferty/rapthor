project(tests)

# add subfolders
if (BUILD_LIB_CPU)
    add_subdirectory(CPU)
endif()
