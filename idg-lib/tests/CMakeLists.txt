project(tests)

# add subfolders
if (BUILD_WITH_CPU)
    add_subdirectory(CPU)
endif()
