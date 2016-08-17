project(api)

# add subfolders
if (BUILD_LIB_CPU)
    add_subdirectory(CPU)
    add_subdirectory(WSClean)
endif()
