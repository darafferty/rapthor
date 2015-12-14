project(tests)

# add subfolders
if (BUILD_LIB_CPU)
    add_subdirectory(CPU)
endif()

if (BUILD_LIB_KNC)
    add_subdirectory(Hybrid)
endif()
