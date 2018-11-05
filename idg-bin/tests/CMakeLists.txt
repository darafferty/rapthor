project(tests)

# add subfolders
add_subdirectory(cxx)
if (BUILD_WITH_PYTHON)
    add_subdirectory(python)
endif()
