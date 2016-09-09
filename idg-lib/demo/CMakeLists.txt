project(demo)

# add subfolders
if (BUILD_WITH_PYTHON)
add_subdirectory(python)
endif(BUILD_WITH_PYTHON)

find_package(OpenCV)
if (OpenCV_FOUND)
add_subdirectory(cxx)
endif()
