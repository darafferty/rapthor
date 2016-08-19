project(api)

find_package(OpenCV)

# add subfolders
if (BUILD_LIB_CPU)
    add_subdirectory(CPU)
    if (OpenCV_FOUND)
    add_subdirectory(WSClean)
    endif()
endif()
