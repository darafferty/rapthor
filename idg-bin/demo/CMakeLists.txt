project(demo)

# add subfolders
if (BUILD_WITH_PYTHON)
add_subdirectory(python)
endif(BUILD_WITH_PYTHON)

# if found casacore, opencv, ...
find_package(CasaCore)
find_package(OpenCV)

if (CASACORE_FOUND AND OpenCV_FOUND)
add_subdirectory(cxx)
endif()
