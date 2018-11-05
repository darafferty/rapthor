project(cuda)

# resolve dependencies
find_package(CUDA)

# add subfolders
add_subdirectory(common)
add_subdirectory(Generic)
add_subdirectory(Unified)
