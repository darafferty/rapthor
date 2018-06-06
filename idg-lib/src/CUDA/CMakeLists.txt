project(cuda)

# resolve dependencies
find_package(CUDA)

# options
option (ENABLE_HOST_REGISTER " Enable CUDA page-locked memory" ON)

# add subfolders
add_subdirectory(common)
add_subdirectory(Generic)
add_subdirectory(Unified)
