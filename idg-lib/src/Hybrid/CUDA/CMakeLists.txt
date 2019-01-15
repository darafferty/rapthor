project(hybrid-cuda)

# options
option (REGISTER_HOST_MEMORY "Register (page-lock) host memory not managed by proxy" OFF)

# add subfolders
add_subdirectory(GenericOptimized)
add_subdirectory(UnifiedOptimized)
