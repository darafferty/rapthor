project(examples-cuda)

set(LINK_LIBRARIES ${LINK_LIBRARIES} idg-cuda)

set(CMAKE_INSTALL_RPATH ${CUDA_LIB_DIR})
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

add_subdirectory(Generic)
add_subdirectory(Unified)
