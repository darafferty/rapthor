project(examples-hybrid-cuda)

set(LINK_LIBRARIES ${LINK_LIBRARIES} idg-hybrid-cuda)

set(CMAKE_INSTALL_RPATH ${CUDA_LIB_DIR})
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

add_subdirectory(GenericOptimized)
add_subdirectory(UnifiedOptimized)
