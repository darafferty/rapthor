project(tests-cuda)

set(LINK_LIBRARIES ${LINK_LIBRARIES} idg-cuda)

add_subdirectory(Generic)
add_subdirectory(Unified)
