project(tests-hybrid-cuda)

set(LINK_LIBRARIES ${LINK_LIBRARIES} idg-common idg-hybrid-cuda)

add_subdirectory(GenericOptimized)
add_subdirectory(UnifiedOptimized)
