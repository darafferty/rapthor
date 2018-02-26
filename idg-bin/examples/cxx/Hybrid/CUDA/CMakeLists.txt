project(examples-hybrid-cuda)

set(LINK_LIBRARIES ${LINK_LIBRARIES} idg-hybrid-cuda)

add_subdirectory(GenericOptimized)
add_subdirectory(UnifiedOptimized)
