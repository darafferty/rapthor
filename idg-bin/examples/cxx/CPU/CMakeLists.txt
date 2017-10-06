project(examples-cpu)

set(LINK_LIBRARIES ${LINK_LIBRARIES} idg-cpu)

add_subdirectory(Reference)
add_subdirectory(Optimized)
