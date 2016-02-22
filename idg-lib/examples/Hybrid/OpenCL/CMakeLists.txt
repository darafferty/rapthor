project(hybrid-opencl-reference.x)

# Set external header files
include_directories(
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/utility
    ${_OPENCL_CPP_INCLUDE_DIRS}
    ${_CLFFT_CPP_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/src/external
 )

set (${PROJECT_NAME}_sources
    main.cpp
)

# Set build target
add_executable (${PROJECT_NAME} ${${PROJECT_NAME}_sources})

# link
link_directories (${LIBRARY_OUTPUT_PATH})

set(LINK_LIRBRARIES idg-utility idg-hybrid-opencl)

if (BUILD_WITH_LIKWID_POWER_SENSOR)
    set(LINK_LIRBRARIES ${LINK_LIRBRARIES} ${NUMA_LIBRARY} ${LIKWID_LIBRARIES})
endif()

target_link_libraries (
    ${PROJECT_NAME}
    ${LINK_LIRBRARIES}
)

# install
install(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION bin/examples
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib/static
)
