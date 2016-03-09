project(test-cpu-sandybridgeep.x)

# Set external header files
include_directories(
  ${CMAKE_SOURCE_DIR}/utility
 )

set (${PROJECT_NAME}_sources 
  main.cpp
)

# Set build target
add_executable (${PROJECT_NAME} ${${PROJECT_NAME}_sources})

# link
link_directories (${LIBRARY_OUTPUT_PATH}) 

set(LINK_LIRBRARIES idg-utility idg-cpu)

if (BUILD_WITH_LIKWID_POWER_SENSOR)
  set(LINK_LIRBRARIES ${LINK_LIRBRARIES} ${NUMA_LIBRARY} ${LIKWID_LIBRARIES})
endif()

target_link_libraries (
  ${PROJECT_NAME} 
  ${LINK_LIRBRARIES}
)

# install
install(TARGETS ${PROJECT_NAME}
  RUNTIME DESTINATION bin/tests
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib/static
)
