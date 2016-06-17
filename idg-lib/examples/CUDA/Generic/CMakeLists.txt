project(cuda-generic.x)

# Set external header files
include_directories(
  ${CUDA_INCLUDE_DIRS}
  ${CMAKE_SOURCE_DIR}/utility
 )

set (${PROJECT_NAME}_sources
  main.cpp
)

# Set build target
add_executable (${PROJECT_NAME} ${${PROJECT_NAME}_sources})

# link
set(LINK_LIRBRARIES idg-utility idg-cuda)

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
