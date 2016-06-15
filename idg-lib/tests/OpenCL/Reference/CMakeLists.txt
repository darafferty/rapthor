project(test-opencl-reference.x)

# Set external header files
include_directories(
  ${OPENCL_INCLUDE_DIRS}
  ${CLFFT_INCLUDE_DIRS}
  ${CMAKE_SOURCE_DIR}/src
 )

set (${PROJECT_NAME}_sources
  main.cpp
)

# Set build target
add_executable (${PROJECT_NAME} ${${PROJECT_NAME}_sources})

# link
link_directories (${LIBRARY_OUTPUT_PATH})

set(LINK_LIRBRARIES idg-utility idg-cpu idg-opencl ${CMAKE_DL_LIBS})

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

# test
add_test(NAME ${PROJECT_NAME}
         COMMAND ${CMAKE_BINARY_DIR}/bin/tests/${PROJECT_NAME})
