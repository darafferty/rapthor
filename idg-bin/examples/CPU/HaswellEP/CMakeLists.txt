project(example-cpu-haswellep.x)

# Set external header files
include_directories(
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/external
  ${CMAKE_SOURCE_DIR}/utility
  ${CMAKE_SOURCE_DIR}/examples/CPU/common
  ${CMAKE_SOURCE_DIR}/src/CPU/common
  ${CMAKE_SOURCE_DIR}/src/common
 )

set (${PROJECT_NAME}_sources 
  main.cpp
)

# Set build target
add_executable (${PROJECT_NAME} ${${PROJECT_NAME}_sources})

# link
link_directories (${LIBRARY_OUTPUT_PATH}) 

target_link_libraries (
  ${PROJECT_NAME} 
  idg-utility
  idg-external
  idg-cpu
  ${CMAKE_DL_LIBS}
)

# install
install(TARGETS ${PROJECT_NAME}
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib/static
)
