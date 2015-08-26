project(cpu-haswellep.x)

# Set external header files
include_directories(
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/examples/external
  ${CMAKE_SOURCE_DIR}/examples/utility
  ${CMAKE_SOURCE_DIR}/src/CPU/reference
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
  idg-examples-external
  idg-examples-utility
  idg-cpu-haswellep
  idg-cpu-reference 
  idg-common
  ${CMAKE_DL_LIBS}
)
