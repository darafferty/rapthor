project(example-cuda-maxwell.x)

# Set external header files
include_directories(
  ${CUDA_INCLUDE_DIRS}
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/external
  ${CMAKE_SOURCE_DIR}/utility
  ${CMAKE_SOURCE_DIR}/examples/CUDA/common
  ${CMAKE_SOURCE_DIR}/src/CUDA/common
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
  idg-cuda
  ${CUDA_CUFFT_LIBRARIES}
  cuda
)

# install
install(TARGETS ${PROJECT_NAME}
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib/static
)
