project(example-cuda-kepler.x)

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
  idg-external
  idg-utility
  idg-cuda-kepler
  idg-cuda-common
  idg-common
  ${CUDA_CUFFT_LIBRARIES}
  cuda
)
