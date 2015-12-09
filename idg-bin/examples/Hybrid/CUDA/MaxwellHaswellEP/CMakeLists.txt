project(example-hybrid-maxwellhaswellep.x)

# Set external header files
include_directories(
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/utility
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
    idg-hybrid-cuda
    ${CMAKE_DL_LIBS}
    ${CUDA_CUFFT_LIBRARIES}
    cuda
)

# install
install(TARGETS ${PROJECT_NAME}
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib/static
)
