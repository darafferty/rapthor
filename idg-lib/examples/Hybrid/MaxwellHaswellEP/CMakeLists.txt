project(example-hybrid-maxwellhaswellep.x)

# Set external header files
include_directories(
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/common
    ${CMAKE_SOURCE_DIR}/external
    ${CMAKE_SOURCE_DIR}/utility
    ${CMAKE_SOURCE_DIR}/examples/Hybrid/common
    ${CMAKE_SOURCE_DIR}/src/CPU/common
    ${CMAKE_SOURCE_DIR}/src/CUDA/common
    ${CMAKE_SOURCE_DIR}/src/CPU/HaswellEP
    ${CMAKE_SOURCE_DIR}/src/CUDA/Maxwell
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
    idg-common
    idg-cpu
    idg-cuda-maxwell
    idg-cuda-common
    idg-hybrid-maxwellhaswellep
    ${CMAKE_DL_LIBS}
    ${CUDA_CUFFT_LIBRARIES}
    cuda
)
