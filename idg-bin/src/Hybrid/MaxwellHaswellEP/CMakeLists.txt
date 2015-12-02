project(idg-hybrid-maxwell-haswellep)

# external header files needed
include_directories (
    ${CUDA_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/src/common
    ${CMAKE_SOURCE_DIR}/src/CPU/common
    ${CMAKE_SOURCE_DIR}/src/CUDA/common
    ${CMAKE_SOURCE_DIR}/src/CPU/HaswellEP
    ${CMAKE_SOURCE_DIR}/src/CUDA/Maxwell
)

# sources and header files
set (${PROJECT_NAME}_headers
    ${CMAKE_SOURCE_DIR}/src/CUDA/common/CU.h
    MaxwellHaswellEP.h
)

set (${PROJECT_NAME}_sources
    ${CMAKE_SOURCE_DIR}/src/CUDA/common/CU.cpp
    MaxwellHaswellEP.cpp
)

# create library
add_library(
    ${PROJECT_NAME} ${${PROJECT_NAME}_headers}
                    ${${PROJECT_NAME}_sources}
)
