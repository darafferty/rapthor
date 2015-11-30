project(idg-hybrid-maxwell-haswellep)

# external header files needed
include_directories (
    ${CMAKE_SOURCE_DIR}/src/common
    ${CMAKE_SOURCE_DIR}/src/CPU/common
)

# sources and header files
set (${PROJECT_NAME}_headers
    MaxwellHaswellEP.h
)

set (${PROJECT_NAME}_sources
    MaxwellHaswellEP.cpp
)

# create library
add_library(
    ${PROJECT_NAME} ${${PROJECT_NAME}_headers}
                    ${${PROJECT_NAME}_sources}
)
