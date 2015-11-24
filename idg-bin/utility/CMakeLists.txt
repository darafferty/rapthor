project(idg-utility)

# external header files needed
include_directories (
    ${CMAKE_SOURCE_DIR}/external
    ${CMAKE_SOURCE_DIR}/utility
)

# sources and header files
set (${PROJECT_NAME}_headers 
  Arguments.h
  Init.h
)

set (${PROJECT_NAME}_sources 
  Init.cpp
)

# create a common library 
add_library(
    ${PROJECT_NAME} ${${PROJECT_NAME}_headers}
                    ${${PROJECT_NAME}_sources}
)

# linker
target_link_libraries(${PROJECT_NAME} idg-utility)

