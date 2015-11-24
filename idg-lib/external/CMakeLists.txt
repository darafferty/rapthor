project(idg-external)

# external header files needed
include_directories (
    ${CMAKE_SOURCE_DIR}/external
)

# sources and header files
set (${PROJECT_NAME}_headers 
  uvwsim.h
)

set (${PROJECT_NAME}_sources 
  uvwsim.cpp
)

# create a common library 
add_library(
    ${PROJECT_NAME} ${${PROJECT_NAME}_headers}
                    ${${PROJECT_NAME}_sources}
)

# linker
target_link_libraries(${PROJECT_NAME} idg-external)
