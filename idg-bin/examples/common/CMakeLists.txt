project(idg-examples-common)

# external header files needed
include_directories (
    ${CMAKE_SOURCE_DIR}/examples/common/external
    ${CMAKE_SOURCE_DIR}/examples/common/utility
)

# sources and header files
set (${PROJECT_NAME}_headers 
  utility/Arguments.h
  utility/Init.h
  external/uvwsim.h
)

set (${PROJECT_NAME}_sources 
  utility/Init.cpp
  external/uvwsim.cpp
)

# create a common library 
add_library(
    ${PROJECT_NAME} ${${PROJECT_NAME}_headers}
                    ${${PROJECT_NAME}_sources}
)

# linker
target_link_libraries(${PROJECT_NAME} idg-examples-common)

