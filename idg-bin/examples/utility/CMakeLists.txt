project(idg-examples-utility)

# external header files needed
include_directories (
    ${CMAKE_SOURCE_DIR}/examples/external
    ${CMAKE_BINARY_DIR}/examples
)

message("test: " ${CMAKE_BINARY_DIR}/examples)
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
    ${PROJECT_NAME} STATIC ${${PROJECT_NAME}_headers}
                           ${${PROJECT_NAME}_sources}
)

# linker
target_link_libraries(${PROJECT_NAME} idg-examples-external)

# installer
# install(TARGETS ${PROJECT_NAME} 
#  RUNTIME DESTINATION bin
#  LIBRARY DESTINATION lib
#  ARCHIVE DESTINATION lib/static
# )

# Copy kernel source files
install(
    FILES
    SKA1_low_ecef.txt
    SKA1_mid_ecef.txt
    SKA1_readme.txt
    SKA1_survey.txt
    DESTINATION
    lib/data
)
