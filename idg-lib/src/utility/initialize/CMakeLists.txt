project(utility-initialize)

# external header files needed
include_directories (
  ${CMAKE_SOURCE_DIR}/utility
)

# sources and header files
set (${PROJECT_NAME}_headers 
  Init.h
)

set (${PROJECT_NAME}_sources 
  Init.cpp
)

# create a common library 
add_library(
    ${PROJECT_NAME} OBJECT ${${PROJECT_NAME}_headers}
                           ${${PROJECT_NAME}_sources}
)
if (NOT BUILD_STATIC_LIBS)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-fPIC")
endif()

# Copy header files
install(
    FILES
    ${${PROJECT_NAME}_headers}
    DESTINATION
    include/utility/initialize
)
