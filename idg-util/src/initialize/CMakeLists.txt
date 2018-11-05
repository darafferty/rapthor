project(utility-initialize)

# external header files needed
include_directories (
    ${CMAKE_SOURCE_DIR}/src/external
)

# sources and header files
set (${PROJECT_NAME}_headers 
    init.h
    initc.h
)

set (${PROJECT_NAME}_sources 
    init.cpp
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
    include/initialize
)
