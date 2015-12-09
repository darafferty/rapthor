project(cuda-maxwell)

# external header files needed
include_directories (
    ${CUDA_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/src
)

# sources and header files
set (${PROJECT_NAME}_headers
    Maxwell.h
    KernelsMaxwell.h
)

set (${PROJECT_NAME}_sources
    Maxwell.cpp
)

# create library
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
   include/CUDA/Maxwell
)
