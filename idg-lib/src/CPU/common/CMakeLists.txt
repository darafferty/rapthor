project(cpu-common)

# external header files needed
include_directories (
  ${CMAKE_SOURCE_DIR}/src/common
)

# sources and header files
set (${PROJECT_NAME}_headers
  CPU.h
  Kernels.h
)

set (${PROJECT_NAME}_sources
  CPU.cpp
  Kernels.cpp
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
  include/CPU/common
)
