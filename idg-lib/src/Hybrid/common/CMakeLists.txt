project(idg-hybrid-common)

# external header files needed
include_directories (
  ${CMAKE_SOURCE_DIR}/src/common
)

# sources and header files
set (${PROJECT_NAME}_headers
  Kernels.h
)

set (${PROJECT_NAME}_sources
  Kernels.cpp
)

# create library
add_library(
  ${PROJECT_NAME} ${${PROJECT_NAME}_headers}
                  ${${PROJECT_NAME}_sources}
)

# install
install(TARGETS ${PROJECT_NAME}
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  )
