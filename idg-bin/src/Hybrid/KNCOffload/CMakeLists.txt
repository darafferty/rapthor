project(idg-hybrid-knc)

# write config file, included in all files of the project
configure_file (
  "${PROJECT_SOURCE_DIR}/set-hybrid-knc-env.sh.in"
  "${CMAKE_BINARY_DIR}/set-hybrid-knc-env.sh"
  )

# external header files needed
include_directories (
  ${CMAKE_SOURCE_DIR}/src/common
  ${CMAKE_SOURCE_DIR}/src/CPU/reference
)

# sources and header files
set (${PROJECT_NAME}_headers
  KNCOffload.h
  Kernels.h
)

set (${PROJECT_NAME}_sources
  kernels/KernelGridder.cpp
  kernels/KernelFFT.cpp
  KNCOffload.cpp
)

# create library
add_library(
  ${PROJECT_NAME} ${${PROJECT_NAME}_headers}
                  ${${PROJECT_NAME}_sources}
  )

# linker
target_link_libraries(${PROJECT_NAME} idg-hybrid-knc)

# Set installation directory
install(TARGETS ${PROJECT_NAME}
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib/static
)
