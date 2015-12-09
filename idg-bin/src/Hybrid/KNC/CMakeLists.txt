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
  KNC.h
  Kernels.h
)

set (${PROJECT_NAME}_sources
  KNC.cpp
  Kernel.cpp
  kernels/KernelGridder.cpp
  kernels/KernelDegridder.cpp
  kernels/KernelFFT.cpp
  kernels/KernelAdder.cpp
  kernels/KernelSplitter.cpp
)

# create library
add_library(
  ${PROJECT_NAME} STATIC ${${PROJECT_NAME}_headers}
                         ${${PROJECT_NAME}_sources}
)

# Set installation directory
install(TARGETS ${PROJECT_NAME}
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

# Copy header files
install(
    FILES
    idg.h
    KNC.h
    DESTINATION
    include/Hybrid/KNC
)
