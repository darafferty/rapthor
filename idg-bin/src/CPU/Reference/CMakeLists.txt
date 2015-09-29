project(idg-cpu-reference)

# external header files needed
include_directories (
  ${CMAKE_SOURCE_DIR}/src/common
  ${CMAKE_SOURCE_DIR}/src/CPU/common
)

# sources and header files
set (${PROJECT_NAME}_headers
  Reference.h
)

set (${PROJECT_NAME}_sources
  Reference.cpp
)

# create library
add_library(
  ${PROJECT_NAME} ${${PROJECT_NAME}_headers}
                  ${${PROJECT_NAME}_sources}
  )

# linker
target_link_libraries(${PROJECT_NAME} idg-cpu-reference)

# Set installation directory
install(TARGETS ${PROJECT_NAME}
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  )

# Copy kernel source files
install(
    FILES
    kernels/KernelAdder.cpp
    kernels/KernelDegridder.cpp
    kernels/KernelFFT.cpp
    kernels/KernelGridder.cpp
    kernels/KernelSplitter.cpp
    DESTINATION
    lib/idg/kernels/CPU/Reference
)

# Copy header files
install(
    FILES
    idg.h
    Reference.h
    DESTINATION
    include/idg/CPU/Reference
)
