project(idg-cpu-haswellep)

# external header files needed
include_directories (
  ${CMAKE_SOURCE_DIR}/src/common
  ${CMAKE_SOURCE_DIR}/src/CPU/reference
)

# sources and header files
set (${PROJECT_NAME}_headers
  HaswellEP.h
)

set (${PROJECT_NAME}_sources
  HaswellEP.cpp
)

# create library
add_library(
  ${PROJECT_NAME} ${${PROJECT_NAME}_headers}
                  ${${PROJECT_NAME}_sources}
  )

# linker
target_link_libraries(${PROJECT_NAME} idg-cpu-haswellep)

# Set installation directory
install(TARGETS ${PROJECT_NAME}
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib/static
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
    lib/kernels/CPU/HaswellEP
)

# Copy header files
install(
    FILES
    idg.h
    HaswellEP.h
    DESTINATION
    include/CPU/HaswellEP
)
