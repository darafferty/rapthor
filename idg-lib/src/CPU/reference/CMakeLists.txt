project(idg-cpu-reference)

# external header files needed 
include_directories (${PROJECT_SOURCE_DIR}/../../common)

# sources and header files
set (${PROJECT_NAME}_headers 
  CPU.h
  Kernels.h
  Types.h
)

set (${PROJECT_NAME}_sources 
  CPU.cpp
  Kernels.cpp
)

# create library
add_library(
  ${PROJECT_NAME} SHARED ${${PROJECT_NAME}_headers} 
  ${${PROJECT_NAME}_sources}
  )

# linker
# add_subdirectory(${COMMON_DIR} ${COMMON_DIR})
target_link_libraries(${PROJECT_NAME} idg-cpu-reference)

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
    lib/kernels/CPU/reference
)

# Copy header files
install(
    FILES
    ${HEADERS}
    CPU.h
    idg-cpu.h
    Types.h
    Kernels.h
    DESTINATION
    include/CPU/reference
)
