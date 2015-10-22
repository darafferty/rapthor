project(idg-cpu-common)

# external header files needed
include_directories (
  ${CMAKE_SOURCE_DIR}/src/common
)

# sources and header files
set (${PROJECT_NAME}_headers
  CPU.h
  Kernels.h
  ${CMAKE_SOURCE_DIR}/src/common/LikwidPowerSensor.h
)

set (${PROJECT_NAME}_sources
  CPU.cpp
  Kernels.cpp
)

# create library
add_library(
  ${PROJECT_NAME} ${${PROJECT_NAME}_headers}
                  ${${PROJECT_NAME}_sources}
  )

# linker
target_link_libraries(${PROJECT_NAME} idg-cpu-common)

# Set installation directory
install(TARGETS ${PROJECT_NAME}
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  )

# Copy header files
install(
    FILES
    CPU.h
    idg-cpu.h
    Kernels.h
    DESTINATION
    include/CPU
)
