project(idg-cuda-common)

# Power sensor 
option (MEASURE_POWER "Measure power" OFF)
if (MEASURE_POWER)
    set(POWER_SENSOR "/dev/ttyUSB0" CACHE FILEPATH "Power sensor")
    set(POWER_FILE "power" CACHE FILEPATH "File to dump power consumption")
endif()

# external header files needed
include_directories (
  ${CUDA_INCLUDE_DIRS}
  ${CMAKE_SOURCE_DIR}/src/common
)

# sources and header files
set (${PROJECT_NAME}_headers
  CU.h
  CUFFT.h
  CUDA.h
  Kernels.h
  Power.h
)

set (${PROJECT_NAME}_sources
  CU.cpp
  CUFFT.cpp
  CUDA.cpp
  Power.cpp
)

# create library
add_library(
  ${PROJECT_NAME} ${${PROJECT_NAME}_headers}
                  ${${PROJECT_NAME}_sources}
  )

# linker
target_link_libraries(${PROJECT_NAME} idg-cuda-common)

if (MEASURE_POWER)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMEASURE_POWER")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DPOWER_SENSOR=${POWER_SENSOR}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DPOWER_FILE=${POWER_FILE}")
endif()

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
    idg-cuda.h
    Kernels.h
    DESTINATION
    include/CUDA
)
