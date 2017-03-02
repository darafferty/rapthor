project(powersensor)

if(BUILD_LIB_CUDA)
    find_package(NvidiaML)
endif()

# sources and header files
set (${PROJECT_NAME}_headers
    PowerSensor.h
    DummyPowerSensor.h
    ArduinoPowerSensor.h
    LikwidPowerSensor.h
    rapl-read.h
    RaplPowerSensor.h
    NVMLPowerSensor.h
)

set (${PROJECT_NAME}_sources
    DummyPowerSensor.cpp
    ArduinoPowerSensor.cpp
    LikwidPowerSensor.cpp
    rapl-read.cpp
    RaplPowerSensor.cpp
    NVMLPowerSensor.cpp
)

include_directories(
    ${LIKWID_INCLUDE_DIRS} % LikwidPowerSensor
    ${CUDA_INCLUDE_DIRS}   % NVMLPowerSensor
)

add_library(
  ${PROJECT_NAME} OBJECT ${${PROJECT_NAME}_headers}
                         ${${PROJECT_NAME}_sources}
)

set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-fPIC")

# Copy header files
install(
    FILES
    ${${PROJECT_NAME}_headers}
    DESTINATION
    include/powersensor
)
