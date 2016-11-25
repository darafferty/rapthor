project(powersensor)

# Find dependencies for Likwid power sensor
find_package(Numa QUIET)
find_package(Likwid QUIET)
if (${NUMA_FOUND} AND ${LIKWID_FOUND})
    set (HAVE_LIKWID TRUE)
    set (HAVE_NUMA TRUE)
else ()
    # These values are not used when both libraries are not found
    unset (LIKWID_INCLUDE_DIRS CACHE)
    unset (LIKWID_LIBRARIES CACHE)
    unset (NUMA_ROOT_DIR CACHE)
    unset (NUMA_INCLUDE_DIR CACHE)
    unset (NUMA_LIBRARY CACHE)
endif ()

# sources and header files
set (${PROJECT_NAME}_headers
    PowerSensor.h
    DummyPowerSensor.h
    ArduinoPowerSensor.h
    LikwidPowerSensor.h
    rapl-read.h
    RaplPowerSensor.h
)

set (${PROJECT_NAME}_sources
    DummyPowerSensor.cpp
    ArduinoPowerSensor.cpp
    LikwidPowerSensor.cpp
    rapl-read.cpp
    RaplPowerSensor.cpp
)

include_directories(
    ${LIKWID_INCLUDE_DIRS}
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
