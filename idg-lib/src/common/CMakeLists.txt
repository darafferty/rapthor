project(idg-common)

# sources and header files
set (${PROJECT_NAME}_headers 
  AbstractProxy.h
  CompilerEnvironment.h
  Parameters.h
  ProxyInfo.h
  RuntimeWrapper.h
  auxiliary.h
)

set (${PROJECT_NAME}_sources 
  CompilerEnvironment.cpp 
  Parameters.cpp 
  ProxyInfo.cpp 
  RuntimeWrapper.cpp 
  auxiliary.cpp
)

#if (${BUILD_WITH_ARDUINO_POWER_SENSOR})
set (${PROJECT_NAME}_headers
    ${${PROJECT_NAME}_headers}
    PowerSensor.h
)
set (${PROJECT_NAME}_sources
    ${${PROJECT_NAME}_sources}
    PowerSensor.cpp
)

# add files for LikwidPowerSensor
set (${PROJECT_NAME}_headers
    ${${PROJECT_NAME}_headers}
    LikwidPowerSensor.h
)
set (${PROJECT_NAME}_sources
    ${${PROJECT_NAME}_sources}
    LikwidPowerSensor.cpp
)

add_library(
    ${PROJECT_NAME} ${${PROJECT_NAME}_headers}
                    ${${PROJECT_NAME}_sources}
)

if (${BUILD_WITH_LIKWID_POWER_SENSOR})
target_link_libraries(
    ${PROJECT_NAME}
    ${NUMA_LIBRARY}
    ${LIKWID_LIBRARIES}
)
endif()

# installer
install(TARGETS ${PROJECT_NAME} 
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib/static
)
