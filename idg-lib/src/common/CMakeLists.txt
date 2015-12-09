project(common)

# sources and header files
set (${PROJECT_NAME}_headers 
  Proxy.h
  CompilerEnvironment.h
  Parameters.h
  ProxyInfo.h
  RuntimeWrapper.h
  auxiliary.h
)

set (${PROJECT_NAME}_sources 
  Proxy.cpp
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
  ${PROJECT_NAME} OBJECT ${${PROJECT_NAME}_headers}
                         ${${PROJECT_NAME}_sources}
)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-fPIC")


# Copy header files
install(
    FILES
    Proxy.h
    CompilerEnvironment.h
    Parameters.h
    ProxyInfo.h
    RuntimeWrapper.h
    auxiliary.h
    LikwidPowerSensor.h
    DESTINATION
    include/common
)
