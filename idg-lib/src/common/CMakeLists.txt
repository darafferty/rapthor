project(common)

# sources and header files
set (${PROJECT_NAME}_headers 
  CompilerEnvironment.h
  Proxy.h
  Parameters.h
  ProxyInfo.h
  RuntimeWrapper.h
  PowerSensor.h
  LikwidPowerSensor.h
  auxiliary.h
  Types.h
)

set (${PROJECT_NAME}_sources 
  CompilerEnvironment.cpp
  Proxy.cpp
  Parameters.cpp
  ProxyInfo.cpp 
  RuntimeWrapper.cpp
  PowerSensor.cpp
  LikwidPowerSensor.cpp
  auxiliary.cpp
  Types.cpp
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
    include/common
)
