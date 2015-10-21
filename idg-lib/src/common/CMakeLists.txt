project(idg-common)

# sources and header files
set (${PROJECT_NAME}_headers 
  AbstractProxy.h
  CompilerEnvironment.h
  Parameters.h
  ProxyInfo.h
  RuntimeWrapper.h
  auxiliary.h
  PowerSensor.h
  likwidPowerSensor.h
)

set (${PROJECT_NAME}_sources 
  CompilerEnvironment.cpp 
  Parameters.cpp 
  ProxyInfo.cpp 
  RuntimeWrapper.cpp 
  auxiliary.cpp
  PowerSensor.cpp
  likwidPowerSensor.cpp
)

# create a common library 
# add_library(
#    ${PROJECT_NAME} SHARED ${${PROJECT_NAME}_headers}
#                           ${${PROJECT_NAME}_sources}
#)
add_library(
    ${PROJECT_NAME} ${${PROJECT_NAME}_headers}
                    ${${PROJECT_NAME}_sources}
)

# linker
target_link_libraries(
    ${PROJECT_NAME}
    idg-common
    if (BUILD_WITH_LIKIWD_POWER_SENSOR)
        numa
        /cm/shared/package/likwid-4.0.0/lib/liblikwid.so
    endif()
)

# installer
install(TARGETS ${PROJECT_NAME} 
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib/static
)
