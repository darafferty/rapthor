project(common)

# external header files
if (BUILD_WITH_POWERSENSOR)
include_directories(
    ${POWERSENSOR_INCLUDE_DIR}
)
endif()

# sources and header files
set (${PROJECT_NAME}_headers
    ProxyInfo.h
    RuntimeWrapper.h
    auxiliary.h
    Types.h
    KernelsInstance.h
    Proxy.h
	Plan.h
    PlanC.h
    PowerSensor.h
    Report.h
)

set (${PROJECT_NAME}_sources
    ProxyInfo.cpp
    RuntimeWrapper.cpp
    auxiliary.cpp
    Types.cpp
    KernelsInstance.cpp
    Proxy.cpp
	Plan.cpp
    PowerSensor.cpp
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
