project(common)

# sources and header files
set (${PROJECT_NAME}_headers
    Exception.h
    ProxyInfo.h
    RuntimeWrapper.h
    auxiliary.h
    Types.h
    KernelTypes.h
    ArrayTypes.h
    Index.h
    KernelsInstance.h
    Proxy.h
	Plan.h
    PlanC.h
    PowerSensor.h
    Report.h
    Math.h
    WTiles.h
)

set (${PROJECT_NAME}_sources
    ProxyInfo.cpp
    RuntimeWrapper.cpp
    auxiliary.cpp
    Types.cpp
    KernelsInstance.cpp
    Proxy.cpp
    ProxyC.cpp
	Plan.cpp
    PowerSensor.cpp
    Report.cpp
)

add_library(
	${PROJECT_NAME} OBJECT ${${PROJECT_NAME}_headers}
	                       ${${PROJECT_NAME}_sources}
)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-fPIC" SOVERSION 0)

# Copy header files
install(
	FILES
	${${PROJECT_NAME}_headers}
	DESTINATION
	include/common
)
