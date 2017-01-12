project(common)

# sources and header files
set (${PROJECT_NAME}_headers
    ProxyInfo.h
    RuntimeWrapper.h
    auxiliary.h
    Types.h
    Kernels.h
    Parameters2.h
    Proxy2.h
	Plan.h
)

set (${PROJECT_NAME}_sources
    ProxyInfo.cpp
    RuntimeWrapper.cpp
    auxiliary.cpp
    Types.cpp
    Kernels.cpp
    Parameters2.cpp
    Proxy2.cpp
	Plan.cpp
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
