project(common)

# sources and header files
set (${PROJECT_NAME}_headers
    Proxy.h
    Parameters.h
    ProxyInfo.h
    RuntimeWrapper.h
    auxiliary.h
    Types.h
    Plan.h
    Kernels.h
    Parameters2.h
    Proxy2.h
	Plan2.h
)

set (${PROJECT_NAME}_sources
    Proxy.cpp
    Parameters.cpp
    ProxyInfo.cpp
    RuntimeWrapper.cpp
    auxiliary.cpp
    Types.cpp
    Plan.cpp
    Kernels.cpp
    Parameters2.cpp
    Proxy2.cpp
	Plan2.cpp
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
