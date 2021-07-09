# Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later

project(common)

# sources and header files
set(${PROJECT_NAME}_headers
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
    ProxyC.h
    Plan.h
    PlanC.h
    PowerSensor.h
    Report.h
    Math.h
    WTiles.h
    WTiling.h)

set(${PROJECT_NAME}_sources
    ProxyInfo.cpp
    RuntimeWrapper.cpp
    auxiliary.cpp
    Types.cpp
    KernelsInstance.cpp
    Proxy.cpp
    ProxyC.cpp
    Plan.cpp
    PlanC.cpp
    PowerSensor.cpp
    Report.cpp
    WTiles.cpp
    WTiling.cpp)

add_library(${PROJECT_NAME} OBJECT ${${PROJECT_NAME}_headers}
                                   ${${PROJECT_NAME}_sources})
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-fPIC" SOVERSION
                                                                       0)

# Copy header files
install(FILES ${${PROJECT_NAME}_headers} DESTINATION include/common)

# ensure include directory can also be found by tests in the build tree
execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory
                        ${CMAKE_BINARY_DIR}/include)
execute_process(
  COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_CURRENT_SOURCE_DIR}
          ${CMAKE_BINARY_DIR}/include/common)
