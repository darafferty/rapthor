# Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later

project(opencl-generic)

# Set headers
include_directories(${IDG_OpenCL_INCLUDE_DIR})

set(${PROJECT_NAME}_headers Generic.h GenericC.h)

# Set sources
set(${PROJECT_NAME}_sources Generic.cpp GenericC.cpp)

# Set build target
add_library(${PROJECT_NAME} OBJECT ${${PROJECT_NAME}_headers}
                                   ${${PROJECT_NAME}_sources})
if(NOT BUILD_STATIC_LIBS)
  set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-fPIC"
                                                   SOVERSION 0)
endif()
if(${CMAKE_VERSION} VERSION_GREATER 2.8.12)
  target_compile_options(${PROJECT_NAME} PRIVATE -Wno-deprecated-declarations)
endif()

# Set install
install(FILES ${${PROJECT_NAME}_headers} DESTINATION include/OpenCL/Generic)
