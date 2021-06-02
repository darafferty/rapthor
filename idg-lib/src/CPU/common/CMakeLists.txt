# Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later

project(cpu-common)

# sources and header files
include_directories(${FFTW3_INCLUDE_DIR})

set(${PROJECT_NAME}_headers CPU.h InstanceCPU.h)

set(${PROJECT_NAME}_sources CPU.cpp InstanceCPU.cpp)

# create library
add_library(${PROJECT_NAME} OBJECT ${${PROJECT_NAME}_headers}
                                   ${${PROJECT_NAME}_sources})
if(NOT BUILD_STATIC_LIBS)
  set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-fPIC"
                                                   SOVERSION 0)
endif()

# Copy header files
install(FILES ${${PROJECT_NAME}_headers} DESTINATION include/CPU/common)
