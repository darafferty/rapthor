# Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later

# the project
project(ImageDomainGridder CXX)

# set cmake module path
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules)

# options
option(BUILD_LIB_CPU "Build CPU libraries" ON)
option(BUILD_LIB_CUDA "Build CUDA libraries" OFF)

# Compiler settings:
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE
      Release
      CACHE STRING "Default build type." FORCE)
endif()

# set debug print info
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG")

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  set(CLANG_CXX_COMPILER TRUE)
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  set(GNU_CXX_COMPILER TRUE)
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
  set(INTEL_CXX_COMPILER TRUE)
  set(CMAKE_EXE_LINKER_FLAGS "-static-intel")
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
  set(MSVC_CXX_COMPILER TRUE)
endif()

# Set the warning level to all and treat all warnings as errors
# TODO: setting of the flags is only applied when using a GNU compiler
if(STRICT_WARNINGS
   AND CMAKE_BUILD_TYPE STREQUAL "Debug"
   AND GNU_CXX_COMPILER)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror")
endif()

# make sure openmp is used if available
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" AND ${CMAKE_CXX_COMPILER_VERSION}
                                                   VERSION_LESS "3.8.0")
  set(OpenMP_CXX_FLAGS "-fopenmp")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
  find_package(OpenMP REQUIRED)
  if(OPENMP_FOUND)
    # Prevent Intel 2016.2 compiler warning
    if(INTEL_CXX_COMPILER)
      set(OpenMP_CXX_FLAGS "-qopenmp")
    endif()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS
        "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
  else()
    message(STATUS "OpenMP support not detected")
  endif()
endif()

# resolve idg library
if(NOT IDG_FOUND)
  find_package(IDG NO_MODULE REQUIRED)
  find_package(IDG_UTIL NO_MODULE REQUIRED)
endif()
include_directories(${IDG_INCLUDE_DIR})
include_directories(${IDG_UTIL_INCLUDE_DIR})

# write config file
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/config/idg-examples-config.h.in"
  "${CMAKE_CURRENT_BINARY_DIR}/idg-examples-config.h")

# add directory of config file to include directories
include_directories(${CMAKE_CURRENT_BINARY_DIR})

# build all examples
add_subdirectory(examples)

# build all tests
if(BUILD_TESTING)
  add_subdirectory(tests)
endif()

# build all demos
if(BUILD_WITH_DEMOS)
  add_subdirectory(demo)
endif()
