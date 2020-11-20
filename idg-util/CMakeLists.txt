# Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later

cmake_minimum_required(VERSION 2.8.0)

# Installation
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR} CACHE PATH
    "IDG install prefix" FORCE)
endif()

# Compiler settings:
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Default build type." FORCE)
endif()

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  set(CLANG_CXX_COMPILER TRUE)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  set(GNU_CXX_COMPILER TRUE)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
  set(INTEL_CXX_COMPILER TRUE)
  set (CMAKE_EXE_LINKER_FLAGS "-static-intel")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
  set(MSVC_CXX_COMPILER TRUE)
endif()

# Set the warning level to all and treat all warnings as errors
# TODO: setting of the flags is only applied when using a GNU compiler
if (CMAKE_BUILD_TYPE STREQUAL "Debug" AND GNU_CXX_COMPILER)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror")
endif()

# make sure C++11 is used (in newer CMake use CXX_STANDARD)
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
if(COMPILER_SUPPORTS_CXX11)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
else()
    message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support.")
endif()

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    set (CMAKE_SHARED_LINKER_FLAGS "-static-intel")
endif()

# make sure OpenMP is used
find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    if (INTEL_CXX_COMPILER)
        set (OpenMP_CXX_FLAGS "-qopenmp")
    endif()
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else()
    message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no OpenMP support.")
endif()

# set debug print info
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG")

# resolve idg library
if (NOT IDG_FOUND)
    find_package(IDG NO_MODULE REQUIRED)
endif()
include_directories(${IDG_INCLUDE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

# add directory of config file to include directories
include_directories(${CMAKE_CURRENT_BINARY_DIR}/src)

# write config file
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/config/idg-util-config.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/src/idg-util-config.h"
)

# install config file
install(
    FILES
    "${CMAKE_CURRENT_BINARY_DIR}/src/idg-util-config.h"
    DESTINATION
    "${CMAKE_INSTALL_PREFIX}/include"
)

# add subfolders
add_subdirectory(data)
add_subdirectory(src)

# Create idg-targets.cmake
export(
    TARGETS idg-util
    FILE ${PROJECT_BINARY_DIR}/CMakeFiles/idg-targets.cmake
)

install(
    EXPORT idg-util-targets
    DESTINATION lib/cmake
)

# Create idg-util-config.cmake
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/config/idg-util-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/idg-util-config.cmake @ONLY)

install(
    FILES ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/idg-util-config.cmake
    DESTINATION ${CMAKE_INSTALL_PREFIX}/share)
