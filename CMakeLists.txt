# Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later

cmake_minimum_required(VERSION 2.8)
project(idg)

option(WRITE_OUT_SCALAR_BEAM "Write scalar beam to scalar_beam.npy" OFF)
option(BUILD_STATIC_LIBS "Build static libraries" OFF)
option(BUILD_WITH_DEMOS "Build with demos" OFF)
option(BUILD_TESTING "" OFF)
include(CTest)

# TODO: integer valued compile variables should be handled more elegantly
# if there is a use case for them. Otherwise, they can be hard-coded.
# This includes NR_POLARIZATIONS and NR_CORRELATIONS, see AST-230
add_compile_options("-DNR_POLARIZATIONS=4")

# build shared or static libraries (default: shared)
if (BUILD_STATIC_LIBS)
set (BUILD_SHARED_LIBS FALSE)
else()
set (BUILD_SHARED_LIBS TRUE)
endif()

# Set rpath to install directory
SET(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib)

set(IDG_FOUND True)
set(IDG_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/idg-lib/src
                    ${CMAKE_BINARY_DIR}/idg-lib/src
                    ${CMAKE_SOURCE_DIR}/idg-util/src
                    ${CMAKE_BINARY_DIR}/idg-util/src )
set(IDG_UTIL_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/idg-util/src
                         ${CMAKE_BINARY_DIR}/idg-util/src )
set(IDG_LIBRARY idg)

# Version
set(GIT_VERSION_HEADER "idg-version.h")
include(cmake/version.cmake)
include_directories(${CMAKE_CURRENT_BINARY_DIR})

# Python
option (BUILD_WITH_PYTHON "Build Python bindings" OFF)

if(${CMAKE_VERSION} VERSION_GREATER "3.12.4")
find_package(Python REQUIRED)
set(PYTHON_INSTALL_DIR ${CMAKE_INSTALL_PREFIX}/lib/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/dist-packages/idg)
else()
find_package(PythonInterp REQUIRED)
set(PYTHON_INSTALL_DIR ${CMAKE_INSTALL_PREFIX}/lib/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/dist-packages/idg)
endif()

if (BUILD_TESTING)
  find_package(Boost COMPONENTS unit_test_framework REQUIRED)
endif()

add_subdirectory("idg-util")
add_subdirectory("idg-lib")
add_subdirectory("idg-bin")
add_subdirectory("idg-api")

# Write environment-module
configure_file (
  "${PROJECT_SOURCE_DIR}/cmake/config/environment-module.in"
  "${PROJECT_BINARY_DIR}/idg.env"
)

# Install environment-module
install(
    FILES
    ${PROJECT_BINARY_DIR}/idg.env
    DESTINATION
    ${CMAKE_INSTALL_PREFIX}/share/
)

# Add integration tests
add_subdirectory(test/integration)
