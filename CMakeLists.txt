# Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later

cmake_minimum_required(VERSION 3.8)

set(IDG_VERSION "1.0.0")
if(IDG_VERSION MATCHES "^([0-9]+)\\.([0-9]+)\\.([0-9]+)")
  set(IDG_VERSION_MAJOR "${CMAKE_MATCH_1}")
  set(IDG_VERSION_MINOR "${CMAKE_MATCH_2}")
  set(IDG_VERSION_PATCH "${CMAKE_MATCH_3}")
else()
  message(FATAL_ERROR "Failed to parse IDG_VERSION='${IDG_VERSION}'")
endif()

project(idg VERSION ${IDG_VERSION})

option(WRITE_OUT_SCALAR_BEAM "Write scalar beam to scalar_beam.npy" OFF)
option(BUILD_STATIC_LIBS "Build static libraries" OFF)
option(BUILD_WITH_DEMOS "Build with demos" OFF)
option(BUILD_WITH_MPI "Build with MPI" OFF)
option(BUILD_PACKAGES "Build package(s)" OFF)
option(BUILD_TESTING "" OFF)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/modules"
                      ${CMAKE_MODULE_PATH})

include(CTest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)

# build shared or static libraries (default: shared)
if(BUILD_STATIC_LIBS)
  set(BUILD_SHARED_LIBS FALSE)
else()
  set(BUILD_SHARED_LIBS TRUE)
endif()

# Set rpath to install directory
set(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(IDG_FOUND True)
set(IDG_INCLUDE_DIR
    ${CMAKE_SOURCE_DIR}/idg-lib/src ${CMAKE_BINARY_DIR}/idg-lib/src
    ${CMAKE_SOURCE_DIR}/idg-util/src ${CMAKE_BINARY_DIR}/idg-util/src)
set(IDG_UTIL_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/idg-util/src
                         ${CMAKE_BINARY_DIR}/idg-util/src)
set(IDG_LIBRARY idg)

# Version. idg-lib puts the version info in idg-config.h.in.
include(cmake/version.cmake)
if(GIT_TAG)
  if(${IDG_VERSION} VERSION_LESS ${GIT_TAG})
    message(
      FATAL_ERROR
        "The IDG version is less than the most recent git tag ${GIT_TAG}. Please check the version and the tag."
    )
  endif()

  # For tag pipelines on CI, the version should always match the tag.
  if(DEFINED ENV{CI_COMMIT_TAG})
    if(NOT $ENV{CI_COMMIT_TAG} VERSION_EQUAL ${IDG_VERSION})
      message(
        FATAL_ERROR
          "Git tag $ENV{CI_COMMIT_TAG} does not match the IDG version ${IDG_VERSION}. Please create a matching tag for the master branch."
      )
    endif()
  endif()
endif()

option(PORTABLE "Build portable binaries" OFF)
if(PORTABLE)
  if(DEFINED TARGET_CPU)
    message(WARNING "You have selected to build PORTABLE binaries. "
                    "TARGET_CPU settings will be ignored.")
    unset(TARGET_CPU CACHE)
  endif()
else()
  if(DEFINED TARGET_CPU)
    add_compile_options(-march=${TARGET_CPU})
  else()
    add_compile_options(-march=native)
  endif()
endif()

# MPI
if(BUILD_WITH_MPI)
  find_package(MPI REQUIRED)
endif()

# Python
option(BUILD_WITH_PYTHON "Build Python bindings" OFF)

if(BUILD_WITH_PYTHON)
  if(${CMAKE_VERSION} VERSION_GREATER "3.12.4")
    find_package(Python REQUIRED)
    set(PYTHON_VERSION ${Python_VERSION})
    set(PYTHON_VERSION_MAJOR ${Python_VERSION_MAJOR})
    set(PYTHON_VERSION_MINOR ${Python_VERSION_MINOR})
  else()
    find_package(PythonInterp REQUIRED)
    set(PYTHON_VERSION ${PYTHON_VERSION_STRING})
  endif()

  if(${PYTHON_VERSION} VERSION_LESS "3.6")
    message(
      FATAL_ERROR
        "IDG requires python version >= 3.6, but found ${PYTHON_VERSION}. Using cmake >= 3.12 might fix this error"
    )
  endif()
  if(BUILD_PACKAGES)
    set(PYTHON_INSTALL_DIR
        lib/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/dist-packages/idg
    )
  else()
    set(PYTHON_INSTALL_DIR
        lib/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/site-packages/idg
    )
  endif()
endif()

if(BUILD_TESTING)
  find_package(
    Boost
    COMPONENTS unit_test_framework
    REQUIRED)

  # Add a test and let ctest automatically (re)build the test if needed.
  function(add_built_test TEST_NAME)
    cmake_parse_arguments(ARG "" "LABEL" "" ${ARGN})

    add_test(NAME build-${TEST_NAME}
             COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target
                     ${TEST_NAME})
    # Running multiple build commands in parallel yields race conditions.
    set_tests_properties(build-${TEST_NAME}
                         PROPERTIES FIXTURES_SETUP ${TEST_NAME} RUN_SERIAL true)
    set(JUNIT_ARG)
    if(ARG_LABEL STREQUAL "unit")
      set(JUNIT_ARG -f JUNIT -k unittest_${TEST_NAME}.xml)
    endif()
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME} ${JUNIT_ARG})
    set_tests_properties(${TEST_NAME} PROPERTIES FIXTURES_REQUIRED ${TEST_NAME})
    if(ARG_LABEL)
      set_tests_properties(${TEST_NAME} PROPERTIES LABELS ${ARG_LABEL})
    endif()
    set_property(
      TEST ${TEST_NAME}
      APPEND
      PROPERTY ENVIRONMENT IDG_DATA_DIR=${CMAKE_SOURCE_DIR}/idg-util/data
               IDG_INC_DIR=${CMAKE_BINARY_DIR}/include
               IDG_LIB_DIR=${CMAKE_BINARY_DIR}/lib)
  endfunction()
endif()

add_subdirectory("idg-util")
add_subdirectory("idg-lib")
add_subdirectory("idg-bin")
add_subdirectory("idg-api")
if(BUILD_WITH_PYTHON)
  add_subdirectory("idg-cal")
endif()
add_subdirectory("doc")

# Write environment-module
configure_file("${PROJECT_SOURCE_DIR}/cmake/config/environment-module.in"
               "${PROJECT_BINARY_DIR}/idg.env")

# Install environment-module
install(FILES ${PROJECT_BINARY_DIR}/idg.env DESTINATION share)

# Add integration tests
add_subdirectory(test)

# Packaging
if(BUILD_PACKAGES)
  add_subdirectory(cpack)
endif()
