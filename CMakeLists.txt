cmake_minimum_required(VERSION 2.8)
project(idg)

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
find_package(Python)
set(PYTHON_INSTALL_DIR ${CMAKE_INSTALL_PREFIX}/lib/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/dist-packages/idg)

enable_testing()
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
