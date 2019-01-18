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
