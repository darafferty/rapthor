cmake_minimum_required(VERSION 2.8)
project(idg)

# Set rpath to install directory
SET(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib)

set(IDG_OnePackage True)
enable_testing()
add_subdirectory("idg-util")
add_subdirectory("idg-lib")
add_subdirectory("idg-bin")
add_subdirectory("idg-api")
