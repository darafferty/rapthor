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

# make sure C++11 is used (in newer CMake use CXX_STANDARD)
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
if(COMPILER_SUPPORTS_CXX11)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
else()
    message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support.")
endif()

# set debug print info
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG")

# resolve idg library
find_package(IDG NO_MODULE REQUIRED)
include_directories(${IDG_INCLUDE_DIR})

# add directory of config file to include directories
include_directories(${CMAKE_CURRENT_BINARY_DIR})

# write config file
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/config/idg-util-config.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/idg-util-config.h"
)

# install config file
install(
    FILES
    "${CMAKE_CURRENT_BINARY_DIR}/idg-util-config.h"
    DESTINATION
    ${CMAKE_INSTALL_PREFIX}/include
)

# add subfolders
add_subdirectory(data)
add_subdirectory(src)
