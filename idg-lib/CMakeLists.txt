cmake_minimum_required(VERSION 2.8.11)

# set cmake policy
cmake_policy(SET CMP0022 OLD)

# the project
project(ImageDomainGridder)
set (VERSION_MAJOR 0)
set (VERSION_MINOR 1)
set (VERSION_PATCH 0)
set (VERSION "${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}")

# Set compile flags
# make sure C++11 is used (in newer CMake use CXX_STANDARD)
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
    message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

# options (to be edited as not functional yet)
option (BUILD_EXAMPLES "Build examples executables" OFF)
option (HAVE_CUDA "Have CUDA environment" OFF)

# write config file, included in all files of the project
configure_file (
  "${PROJECT_SOURCE_DIR}/idg-config.h.in"
  "${PROJECT_BINARY_DIR}/idg-config.h"
  )
include_directories (${CMAKE_BINARY_DIR})

# add subfolders
add_subdirectory(src)

if (BUILD_EXAMPLES)
add_subdirectory(examples)
endif (BUILD_EXAMPLES)
