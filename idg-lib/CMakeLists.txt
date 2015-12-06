cmake_minimum_required(VERSION 2.8.7)

# set cmake policy
if (
    "${CMAKE_MAJOR_VERSION}" STREQUAL "2" AND
    "${CMAKE_MINOR_VERSION}" STREQUAL "8" AND
    "${CMAKE_PATCH_VERSION}" STREQUAL "12")
    cmake_policy(SET CMP0022 NEW)
endif()

# the project
project(ImageDomainGridder CXX)
set (VERSION_MAJOR 0)
set (VERSION_MINOR 1)
set (VERSION_PATCH 0)
set (VERSION "${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}")

# set cmake module path
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# options
option (BUILD_STATIC_LIBS "Build static libraries" OFF)
option (BUILD_WITH_CPU "Build CPU libraries" ON)
option (BUILD_WITH_MIC "Have Xeon Phi environment" OFF) 
option (BUILD_WITH_CUDA "Have CUDA environment" OFF)
option (BUILD_WITH_OPENCL "Have OpenCL environment" OFF)
option (BUILD_WITH_PYTHON "Have Python environment" OFF)
option (BUILD_WITH_BOOST_PYTHON "Have Python environment" OFF)
option (BUILD_EXAMPLES "Build example executables" OFF) 
option (BUILD_TESTS "Build test executables" OFF) 
option (REPORT_VERBOSE "Report detailed performance metrics" OFF)
option (REPORT_TOTAL "Report total performance metrics" OFF)
option (BUILD_WITH_ARDUINO_POWER_SENSOR "Measure power using arduino power sensor" OFF)
option (BUILD_WITH_LIKWID_POWER_SENSOR "Measure power using likiwd library" OFF)

# Power sensor
if (BUILD_WITH_ARDUINO_POWER_SENSOR)
    set(POWER_SENSOR "/dev/ttyUSB0" CACHE FILEPATH "Power sensor")
    set(POWER_FILE "power" CACHE FILEPATH "File to dump power consumption")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMEASURE_POWER_ARDUINO")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DPOWER_SENSOR=${POWER_SENSOR}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DPOWER_FILE=${POWER_FILE}")
endif()

# Likwid power sensor
if (BUILD_WITH_LIKWID_POWER_SENSOR)
    find_package(Numa)
    find_package(Likwid)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMEASURE_POWER_LIKWID")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DPOWER_SENSOR=${POWER_SENSOR}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DPOWER_FILE=${POWER_FILE}")
endif()

# build shared or static libraries (default: shared)
if (BUILD_STATIC_LIBS)
set (BUILD_SHARED_LIBS FALSE)
else() 
set (BUILD_SHARED_LIBS TRUE)
endif()

# Compiler stettings:
if (NOT CMAKE_BUILD_TYPE)
   set(CMAKE_BUILD_TYPE "Release")
endif()
# set debug print info
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG")

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  set(USING_CLANG_CXX_COMPILER TRUE)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  set(USING_GNU_CXX_COMPILER TRUE)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
  set(USING_INTEL_CXX_COMPILER TRUE)
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mkl")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
  set(USING_MSVC_CXX_COMPILER TRUE)
endif()

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
# make sure openmp is used if available
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# find CUDA environment 
# (see http://www.cmake.org/cmake/help/v3.0/module/FindCUDA.html)
if (BUILD_WITH_CUDA)
  find_package(CUDA)
endif (BUILD_WITH_CUDA)

# find OpenCL environment
# (see http://www.cmake.org/cmake/help/v3.1/module/FindOpenCL.html)
if (BUILD_WITH_OPENCL)
    find_package(OpenCL)
    find_package(clFFT)
endif (BUILD_WITH_OPENCL)

# set default path to build executables and libraries in BINARY_DIR
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

# write config file, included in all files of the project
configure_file (
  "${PROJECT_SOURCE_DIR}/idg-config.h.in"
  "${PROJECT_BINARY_DIR}/idg-config.h"
  )
include_directories (${CMAKE_BINARY_DIR})

# write file to initialize enoviroment
configure_file (
  "${PROJECT_SOURCE_DIR}/init-environment.sh.in"
  "${PROJECT_BINARY_DIR}/init-environment.sh"
  )

# add subfolders
add_subdirectory(src)

if (BUILD_EXAMPLES OR BUILD_TESTS)
add_subdirectory(external)
add_subdirectory(utility)
endif ()

if (BUILD_EXAMPLES)
add_subdirectory(examples)
endif (BUILD_EXAMPLES)

if (BUILD_TESTS)
add_subdirectory(tests)
endif (BUILD_TESTS)
