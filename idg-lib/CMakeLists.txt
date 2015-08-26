cmake_minimum_required(VERSION 2.8.11)

# set cmake policy
cmake_policy(SET CMP0022 OLD)

# the project
project(ImageDomainGridder)
set (VERSION_MAJOR 0)
set (VERSION_MINOR 1)
set (VERSION_PATCH 0)
set (VERSION "${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}")

# options (to be edited as not functional yet)
option (BUILD_STATIC_LIBS "Build static libraries" OFF)
option (BUILD_EXAMPLES "Build examples executables" OFF) 
option (BUILD_WITH_CUDA "Have CUDA environment" OFF)
option (REPORT_VERBOSE "Report detailed performance metrics" OFF)
option (REPORT_TOTAL "Report total performance metrics" OFF)

# build shared or static libraries (default: shared)
set (BUILD_SHARED_LIBS !${BUILD_STATIC_LIBS})

# Compiler:
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  set(USING_CLANG_CXX_COMPILER TRUE)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  set(USING_GNU_CXX_COMPILER TRUE)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
  set(USING_INTEL_CXX_COMPILER TRUE)
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
# set debug print info
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DDEBUG")

# find CUDA environment 
# (see http://www.cmake.org/cmake/help/v3.0/module/FindCUDA.html)
if (BUILD_WITH_CUDA)
  find_package(CUDA)
  if (CUDA_FOUND)
    message("CUDA VERSION: " ${CUDA_VERSION})
    message("CUDA_INCLUDE_DIRS: " ${CUDA_INCLUDE_DIRS})
    message("CUDA_LIBRARIES: " ${CUDA_LIBRARIES})
    message("CUDA_CUFFT_LIBRARIES: " ${CUDA_CUFFT_LIBRARIES})
    message("CUDA_CUBLAS_LIBRARIES:" ${CUDA_CUBLAS_LIBRARIES})
    # ...
  endif (CUDA_FOUN)
endif (BUILD_WITH_CUDA)

# set default path to build executables and libraries in BINARY_DIR
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

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
