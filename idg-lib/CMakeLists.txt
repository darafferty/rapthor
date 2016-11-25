cmake_minimum_required(VERSION 2.8.0)

# set policies
cmake_policy(SET CMP0003 NEW) # 2.6.0
cmake_policy(SET CMP0012 NEW) # 2.8.0

# the project
project(ImageDomainGridder CXX)
set (VERSION_MAJOR 2016)
set (VERSION_MINOR 11)
set (VERSION_PATCH 0)
set (VERSION "${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}")


# set cmake module path
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/modules")


# options
option (BUILD_STATIC_LIBS "Build static libraries" OFF)
option (BUILD_LIB_CPU "Build CPU libraries" ON)
option (BUILD_LIB_CUDA "Have CUDA environment" OFF)
option (BUILD_LIB_OPENCL "Have OpenCL environment" OFF)
option (BUILD_WITH_PYTHON "Have Python environment" OFF)
option (BUILD_WITH_MKL "Build with MKL (if available)" OFF)
option (REPORT_VERBOSE "Report detailed performance metrics" OFF)
option (REPORT_TOTAL "Report total performance metrics" OFF)
option (REPORT_OPS "Report performance in ops (fma, sincos) instead of flops (add, mul)" OFF)


# build shared or static libraries (default: shared)
if (BUILD_STATIC_LIBS)
set (BUILD_SHARED_LIBS FALSE)
else()
set (BUILD_SHARED_LIBS TRUE)
endif()

# Installation
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR} CACHE PATH
    "IDG install prefix" FORCE)
endif()


# Compiler stettings:
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Default build type." FORCE)
endif()
# set debug print info
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG")

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  set(CLANG_CXX_COMPILER TRUE)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  set(GNU_CXX_COMPILER TRUE)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
  set(INTEL_CXX_COMPILER TRUE)
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mkl")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
  set(MSVC_CXX_COMPILER TRUE)
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
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" AND ${CMAKE_CXX_COMPILER_VERSION} VERSION_LESS "3.8.0")
    set (OpenMP_CXX_FLAGS "-fopenmp")
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    find_package(OpenMP REQUIRED)
    if (OPENMP_FOUND)
        # Prevent Intel 2016.2 compiler warning
        if (INTEL_CXX_COMPILER)
        set (OpenMP_CXX_FLAGS "-qopenmp")
        endif()
        set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    else()
        message(STATUS "OpenMP support not detected")
    endif()
endif()


# find OpenCL environment
if (BUILD_LIB_OPENCL)
    find_package(OpenCL)
    find_package(clFFT)
endif (BUILD_LIB_OPENCL)

# write config file, included in all files of the project
configure_file (
  "${PROJECT_SOURCE_DIR}/cmake/config/idg-config.h.in"
  "${PROJECT_BINARY_DIR}/src/idg-config.h"
  )
include_directories (${CMAKE_BINARY_DIR}/src)
# Copy header files
install(
    FILES
    ${PROJECT_BINARY_DIR}/src/idg-config.h
    DESTINATION
    include
)


# add main include folder
include_directories (${CMAKE_SOURCE_DIR}/src)

# write file to initialize enoviroment
configure_file (
  "${PROJECT_SOURCE_DIR}/cmake/config/init-environment.sh.in"
  "${PROJECT_BINARY_DIR}/init-environment.sh"
  )


# add subfolders, after specifying the targets to build
add_subdirectory(src)

#####################################################
# Create IDGConfig.cmake and IDGConfigVersion.cmake #
# for other cmake projects to make use of IDG       #
# See: https://cmake.org/Wiki/CMake/Tutorials/      #
# How_to_create_a_ProjectConfig.cmake_file          #
#####################################################
set(TARGETS_TO_BUILD idg)
set(TARGETS_TO_BUILD ${TARGETS_TO_BUILD} idg-common)
set(TARGETS_TO_BUILD ${TARGETS_TO_BUILD} idg-powersensor)
if (BUILD_LIB_CPU)
set(TARGETS_TO_BUILD ${TARGETS_TO_BUILD} idg-cpu)
endif()
if (BUILD_LIB_CUDA)
set(TARGETS_TO_BUILD ${TARGETS_TO_BUILD} idg-cuda)
endif()
if (BUILD_LIB_CPU AND BUILD_LIB_CUDA)
set(TARGETS_TO_BUILD ${TARGETS_TO_BUILD} idg-hybrid-cuda)
endif()
if (BUILD_LIB_OPENCL)
set(TARGETS_TO_BUILD ${TARGETS_TO_BUILD} idg-opencl)
include_directories(
    ${OPENCL_INCLUDE_DIRS}
    ${CLFFT_INCLUDE_DIRS}
)
endif()
if (BUILD_LIB_CPU)
set(TARGETS_TO_BUILD ${TARGETS_TO_BUILD} idg-fft)
endif()
export(TARGETS ${TARGETS_TO_BUILD}
       FILE "${PROJECT_BINARY_DIR}/CMakeFiles/IDGTargets.cmake")

install(EXPORT IDGTargets
        DESTINATION lib/cmake)

configure_file(
    "${PROJECT_SOURCE_DIR}/cmake/config/IDGConfig.cmake.in"
    "${PROJECT_BINARY_DIR}/CMakeFiles/IDGConfig.cmake" @ONLY)

configure_file(
    "${PROJECT_SOURCE_DIR}/cmake/config/IDGConfigVersion.cmake.in"
    "${PROJECT_BINARY_DIR}/CMakeFiles/IDGConfigVersion.cmake" @ONLY)

install(FILES
    "${PROJECT_BINARY_DIR}/CMakeFiles/IDGConfig.cmake"
    "${PROJECT_BINARY_DIR}/CMakeFiles/IDGConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_PREFIX})
