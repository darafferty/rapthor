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
set (VERSION_PATCH 2)
set (VERSION "${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}")


# set cmake module path
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/modules")


# options
option (BUILD_STATIC_LIBS "Build static libraries" OFF)
option (BUILD_LIB_CPU "Build CPU libraries" ON)
option (BUILD_LIB_CUDA "Have CUDA environment" OFF)
option (BUILD_LIB_OPENCL "Have OpenCL environment" OFF)
option (BUILD_LIB_KNC "Have Xeon Phi environment" OFF)
option (BUILD_WITH_PYTHON "Have Python environment" OFF)
option (BUILD_WITH_EXAMPLES "Build example executables" OFF)
option (BUILD_WITH_TESTS "Build test executables" OFF)
option (BUILD_WITH_DEMO "Build LOFAR demo program" OFF)
option (REPORT_VERBOSE "Report detailed performance metrics" OFF)
option (REPORT_TOTAL "Report total performance metrics" OFF)


# Likwid power sensor
find_package(Numa QUIET)
find_package(Likwid QUIET)
if (${NUMA_FOUND} AND ${LIKWID_FOUND})
    set (HAVE_LIKWID TRUE)
else ()
    # These values are not used when both libraries are not found
    unset (LIKWID_INCLUDE_DIRS CACHE)
    unset (LIKWID_LIBRARIES CACHE)
    unset (NUMA_ROOT_DIR CACHE)
    unset (NUMA_INCLUDE_DIR CACHE)
    unset (NUMA_LIBRARY CACHE)
endif ()


# build shared or static libraries (default: shared)
if (BUILD_STATIC_LIBS)
set (BUILD_SHARED_LIBS FALSE)
else()
set (BUILD_SHARED_LIBS TRUE)
endif()

# test
if (BUILD_WITH_TESTS)
enable_testing()
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


# find MKL
if (NOT INTEL_CXX_COMPILER)
find_package(MKL QUIET)
if (${MKL_FOUND})
    get_filename_component(MKL_LIB_DIR ${MKL_LIBRARIES} PATH)
    set (HAVE_MKL TRUE)
    set(FFTW3F_LIBRARY ${MKL_LIBRARIES})
    set(FFTW3_LIBRARY ${MKL_LIBRARIES})
else()
    unset (MKL_INCLUDE_DIRS CACHE)
    unset (MKL_LIBRARIES CACHE)
    find_package(FFTW3 COMPONENTS single double)
endif()
endif()

# find CUDA environment
if (BUILD_LIB_CUDA)
    find_package(CUDA)
    find_package(CUDAdriver)
endif (BUILD_LIB_CUDA)

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

if (BUILD_WITH_EXAMPLES)
add_subdirectory(examples)
endif (BUILD_WITH_EXAMPLES)

if (BUILD_WITH_TESTS)
add_subdirectory(tests)
endif (BUILD_WITH_TESTS)

if (BUILD_WITH_DEMO)
add_subdirectory(demo)
endif (BUILD_WITH_DEMO)


#####################################################
# Create IDGConfig.cmake and IDGConfigVersion.cmake #
# for other cmake projects to make use of IDG       #
# See: https://cmake.org/Wiki/CMake/Tutorials/      #
# How_to_create_a_ProjectConfig.cmake_file          #
#####################################################
set(TARGETS_TO_BUILD idg)
set(TARGETS_TO_BUILD ${TARGETS_TO_BUILD} idg-common)
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
endif()
if (BUILD_LIB_KNC)
set(TARGETS_TO_BUILD ${TARGETS_TO_BUILD} idg-knc)
endif()
if (BUILD_WITH_EXAMPLES OR BUILD_WITH_TESTS)
set(TARGETS_TO_BUILD ${TARGETS_TO_BUILD} idg-utility)
endif()
if (BUILD_LIB_CPU OR BUILD_WITH_EXAMPLES OR BUILD_WITH_TESTS)
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


