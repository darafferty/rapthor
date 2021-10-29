# Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later

project(idg-api)

# make sure C++11 is used (in newer CMake use CXX_STANDARD)
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-std=c++11" COMPILER_SUPPORTS_CXX11)
check_cxx_compiler_flag("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
  message(
    STATUS
      "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler."
  )
endif()

# make sure openmp is used if available
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" AND ${CMAKE_CXX_COMPILER_VERSION}
                                                   VERSION_LESS "3.8.0")
  set(OpenMP_CXX_FLAGS "-fopenmp")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
  find_package(OpenMP REQUIRED)
  if(OPENMP_FOUND)
    # Prevent Intel 2016.2 compiler warning
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
      set(OpenMP_CXX_FLAGS "-qopenmp")
    endif()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS
        "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
  else()
    message(STATUS "OpenMP support not detected")
  endif()
endif()

# set cmake module path
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules)

if(NOT IDG_FOUND)
  find_package(IDG NO_MODULE)
endif()

include_directories(${IDG_INCLUDE_DIR})

include_directories(external)

# Try to use MKL libraries
if(BUILD_WITH_MKL)
  find_package(MKL REQUIRED)
endif()

if(BUILD_WITH_MKL AND (${MKL_FOUND}))
  set(${PROJECT_NAME}_link_libraries ${IDG_LIBRARY} ${MKL_LIBRARIES})
else()
  # Fallback to LAPACK libraries
  # Use LAPACK headers
  find_package(LAPACK REQUIRED)

  set(${PROJECT_NAME}_link_libraries ${IDG_LIBRARY} ${LAPACK_LIBRARIES})
endif()

# build all tests
if(BUILD_TESTING)
  add_subdirectory(tests)
endif()

# sources and header files
set(${PROJECT_NAME}_headers
    Buffer.h
    BufferSet.h
    BulkDegridder.h
    DegridderBuffer.h
    GridderBuffer.h
    taper.h
    Value.h)

set(${PROJECT_NAME}_sources Buffer.cpp BufferSet.cpp BulkDegridder.cpp
                            DegridderBuffer.cpp GridderBuffer.cpp taper.cpp)

# enable rpath
list(APPEND CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

add_library(${PROJECT_NAME} ${${PROJECT_NAME}_headers}
                            ${${PROJECT_NAME}_sources})
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-fPIC" SOVERSION
                                                                       1)
target_link_libraries(${PROJECT_NAME} ${${PROJECT_NAME}_link_libraries})

# Copy header files
install(
  FILES ${${PROJECT_NAME}_headers}
  COMPONENT api
  DESTINATION include/idg-api)

install(
  FILES idg-api.h
  COMPONENT api
  DESTINATION include)

install(
  TARGETS ${PROJECT_NAME}
  COMPONENT api
  EXPORT IDGAPITargets
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib/static
  PUBLIC_HEADER DESTINATION include)

install(EXPORT IDGAPITargets DESTINATION lib/cmake)

# Prepare config files
configure_file("${PROJECT_SOURCE_DIR}/cmake/config/idgapi-config.cmake.in"
               "${PROJECT_BINARY_DIR}/CMakeFiles/idgapi-config.cmake" @ONLY)
configure_file(
  "${PROJECT_SOURCE_DIR}/cmake/config/idgapi-config-version.cmake.in"
  "${PROJECT_BINARY_DIR}/CMakeFiles/idgapi-config-version.cmake" @ONLY)

install(
  FILES "${PROJECT_BINARY_DIR}/CMakeFiles/idgapi-config.cmake"
        "${PROJECT_BINARY_DIR}/CMakeFiles/idgapi-config-version.cmake"
  COMPONENT api
  DESTINATION share/idgapi/cmake)
