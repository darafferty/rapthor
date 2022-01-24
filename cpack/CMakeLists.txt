# Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later

# Ensure that we use /usr as install prefix when packaging.
set(CPACK_PACKAGING_INSTALL_PREFIX "/usr")

# Base list of components.
set(CPACK_COMPONENTS_ALL api base util
                         # examples
)

# Dependencies between base components.
set(CPACK_COMPONENT_API_DEPENDS base)
# set(CPACK_COMPONENT_UTIL_DEPENDS base)
# set(CPACK_COMPONENT_EXAMPLES_DEPENDS api util)

# Depending on the way the project is configured, more components and more
# dependencies between components will be defined.
if(BUILD_LIB_CPU)
  list(APPEND CPACK_COMPONENTS_ALL cpu)
  list(APPEND CPACK_COMPONENT_API_DEPENDS cpu)
  set(CPACK_COMPONENT_CPU_DEPENDS base)
endif()
if(BUILD_LIB_CUDA)
  list(APPEND CPACK_COMPONENTS_ALL gpu)
  list(APPEND CPACK_COMPONENT_API_DEPENDS gpu)
  set(CPACK_COMPONENT_GPU_DEPENDS base)
endif()
if(BUILD_LIB_CPU AND BUILD_LIB_CUDA)
  list(APPEND CPACK_COMPONENTS_ALL hybrid)
  list(APPEND CPACK_COMPONENT_API_DEPENDS hybrid)
  set(CPACK_COMPONENT_HYBRID_DEPENDS cpu gpu)
endif()

if(BUILD_WITH_PYTHON)
  list(APPEND CPACK_COMPONENTS_ALL python)
  set(CPACK_DEBIAN_PYTHON_PACKAGE_DEPENDS "python3, python3-numpy")
  if(BUILD_LIB_CPU)
    list(APPEND CPACK_COMPONENT_PYTHON_DEPENDS cpu)
  endif()
  if(BUILD_LIB_CUDA)
    list(APPEND CPACK_COMPONENT_PYTHON_DEPENDS gpu)
  endif()
  if(BUILD_LIB_CPU AND BUILD_LIB_CUDA)
    list(APPEND CPACK_COMPONENT_PYTHON_DEPENDS hybrid)
  endif()
endif()

# if(BUILD_WITH_DEMOS)
#   list(APPEND CPACK_COMPONENTS_ALL demos)
#   set(CPACK_COMPONENT_DEMO_DEPENDS python)
# endif()

# Enable creation of separate packages, instead of one monolithic package
set(CPACK_DEBIAN_ENABLE_COMPONENT_DEPENDS ON)

# We use git to determine the version string from the latest tag.
find_package(Git)
if(Git_FOUND)
  execute_process(
    COMMAND bash -c "${GIT_EXECUTABLE} describe --tags --dirty"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE VERSION_STRING
    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
endif()
if(NOT VERSION_STRING)
  set(VERSION_STRING ${IDG_VERSION})
endif()

include(${CMAKE_CURRENT_SOURCE_DIR}/DetermineTargetCPU.cmake)

# Set package name: encode target CPU, if defined, in package name.
if(DEFINED IDENTIFIED_TARGET_CPU)
  set(CPACK_PACKAGE_NAME "idg-${IDENTIFIED_TARGET_CPU}")
else()
  set(CPACK_PACKAGE_NAME "idg")
endif()

set(CPACK_PACKAGE_VENDOR "ASTRON")
set(CPACK_PACKAGE_VERSION "${VERSION_STRING}")

set(CPACK_PACKAGE_DESCRIPTION_FILE "${CMAKE_SOURCE_DIR}/README.md")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Image Domain Gridder")
set(CPACK_PACKAGE_HOMEPAGE_URL "https://git.astron.nl/RD/idg")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_SOURCE_DIR}/LICENSE")

set(CPACK_GENERATOR "DEB")
set(CPACK_SOURCE_GENERATOR "TGZ")

# Create separate packages for each component
set(CPACK_DEB_COMPONENT_INSTALL ON)

set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE "amd64")
set(CPACK_DEBIAN_FILE_NAME DEB-DEFAULT)
set(CPACK_DEBIAN_PACKAGE_MAINTAINER "deb-packages@astron.nl")
set(CPACK_DEBIAN_PACKAGE_SECTION "science")
set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS ON)

# Determine list of conflicting package names
# Iterate over all components, each component has its own package
foreach(_component ${CPACK_COMPONENTS_ALL})
  string(TOUPPER ${_component} _COMPONENT)
  if(DEFINED IDENTIFIED_TARGET_CPU)
    set(_conflicts idg-${_component})
  else()
    set(_conflicts)
  endif()
  # Packages not built for current target CPU always conflict
  foreach(_cpu ${KNOWN_TARGET_CPUS})
    if(NOT "${_cpu}" STREQUAL "${IDENTIFIED_TARGET_CPU}")
      if("${_conflicts}" STREQUAL "")
        set(_conflicts "idg-${_cpu}-${_component}")
      else()
        set(_conflicts "${_conflicts}, idg-${_cpu}-${_component}")
      endif()
    endif()
  endforeach()
  set(CPACK_DEBIAN_${_COMPONENT}_PACKAGE_CONFLICTS "${_conflicts}")
endforeach()

include(CPack)

message(STATUS "Package name: ${CPACK_PACKAGE_NAME}")
message(STATUS "Package version: ${CPACK_PACKAGE_VERSION}")
message(STATUS "Package components: ${CPACK_COMPONENTS_ALL}")
