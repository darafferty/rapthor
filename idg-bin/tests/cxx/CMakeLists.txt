# Copyright (C) 2021 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later

project(cxx)

# Set common include directories for all subdirectories.
include_directories(${IDG_BIN_UTILITY_INCLUDE_DIR} common)

# Set the path to the common dir, for use in idg_bin_cxx_test.
# (When invoking idg_bin_cxx_test, CMAKE_CURRENT_SOURCE_DIR will differ.)
set(IDG_BIN_TEST_CXX_COMMON_DIR ${CMAKE_CURRENT_SOURCE_DIR}/common)

# All subdirectories use this function for defining the test.
function(idg_bin_cxx_test)
  set(${PROJECT_NAME}_sources main.cpp
                              ${IDG_BIN_TEST_CXX_COMMON_DIR}/common.cpp)

  add_executable(${PROJECT_NAME} ${${PROJECT_NAME}_sources})

  target_link_libraries(${PROJECT_NAME} ${LINK_LIBRARIES})

  install(
    TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION bin/tests/cxx
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib/static)

  add_built_test(${PROJECT_NAME} LABEL bin_cxx)
endfunction()

if(BUILD_LIB_CPU)
  # set common link libraries
  set(LINK_LIBRARIES idg-util idg-cpu)
  add_subdirectory(CPU)
  # CUDA tests need to compare against CPU reference
  if(BUILD_LIB_CUDA)
    add_subdirectory(CUDA)
  endif()
  # OpenCL tests need to compare against CPU reference
  if(BUILD_LIB_OPENCL)
    add_subdirectory(OpenCL)
  endif()
endif()
add_subdirectory(Hybrid)
