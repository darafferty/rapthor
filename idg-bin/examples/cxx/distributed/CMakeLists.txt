# Copyright (C) 2021 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later

# FTI configuration (optional)
option(BUILD_WITH_FTI "Build with FTI" OFF)
if(${BUILD_WITH_FTI})
  find_package(FTI REQUIRED)
  add_compile_definitions("HAVE_FTI")
  # The INTERFACE_INCLUDE_DIRECTORIES target property
  # does not seem to work properly, manually find the
  # FTI header file.
  find_path(FTI_INCLUDE_DIR fti.h HINTS PATH ${FTI_DIR} SUFFIX ../../..)
endif()

# Add subdirectories
add_subdirectory(CPU)
if(BUILD_LIB_CUDA)
  add_subdirectory(CUDA)
  add_subdirectory(Hybrid)
endif()
