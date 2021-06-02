# Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later

project(cxx)

# set utility include directory
include_directories(${IDG_BIN_UTILITY_INCLUDE_DIR})

# set common include directory
include_directories(common)

# set common link libraries
set(LINK_LIBRARIES idg-util)

# add subfolders
if(BUILD_LIB_CPU)
  add_subdirectory(CPU)
  add_subdirectory(Hybrid)
  add_subdirectory(plan)
endif()
if(BUILD_LIB_CUDA)
  add_subdirectory(CUDA)
endif()
if(BUILD_LIB_OPENCL)
  add_subdirectory(OpenCL)
endif()
if(BUILD_WITH_MPI)
  add_subdirectory(distributed)
endif()
