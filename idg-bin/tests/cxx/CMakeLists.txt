# Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later

project(cxx)

# set utility include directory
include_directories(${IDG_BIN_UTILITY_INCLUDE_DIR})

# set common include directory
include_directories(common)

# add subfolders
if (BUILD_LIB_CPU)
    # set common link libraries
    set(LINK_LIBRARIES idg-util idg-cpu)
    add_subdirectory(CPU)
    # CUDA tests need to compare against CPU reference
    if (BUILD_LIB_CUDA)
        add_subdirectory(CUDA)
    endif()
endif()
add_subdirectory(Hybrid)
if (BUILD_LIB_OPENCL)
    add_subdirectory(OpenCL)
endif()
