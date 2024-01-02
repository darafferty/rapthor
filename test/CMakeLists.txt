# Copyright (C) 2021 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later

add_subdirectory(integration)
add_subdirectory(pyproxies)
if(BUILD_LIB_CUDA)
  add_subdirectory(cudakernels)
endif()
