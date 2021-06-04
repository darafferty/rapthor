# Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later

project(examples-hybrid)

if(BUILD_LIB_CPU AND BUILD_LIB_CUDA)
  add_subdirectory(CUDA)
endif()
