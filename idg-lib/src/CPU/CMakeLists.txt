# Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later

project(cpu)

# add subfolders
add_subdirectory(common)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_subdirectory(Reference)
add_subdirectory(Optimized)
