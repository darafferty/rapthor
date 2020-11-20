# Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later

project(demo)

# add subfolders
if (BUILD_WITH_PYTHON)
add_subdirectory(python)
endif(BUILD_WITH_PYTHON)
