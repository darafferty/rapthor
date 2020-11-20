# Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later

project(examples)

# add subfolders
add_subdirectory(cxx)
if (BUILD_WITH_PYTHON)
add_subdirectory(python)
endif()
