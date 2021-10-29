# Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later

project(examples-cuda)

set(LINK_LIBRARIES ${LINK_LIBRARIES} idg-cuda)

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

add_subdirectory(Generic)
add_subdirectory(Unified)
