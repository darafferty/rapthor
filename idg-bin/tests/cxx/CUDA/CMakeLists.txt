# Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later

project(tests-cuda)

set(LINK_LIBRARIES ${LINK_LIBRARIES} idg-cuda)

add_subdirectory(Generic)
add_subdirectory(Unified)
