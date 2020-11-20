# Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later

project(examples-cpu)

set(LINK_LIBRARIES ${LINK_LIBRARIES} idg-cpu)

add_subdirectory(Reference)
add_subdirectory(Optimized)
