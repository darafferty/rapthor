# Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later

project(tests-hybrid-cuda)

set(LINK_LIBRARIES ${LINK_LIBRARIES} idg-common idg-hybrid-cuda)

add_subdirectory(GenericOptimized)
add_subdirectory(UnifiedOptimized)
