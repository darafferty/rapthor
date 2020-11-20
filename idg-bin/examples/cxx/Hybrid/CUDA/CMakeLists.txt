# Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later

project(examples-hybrid-cuda)

set(LINK_LIBRARIES ${LINK_LIBRARIES} idg-hybrid-cuda)

set(CMAKE_INSTALL_RPATH ${CUDA_LIB_DIR})
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

add_subdirectory(GenericOptimized)
add_subdirectory(UnifiedOptimized)
