# Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later

# - Try to find MKL
# This module tries to find an MKL library on your system.
#
# It looks for MKL libraries in your LD_LIBRARY_PATH and tries
# to resolve the include directory relative to the libraries.
# As an alternative, specify MKL_LIB and MKL_INCLUDE manually
#
# Once done this will define
#  MKL_FOUND        - system has MKL
#  MKL_INCLUDE_DIRS - the MKL include directory
#  MKL_LIBRARIES    - link these to use MKL

find_package(PackageHandleStandardArgs)

# Find libmkl_core.so
find_library(MKL_LIBRARIES
    NAMES mkl_rt
    HINTS ENV MKL_LIB
    HINTS ENV LD_LIBRARY_PATH
)
get_filename_component(MKL_LIB_DIR ${MKL_LIBRARIES} PATH)

# Look for MKL header files in default location (relative to library)
get_filename_component(_MKL_INCLUDE_DIR ${MKL_LIB_DIR}/../../include ABSOLUTE)
find_path(MKL_INCLUDE_DIRS mkl.h PATHS ${_MKL_INCLUDE_DIR})

# Look for MKL header files in $MKL_INCLUDE directory
set(_MKL_INCLUDE_DIR $ENV{MKL_INCLUDE})
find_path(MKL_INCLUDE_DIRS mkl.h PATHS ${_MKL_INCLUDE_DIR})

find_package_handle_standard_args(MKL DEFAULT_MSG MKL_LIBRARIES MKL_INCLUDE_DIRS)
