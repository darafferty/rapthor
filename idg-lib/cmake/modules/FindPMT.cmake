# Copyright (C) 2023 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later

# CMake find_package() modules for the Power Measurement Toolkit (PMT) library.
# https://git.astron.nl/RD/pmt
#
# Variables used by this module:
#  PMT_ROOT_DIR    - PMT root directory
#
# Variables defined by this module:
#  PMT_FOUND       - system has PMT
#  PMT_INCLUDE_DIR - the PMT include directory
#  PMT_LIBRARY     - link these to use PMT

find_package(PackageHandleStandardArgs)

find_path(
  PMT_ROOT_DIR
  NAMES include/pmt.h
  PATHS ENV PMT_ROOT)

find_path(
  PMT_INCLUDE_DIR
  NAMES pmt.h
  HINTS ${PMT_ROOT_DIR}
  PATH_SUFFIXES include)

find_library(
  PMT_LIBRARY
  NAMES pmt
  HINTS ${PMT_ROOT_DIR}
  PATH_SUFFIXES lib)

mark_as_advanced(PMT_ROOT_DIR)

find_package_handle_standard_args(PMT DEFAULT_MSG PMT_LIBRARY PMT_INCLUDE_DIR)
