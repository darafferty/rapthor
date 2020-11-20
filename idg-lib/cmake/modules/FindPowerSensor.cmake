# Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later

# This module tries to find the PowerSensor library on your system
#
# Once done this will define
#  POWERSENSOR_FOUND       - system has PowerSensor
#  POWERSENSOR_INCLUDE_DIR - the PowerSensor include directory
#  POWERSENSOR_LIBRARY     - link these to use PowerSensor

find_package(PackageHandleStandardArgs)

find_path(
    POWERSENSOR_ROOT_DIR
    NAMES include/powersensor.h
    PATHS ENV POWERSENSOR_ROOT
)

find_path(
    POWERSENSOR_INCLUDE_DIR
    NAMES powersensor.h
    HINTS ${POWERSENSOR_ROOT_DIR}
    PATH_SUFFIXES include
)

find_library(
    POWERSENSOR_LIBRARY
    NAMES powersensor
    HINTS ${POWERSENSOR_ROOT_DIR}
    PATH_SUFFIXES lib
)

mark_as_advanced(POWERSENSOR_ROOT_DIR)

find_package_handle_standard_args(PowerSensor DEFAULT_MSG POWERSENSOR_LIBRARY POWERSENSOR_INCLUDE_DIR)
