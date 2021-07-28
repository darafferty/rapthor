# Copyright (C) 2021 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later

# NOTE: tPyProxies should only be run after installation
add_test(
  NAME tPyProxies
  COMMAND
    python3 -m pytest -v --exitfirst
    --junitxml=${CMAKE_BINARY_DIR}/test_pyproxies.xml
    "${CMAKE_CURRENT_SOURCE_DIR}/test_pyproxies.py")

set_tests_properties(
  tPyProxies
  PROPERTIES
    LABELS
    "pyintegration"
    ENVIRONMENT
    "IDG_INC_DIR=${CMAKE_BINARY_DIR}/include;IDG_LIB_DIR=${CMAKE_BINARY_DIR}/lib;PYTHONPATH=${CMAKE_INSTALL_PREFIX}/lib/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/site-packages:${CMAKE_SOURCE_DIR}/idg-bin/tests/python"
)
