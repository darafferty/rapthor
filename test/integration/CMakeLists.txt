# Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later

# NOTE: integration tests run only after installing IDG

# List of integration tests
set(INTEGRATION_TESTS tSinglePointSource tGridDegrid tIDGCalDPStep tCalibrate)

# Add integration tests to "integration", can be invoked with
# "ctest -L integration"
foreach(TEST ${INTEGRATION_TESTS})
  add_test(${TEST} ${CMAKE_CURRENT_SOURCE_DIR}/${TEST}.sh)
  set_tests_properties(
    ${TEST}
    PROPERTIES
      LABELS
      "integration"
      ENVIRONMENT
      "IDG_INC_DIR=${CMAKE_BINARY_DIR}/include;IDG_LIB_DIR=${CMAKE_BINARY_DIR}/lib;IDG_PYTHONPATH=${CMAKE_INSTALL_PREFIX}/${PYTHON_INSTALL_DIR}/.."
      RUN_SERIAL
      TRUE # Parallel runs fail, for unknown reasons. :-(
  )
endforeach()
