# Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later

# Install Python modules.
install(FILES #        __init__.py
              h5parmwriter.py idgcaldpstep.py basisfunctions.py idgcalutils.py
        DESTINATION ${PYTHON_INSTALL_DIR})

add_test(
  NAME idgcal
  COMMAND python3 -m pytest -v
          --junitxml=${CMAKE_CURRENT_BINARY_DIR}/unittest_idgcal.xml .
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/unit_tests)
set_tests_properties(idgcal PROPERTIES LABELS "unit")
