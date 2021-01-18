# Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later

# Install Python modules.
install(FILES
#        __init__.py
        h5parmwriter.py
        idgcaldpstep.py

        DESTINATION
        ${PYTHON_INSTALL_DIR})