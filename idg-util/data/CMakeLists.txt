# Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later

# data files
set (${PROJECT_NAME}_data
    SKA1_low.txt
    SKA1_mid.txt
    LOFAR_lba.txt
    LOFAR_hba.txt
)

# Copy data files
install(
    FILES
    ${${PROJECT_NAME}_data}
    DESTINATION
    data
)
