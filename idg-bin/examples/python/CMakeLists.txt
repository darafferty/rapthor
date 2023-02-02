# Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later

install(
  FILES common.py
        cpu-reference.py
        cpu-optimized.py
        cuda-generic.py
        cuda-unified.py
        hybrid-cuda-generic-optimized.py
        coverage.py
        tiles.py
  PERMISSIONS
    OWNER_READ
    OWNER_WRITE
    OWNER_EXECUTE
    GROUP_READ
    GROUP_EXECUTE
    WORLD_READ
    WORLD_EXECUTE
  DESTINATION bin/examples/python)
