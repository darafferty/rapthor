# Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later

install(
  FILES common.py test-cpu-optimized.py test-cuda-generic.py
        test-hybrid-cuda-generic-optimized.py test-opencl-generic.py
        test-python-reference.py
  PERMISSIONS
    OWNER_READ
    OWNER_WRITE
    OWNER_EXECUTE
    GROUP_READ
    GROUP_EXECUTE
    WORLD_READ
    WORLD_EXECUTE
  DESTINATION bin/tests/python)
