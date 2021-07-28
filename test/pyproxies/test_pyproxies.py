# Copyright (C) 2021 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later

import pytest
import numpy as np

# Make sure idg and common are on your PYTHONPATH
# common is located in ${CMAKE_SOURCE_DIR}/idg-bin/tests/python
import idg
from common import main


@pytest.mark.parametrize(
    "proxy",
    [
        "idg.CPU.Optimized",
        "idg.HybridCUDA.GenericOptimized",
        "idg.CUDA.Generic",
        # OpenCL test is skipped
        "idg.OpenCL.Generic",
    ],
)
def test_proxies(proxy):
    try:
        proxy = eval(proxy)
    except:
        # Skip if proxy cannot be found
        pytest.skip()

    ref_grid, ref_visibilities, proxy_grid, proxy_visibilities = main(proxy, plot=False)

    np.testing.assert_allclose(proxy_grid, ref_grid, rtol=1e-2, atol=2e-2)
    np.testing.assert_allclose(proxy_visibilities, ref_visibilities, rtol=1e-3)
