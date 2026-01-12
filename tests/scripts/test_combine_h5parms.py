"""
Test cases for the combine_h5parms script in the rapthor package.
"""

import numpy as np
import pytest
from rapthor.scripts.combine_h5parms import (
    average_polarizations, combine_phase1_amp1_amp2, combine_phase1_amp2,
    combine_phase1_phase2_amp2, combine_phase1_phase2_amp2_diagonal,
    combine_phase1_phase2_amp2_scalar, copy_solset, expand_array,
    interpolate_solutions, main)


def test_expand_array():
    array = np.array([[1, 2], [3, 4]])
    new_shape = (2, 2, 2)
    new_axis_ind = 1
    expected_shape = (2, 2, 2)
    expanded_array = expand_array(array, new_shape, new_axis_ind)
    assert expanded_array.shape == expected_shape, (
        f"Expected shape {expected_shape}, got {expanded_array.shape}"
    )
    # Check if the values are correctly expanded
    expected_values = np.array([[[1, 2], [1, 2]], [[3, 4], [3, 4]]])
    assert np.array_equal(expanded_array, expected_values), (
        f"Expected values {expected_values}, got {expanded_array}"
    )


def test_average_polarizations(soltab):
    # average_polarizations(soltab)
    pass


def test_interpolate_solutions():
    # # Create dummy fast and slow solution tables
    # fast_soltab = "fast_soltab"
    # slow_soltab = "slow_soltab"
    # final_axes_shapes = {
    #     'dir': 2,
    #     'freq': 2,
    # }
    # interpolate_solutions(fast_soltab, slow_soltab, final_axes_shapes,
    #                       slow_vals=None, slow_weights=None)
    pass


def test_combine_phase1_amp2():
    # # Create dummy solset objects
    # ss1 = "solset1"
    # ss2 = "solset2"
    # sso = "solset_output"
    # combine_phase1_amp2(ss1, ss2, sso)
    pass


def test_combine_phase1_amp1_amp2():
    # # Create dummy solset objects
    # ss1 = "solset1"
    # ss2 = "solset2"
    # sso = "solset_output"
    # combine_phase1_amp1_amp2(ss1, ss2, sso)
    pass


def test_combine_phase1_phase2_amp2():
    # # Create dummy solset objects
    # ss1 = "solset1"
    # ss2 = "solset2"
    # sso = "solset_output"
    # combine_phase1_phase2_amp2(ss1, ss2, sso)
    pass


def test_combine_phase1_phase2_amp2_diagonal():
    # # Create dummy solset objects
    # ss1 = "solset1"
    # ss2 = "solset2"
    # sso = "solset_output"
    # combine_phase1_phase2_amp2_diagonal(ss1, ss2, sso)
    pass


def test_combine_phase1_phase2_amp2_scalar():
    # # Create dummy solset objects
    # ss1 = "solset1"
    # ss2 = "solset2"
    # sso = "solset_output"
    # combine_phase1_phase2_amp2_scalar(ss1, ss2, sso)
    pass


def test_copy_solset():
    # # Create dummy solset objects
    # ss1 = "solset1"
    # ss2 = "solset2"
    # copy_solset(ss1, ss2)
    pass


def test_main():
    """Test the main function of the combine_h5parms script."""
    # import losoto.h5parm
    # # Create dummy h5parm files
    # h5parm1 = losoto.h5parm.h5parm("test1.h5", readonly=False)
    # h5parm2 = losoto.h5parm.h5parm("test2.h5", readonly=False)
    # outh5parm = losoto.h5parm.h5parm("output.h5", readonly=False)
    # mode = 'phase1_amp2'  # Example mode, can be changed to test other modes
    # main(h5parm1, h5parm2, outh5parm, mode, solset1='sol000', solset2='sol000',
    #      reweight=False, cal_names=None, cal_fluxes=None)
    pass
