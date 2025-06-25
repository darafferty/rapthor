"""
Tests for the process_gains.py script.
"""

import numpy as np
from rapthor.scripts.process_gains import (flag_amps, get_angular_distance,
                                           get_ant_dist, get_median_amp,
                                           get_smooth_box_size, main,
                                           normalize_direction,
                                           smooth_solutions, transfer_flags)


def test_get_ant_dist():
    ant_xyz = np.array([1, 2, 3])
    ref_xyz = np.array([4, 5, 6])
    dist = get_ant_dist(ant_xyz, ref_xyz)
    assert np.isclose(dist, 3 * np.sqrt(3)), f"Expected distance 3*sqrt(3), got {dist}"


def test_get_angular_distance():
    ra_dec1 = (0, 0)
    ra_dec2 = (45, 45)
    dist = get_angular_distance(ra_dec1, ra_dec2)
    assert np.isclose(dist, 60), f"Expected angular distance 60 degrees, got {dist}"


def test_normalize_direction(soltab):
    # max_station_delta = 0.0
    # scale_delta_with_dist = False
    # phase_center = None
    # normalize_direction(soltab, max_station_delta, scale_delta_with_dist, phase_center)
    # result = soltab.getValues()
    # # Check result
    pass


def test_smooth_solutions():
    # ampsoltab = None  # Replace with actual amplitude solution table
    # phasesoltab = None  # Replace with actual phase solution table if available
    # ref_id = 0  # Replace with actual reference ID if needed
    # result = smooth_solutions(ampsoltab, phasesoltab=None, ref_id=0)
    # # Check result
    pass


def test_get_smooth_box_size():
    # ampsoltab = None  # Replace with actual amplitude solution table
    # direction = None  # Replace with actual direction if needed
    # result = get_smooth_box_size(ampsoltab, direction)
    # # Check result
    pass


def test_get_median_amp():
    # amps = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # weights = np.array([1, 1, 1, 1, 1])
    # median_amp = 3.0  # Expected median value
    # result = get_median_amp(amps, weights)
    # # Check result
    pass


def test_flag_amps(soltab):
    # lowampval = None  # Replace with actual low amplitude value if needed
    # highampval = None  # Replace with actual high amplitude value if needed
    # threshold_factor = 0.2  # Example threshold factor, adjust as needed
    # result = flag_amps(soltab, lowampval, highampval, threshold_factor)
    # # Check result
    pass


def test_transfer_flags():
    # soltab1 = None  # Replace with actual source solution table
    # soltab2 = None  # Replace with actual target solution table
    # transfer_flags(soltab1, soltab2)
    # result = soltab2.getValues()
    # # Check result
    pass


def test_main():
    # h5parmfile = "test.h5"  # Replace with actual H5parm file path
    # solsetname = "sol000"  # Replace with actual solution set name if needed
    # ampsoltabname = "amplitude000"  # Replace with actual amplitude solution table name if needed
    # phasesoltabname = "phase000"  # Replace with actual phase solution table name if needed
    # ref_id = None  # Replace with actual reference ID if needed
    # smooth = False  # Set to True if smoothing is required
    # normalize = False  # Set to True if normalization is required
    # flag = False  # Set to True if flagging is required
    # lowampval = None  # Replace with actual low amplitude value if needed
    # highampval = None  # Replace with actual high amplitude value if needed
    # max_station_delta = 0.0  # Set to a value if station delta is required
    # scale_delta_with_dist = False  # Set to True if scaling delta with distance is required
    # phase_center = None  # Replace with actual phase center if needed
    # main(
    #     h5parmfile,
    #     solsetname,
    #     ampsoltabname,
    #     phasesoltabname,
    #     ref_id,
    #     smooth,
    #     normalize,
    #     flag,
    #     lowampval,
    #     highampval,
    #     max_station_delta,
    #     scale_delta_with_dist,
    #     phase_center,
    # )
    # # Check contents of the H5parm file or solution set after running main
    pass
