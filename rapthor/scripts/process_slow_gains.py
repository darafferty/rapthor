#!/usr/bin/env python3
"""
Script to process gain solutions
"""
import argparse
from argparse import RawTextHelpFormatter
from losoto.h5parm import h5parm
import numpy as np
from rapthor.lib import miscellaneous as misc
from scipy.optimize import curve_fit
from astropy.stats import sigma_clipped_stats
from scipy.ndimage import generic_filter
import sys


def get_ant_dist(ant_xyz, ref_xyz):
    """
    Returns distance between ant and ref in m

     Parameters
    ----------
    ant_xyz : array
        Array of station position
    ref_xyz : array
        Array of reference position

    Returns
    -------
    dist : float
        Distance between station and reference positions

    """
    import numpy as np

    return np.sqrt((ref_xyz[0] - ant_xyz[0])**2 + (ref_xyz[1] - ant_xyz[1])**2 + (ref_xyz[2] - ant_xyz[2])**2)


def func(x, m, c):
    return m * x + c


def normalize_direction(soltab, remove_core_gradient=True, solset=None, ref_id=0):
    """
    Normalize amplitudes so that the mean of the XX and YY median amplitudes
    is equal to unity, per direction

    Parameters
    ----------
    soltab : solution table
        Input table with solutions. Solution axes are assumed to be in the
        standard DDECal order of ['time', 'freq', 'ant', 'dir', 'pol']
    remove_core_gradient : bool, optional
        If True, remove any gradient with distance from the core stations
    solset : solution set, optional
        Input set, needed if remove_core_gradient is True
    ref_id : int, optional
        Index of reference station, needed if remove_core_gradient is True
    """
    # Make a copy of the input data to fill with normalized values
    parms = soltab.val[:]
    weights = soltab.weight[:]
    initial_flagged_indx = np.logical_or(np.isnan(parms), weights == 0.0)
    parms[initial_flagged_indx] = np.nan

    # Work in log space, as required for amplitudes
    parms = np.log10(parms)

    # Find and remove any gradient for each direction separately
    if remove_core_gradient:
        dist = []
        station_names = soltab.ant[:]
        if type(station_names) is not list:
            station_names = station_names.tolist()
        station_dict = solset.getAnt()
        station_positions = []
        for station in station_names:
            station_positions.append(station_dict[station])
        for s in range(len(station_names)):
            if s == ref_id:
                dist.append(1.0)
            else:
                dist.append(get_ant_dist(station_positions[s], station_positions[ref_id]))
        for dir in range(len(soltab.dir[:])):
            dist_vals = []
            mean_vals = []
            stat_names = []
            for s in range(len(station_names)):
                if 'CS' in station_names[s] and s != ref_id:
                    if not np.all(np.isnan(parms[:, :, s, dir, :])):
                        mean_vals.append(np.nanmean(parms[:, :, s, dir, :]))
                        dist_vals.append(dist[s])
                        stat_names.append(station_names[s])

            # Find best-fit gradient for core only
            x = np.log10(np.array(dist_vals))
            y = np.array(mean_vals)
            w = np.ones_like(y)
            popt, pcov = curve_fit(func, x, y, sigma=w, p0=[0.0, 1.0])

            # Divide out the gradient, assuming the values at the largest distances
            # are more likely correct (and so should be around 1.0, after normalization)
            for s in range(len(station_names)):
                if 'CS' in station_names[s]:
                    if s == ref_id:
                        # For the reference station, take as the distance that of a
                        # neighboring station, to avoid large extrapolations to zero
                        # distance
                        parms[:, :, s, dir, :] -= popt[0]*np.log10(dist[s+1]) + popt[1] - (popt[0]*np.log10(np.max(dist_vals)) + popt[1])
                    else:
                        parms[:, :, s, dir, :] -= popt[0]*np.log10(dist[s]) + popt[1] - (popt[0]*np.log10(np.max(dist_vals)) + popt[1])

    # Normalize each direction separately so that the mean of the XX and YY median
    # amplitudes is unity over all times, frequencies, and pols
    for dir in range(len(soltab.dir[:])):
        norm_factor = np.log10(get_median_amp(10**parms[:, :, :, dir, :], weights[:, :, :, dir, :]))
        parms[:, :, :, dir, :] -= norm_factor

    # Convert back to non-log values and make sure flagged solutions are still flagged
    parms = 10**parms
    parms[initial_flagged_indx] = np.nan
    weights[initial_flagged_indx] = 0.0
    soltab.setValues(parms)
    soltab.setValues(weights, weight=True)


def smooth_solutions(ampsoltab, phasesoltab=None):
    """
    Smooth solutions per direction

    The code is mostly taken from LoSoTo's smooth operation, which is not
    used directly as it does not support different smooth box sizes in each
    direction.

    Parameters
    ----------
    ampsoltab : solution table
        Input table with amplitude solutions. Solution axes are assumed to be in the
        standard DDECal order of ['time', 'freq', 'ant', 'dir', 'pol']
    phasesoltab : solution table, optional
        Input table with phase solutions; if specified, the phase solutions will
        also be smoothed. Solution axes are assumed to be in the standard DDECal
        order of ['time', 'freq', 'ant', 'dir', 'pol']
    """
    # Make a copy of the input data to fill with smoothed values
    amps = ampsoltab.val[:]
    if phasesoltab is not None:
        phases = phasesoltab.val[:]

    for direction in range(len(ampsoltab.dir[:])):
        # Get the smoothing box size. A value of None indicates that smoothing is
        # not needed for this direction
        size = get_smooth_box_size(ampsoltab, direction)
        if size is not None:
            # Process the amplitudes
            vals = ampsoltab.val[:, :, :, direction, :]
            vals = np.log10(vals)
            weights = ampsoltab.weight[:, :, :, direction, :]
            vals_bkp = vals[weights == 0]
            np.putmask(vals, weights == 0, np.nan)
            valsnew = generic_filter(vals, np.nanmedian, size=size, mode='constant', cval=np.nan)
            valsnew[weights == 0] = vals_bkp
            valsnew = 10**valsnew
            amps[:, :, :, direction, :] = valsnew

            # Process the phases
            if phasesoltab is not None:
                vals = phasesoltab.val[:, :, :, direction, :]
                weights = phasesoltab.weight[:, :, :, direction, :]
                vals_bkp = vals[weights == 0]
                vals = np.exp(1j*vals)
                valsreal = np.real(vals)
                valsimag = np.imag(vals)
                np.putmask(valsreal, weights == 0, np.nan)
                np.putmask(valsimag, weights == 0, np.nan)
                valsrealnew = generic_filter(valsreal, np.nanmedian, size=size, mode='constant', cval=np.nan)
                valsimagnew = generic_filter(valsimag, np.nanmedian, size=size, mode='constant', cval=np.nan)
                valsnew = valsrealnew + 1j*valsimagnew
                valsnew = np.angle(valsnew)
                valsnew[weights == 0] = vals_bkp
                phases[:, :, :, direction, :] = valsnew

    # Save the smoothed solutions
    ampsoltab.setValues(amps)
    if phasesoltab is not None:
        phasesoltab.setValues(phases)


def get_smooth_box_size(ampsoltab, direction):
    """
    Determine the smoothing box size for a given direction from the
    noise in the solutions

    Parameters
    ----------
    ampsoltab : solution table
        Input table with amplitude solutions. Solution axes are assumed to be in the
        standard DDECal order of ['time', 'freq', 'ant', 'dir', 'pol']
    direction : int
        Index of direction to consider

    Returns
    -------
    box_size : int
        Box size for smoothing
    """
    unflagged_indx = np.logical_and(~np.isnan(ampsoltab.val[:, :, :, direction, :]),
                                    ampsoltab.weight[:, :, :, direction, :] != 0.0)
    noise = sigma_clipped_stats(np.log10(ampsoltab.val[:, :, :, direction, :][unflagged_indx]))[2]
    if noise >= 0.1:
        box_size = 9
    elif noise < 0.1 and noise >= 0.08:
        box_size = 7
    elif noise < 0.08 and noise >= 0.07:
        box_size = 5
    elif noise < 0.07 and noise >= 0.04:
        box_size = 3
    else:
        box_size = None

    return box_size


def get_median_amp(amps, weights):
    """
    Returns the mean of the (unflagged) XX and YY median amplitudes

    Parameters
    ----------
    amps : array
        Array of amplitudes, with the polarization axis last
    weights : array
        Array of weights, with the polarization axis last

    Returns
    -------
    medamp : float
        The mean of the XX and YY median amplitudes
    """
    amps_xx = amps[..., 0]
    amps_yy = amps[..., -1]
    weights_xx = weights[..., 0]
    weights_yy = weights[..., -1]

    idx_xx = np.logical_and(~np.isnan(amps_xx), weights_xx != 0.0)
    idx_yy = np.logical_and(~np.isnan(amps_yy), weights_yy != 0.0)
    medamp = 0.5 * (10**(sigma_clipped_stats(np.log10(amps_xx[idx_xx]))[1]) +
                    10**(sigma_clipped_stats(np.log10(amps_yy[idx_yy]))[1]))

    return medamp


def flag_amps(soltab, lowampval=None, highampval=None, threshold_factor=0.5):
    """
    Flag high and low amplitudes per direction

    Parameters
    ----------
    soltab : solution table
        Input table with solutions. Solution axes are assumed to be in the
        standard DDECal order of ['time', 'freq', 'ant', 'dir', 'pol']
    lowampval : float, optional
        The threshold value below which amplitudes are flagged (must be >= 0.1).
        If None, the threshold is calculated per direction as as
        lowampval = median_val * threshold_factor
    highampval : float, optional
        The threshold value above which amplitudes are flagged (must be <= 10).
        If None, the threshold is calculated per direction as
        highampval = median_val / threshold_factor
    threshold_factor : float, optional
        If lowampval and/or highampval is None, this factor is used to
        determine their values. It must lie in the range (0, 1)
    """
    if threshold_factor <= 0.0 or threshold_factor >= 1.0:
        sys.exit('ERROR: threshold_factor must be in the range (0, 1)')

    # Get the current flags
    amps = soltab.val[:]
    weights = soltab.weight[:]

    for dir in range(len(soltab.dir[:])):
        amps_dir = amps[:, :, :, dir, :]
        weights_dir = weights[:, :, :, dir, :]
        medamp = get_median_amp(amps_dir, weights_dir)
        if lowampval is None:
            low = medamp * threshold_factor
        else:
            low = lowampval
        if highampval is None:
            high = medamp / threshold_factor
        else:
            high = highampval
        if low < 0.1:
            low = 0.1
        if high > 10.0:
            high = 10.0
        if low >= high:
            high = low * 2.0

        # Flag, setting flagged values to NaN and weights to 0
        initial_flagged_indx = np.logical_or(np.isnan(amps_dir), weights_dir == 0.0)
        amps_dir[initial_flagged_indx] = medamp
        new_flag_indx = np.logical_or(amps_dir < low, amps_dir > high)
        amps_dir[initial_flagged_indx] = np.nan
        amps_dir[new_flag_indx] = np.nan
        weights_dir[initial_flagged_indx] = 0.0
        weights_dir[new_flag_indx] = 0.0
        amps[:, :, :, dir, :] = amps_dir
        weights[:, :, :, dir, :] = weights_dir

    # Save the new flags
    soltab.setValues(amps)
    soltab.setValues(weights, weight=True)


def transfer_flags(soltab1, soltab2):
    """
    Transfers the flags from soltab1 to soltab2

    Note: both solution tables must have the same shape. Existing flagged data in
    soltab2 are not affected

    Parameters
    ----------
    soltab1 : solution table
        Table from which flags are transferred
    soltab2 : solution table
        Table 2 to which flags are transferred
    """
    flagged_indx = np.logical_or(np.isnan(soltab1.val), soltab1.weight == 0.0)
    vals2 = soltab2.val[:]
    weights2 = soltab2.weight[:]
    vals2[flagged_indx] = np.nan
    weights2[flagged_indx] = 0.0

    # Save the new flags
    soltab2.setValues(vals2)
    soltab2.setValues(weights2, weight=True)


def main(h5parmfile, solsetname='sol000', ampsoltabname='amplitude000',
         phasesoltabname='phase000', ref_id=None, smooth=False, normalize=False,
         flag=False, lowampval=None, highampval=None):
    """
    Process gain solutions

    Parameters
    ----------
    h5parmfile : str
        Filename of h5parm
    solsetname : str, optional
        Name of solset
    ampsoltabname : str, optional
        Name of amplitude soltab
    phasesoltabname : str, optional
        Name of phase soltab
    ref_id : int, optional
        Index of reference station. If None, a reference station is chosen
        automatically
    smooth : bool, optional
        Smooth amp solutions
    normalize : bool, optional
        Normalize amp solutions
    flag : bool, optional
        Flag amp solutions
    lowampval : float, optional
        The threshold value below which amplitudes are flagged. If None, the
        threshold is set to 0.5 times the median
    highampval : float, optional
        The threshold value above which amplitudes are flagged. If None, the
        threshold is set to 2 times the median
    """
    # Read in solutions
    H = h5parm(h5parmfile, readonly=False)
    solset = H.getSolset(solsetname)
    ampsoltab = solset.getSoltab(ampsoltabname)
    phasesoltab = solset.getSoltab(phasesoltabname)

    # Process the solutions
    if flag:
        flag_amps(ampsoltab, lowampval=lowampval, highampval=highampval)
        transfer_flags(ampsoltab, phasesoltab)
    if smooth:
        smooth_solutions(ampsoltab, phasesoltab=phasesoltab)
    if normalize:
        if ref_id is None:
            ref_id = misc.get_reference_station(phasesoltab, 10)
        normalize_direction(ampsoltab, remove_core_gradient=True, solset=solset, ref_id=ref_id)
    H.close()


if __name__ == '__main__':
    descriptiontext = "Process gain solutions.\n"

    parser = argparse.ArgumentParser(description=descriptiontext, formatter_class=RawTextHelpFormatter)
    parser.add_argument('h5parmfile', help='Filename of input h5parm')
    parser.add_argument('--solsetname', help='Solset name', type=str, default='sol000')
    parser.add_argument('--ampsoltabname', help='Amplitude soltab name', type=str, default='amplitude000')
    parser.add_argument('--phasesoltabname', help='Phase soltab name', type=str, default='phase000')
    parser.add_argument('--ref_id', help='Reference station', type=int, default=0)
    parser.add_argument('--normalize', help='Normalize amplitude solutions', type=bool, default=False)
    parser.add_argument('--smooth', help='Smooth amplitude solutions', type=bool, default=False)
    parser.add_argument('--flag', help='Flag amplitude solutions', type=bool, default=False)
    parser.add_argument('--lowampval', help='Low threshold for amplitude flagging', type=float, default=None)
    parser.add_argument('--highampval', help='High threshold for amplitude flagging', type=float, default=None)
    args = parser.parse_args()
    main(args.h5parmfile, solsetname=args.solsetname, ampsoltabname=args.ampsoltabname,
         phasesoltabname=args.phasesoltabname, ref_id=args.ref_id, smooth=args.smooth,
         normalize=args.normalize, flag=args.flag, lowampval=args.lowampval,
         highampval=args.highampval)
