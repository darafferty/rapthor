#!/usr/bin/env python3
"""
Script to process gain solutions
"""
from argparse import ArgumentParser, RawTextHelpFormatter
from losoto.h5parm import h5parm
import numpy as np
from rapthor.lib import miscellaneous as misc
from astropy.stats import sigma_clipped_stats
from astropy.coordinates import SkyCoord
import astropy.units as u
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


def get_angular_distance(ra_dec1, ra_dec2):
    """
    Return the distance in degrees between the given coordinates

    Parameters
    ----------
    ra_dec1 : tuple of floats
        The coordinates of direction 1 as (RA, Dec) in degrees
    ra_dec2 : tuple of floats
        The coordinates of direction 2 as (RA, Dec) in degrees

    Returns
    -------
    dist : float
        Distance in degrees
    """
    coord1 = SkyCoord(ra_dec1[0], ra_dec1[1], unit=(u.degree, u.degree), frame='fk5')
    coord2 = SkyCoord(ra_dec2[0], ra_dec2[1], unit=(u.degree, u.degree), frame='fk5')

    return coord1.separation(coord2).value


def normalize_direction(soltab, max_station_delta=0.0, scale_delta_with_dist=False,
                        phase_center=None):
    """
    Normalize amplitudes so that the mean of the XX and YY median amplitudes
    for each station is equal to unity, per direction

    Parameters
    ----------
    soltab : solution table
        Input table with amplitude solutions. Solution axes are assumed to be in the
        standard DDECal order of ['time', 'freq', 'ant', 'dir', 'pol']
    max_station_delta : float, optional
        The maximum allowed difference from unity of the median of the amplitudes, per
        station (must be >= 0)
    scale_delta_with_dist : bool, optional
        If True, max_station_delta is scaled (linearly) with the distance from the
        patch direction to the phase center, allowing larger deltas for more
        distant sources (if there is only a single direction, scaling is disabled
        and the value of this parameter is ignored)
    phase_center : tuple of floats, optional
        The phase center of the observation as (RA, Dec) in degrees. Required when
        scale_delta_with_dist = True
    """
    if max_station_delta < 0.0:
        max_station_delta = 0.0
    if len(soltab.dir[:]) == 1:
        # If there is only a single direction, disable the scaling with distance
        scale_delta_with_dist = False

    # Make a copy of the input data to fill with normalized values
    parms = soltab.val[:]
    weights = soltab.weight[:]

    # Find the distance to each direction from the phase center
    if scale_delta_with_dist:
        if phase_center is None:
            raise ValueError("The phase_center must be specified if scale_delta_with_dist = True")

        source_dict = soltab.getSolset().getSou()
        dist = []
        for dir in soltab.dir[:]:
            ra_dec = (source_dict[dir][0]*180/np.pi, source_dict[dir][1]*180/np.pi)  # degrees
            dist.append(get_angular_distance(ra_dec, phase_center))
        max_dist = max(dist)
        if max_dist == 0:
            # This state should never be reached, since it should only occur when
            # there is a single direction (and that direction is centered on the
            # phase center), but we check for it anyway to ensure there is no
            # divide-by-zero later
            scale_delta_with_dist = False

    # Normalize each direction separately so that the mean of the XX and YY median
    # amplitudes is unity (within max_station_delta) for each station over all times,
    # frequencies, and pols
    for dir in range(len(soltab.dir[:])):
        if scale_delta_with_dist:
            max_delta = max_station_delta * dist[dir] / max_dist
        else:
            max_delta = max_station_delta

        # First, renormalize the direction so that core stations have a median
        # amplitude of unity
        if any(core_stations := ['CS' in stat for stat in soltab.ant[:]]):
            # no stations are marked with 'CS' => skip 
            median_dir = get_median_amp(parms[:, :, core_stations, dir, :],
                                        weights[:, :, core_stations, dir, :])
            parms[:, :, :, dir, :] /= median_dir

        # Now renormalize station-by-station, allowing some delta from unity
        for s in range(len(soltab.ant[:])):
            median_station = get_median_amp(parms[:, :, s, dir, :], weights[:, :, s, dir, :])
            norm_delta = min(max_delta, abs(1 - median_station))
            if median_station < 1.0:
                parms[:, :, s, dir, :] /= median_station * (1 + norm_delta)
            else:
                parms[:, :, s, dir, :] /= median_station / (1 + norm_delta)
        if max_delta > 0:
            # Do one final normalization to make sure the overall median for this direction
            # is unity
            median_dir = get_median_amp(parms[:, :, :, dir, :], weights[:, :, :, dir, :])
            parms[:, :, :, dir, :] /= median_dir

    # Save the normalized values
    soltab.setValues(parms)


def smooth_solutions(ampsoltab, phasesoltab=None, ref_id=0):
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
    ref_id : int, optional
        Index of reference station for phases
    """
    # Make a copy of the input data to fill with smoothed values
    amps = ampsoltab.val[:]
    nstat = amps.shape[2]
    npol = amps.shape[4]
    if phasesoltab is not None:
        phases = phasesoltab.val[:]
        for stat in range(nstat):
            phases[:, :, stat, :, :] = phases[:, :, stat, :, :] - phases[:, :, ref_id, :, :]

    for direction in range(len(ampsoltab.dir[:])):
        # Get the smoothing box size. A value of None indicates that smoothing is
        # not needed for this direction
        size = get_smooth_box_size(ampsoltab, direction)
        if size is not None:
            # Smooth solutions for each direction, station, and polarization separately
            for stat in range(nstat):
                for pol in range(npol):
                    vals = amps[:, :, stat, direction, pol]
                    vals = np.log10(vals)
                    weights = ampsoltab.weight[:, :, stat, direction, pol]
                    vals_bkp = vals[weights == 0]
                    np.putmask(vals, weights == 0, np.nan)
                    valsnew = generic_filter(vals, np.nanmedian, size=size, mode='constant', cval=np.nan)
                    valsnew[weights == 0] = vals_bkp
                    valsnew = 10**valsnew
                    amps[:, :, stat, direction, pol] = valsnew

                    # Process the phases
                    if phasesoltab is not None:
                        vals = phases[:, :, stat, direction, pol]
                        weights = phasesoltab.weight[:, :, stat, direction, pol]
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
                        phases[:, :, stat, direction, pol] = valsnew

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
    unflagged_indx = np.logical_and(np.isfinite(ampsoltab.val[:, :, :, direction, :]),
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

    idx_xx = np.logical_and(np.isfinite(amps_xx), weights_xx != 0.0)
    idx_yy = np.logical_and(np.isfinite(amps_yy), weights_yy != 0.0)
    medamp = 0.5 * (10**(sigma_clipped_stats(np.log10(amps_xx[idx_xx]))[1]) +
                    10**(sigma_clipped_stats(np.log10(amps_yy[idx_yy]))[1]))

    return medamp


def flag_amps(soltab, lowampval=None, highampval=None, threshold_factor=0.2):
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
        for s in range(len(soltab.ant[:])):
            amps_dir = amps[:, :, s, dir, :]
            weights_dir = weights[:, :, s, dir, :]
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
            initial_flagged_indx = np.logical_or(~np.isfinite(amps_dir), weights_dir == 0.0)
            amps_dir[initial_flagged_indx] = medamp
            new_flag_indx = np.logical_or(amps_dir < low, amps_dir > high)
            amps_dir[initial_flagged_indx] = np.nan
            amps_dir[new_flag_indx] = np.nan
            weights_dir[initial_flagged_indx] = 0.0
            weights_dir[new_flag_indx] = 0.0
            amps[:, :, s, dir, :] = amps_dir
            weights[:, :, s, dir, :] = weights_dir

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
    flagged_indx = np.logical_or(~np.isfinite(soltab1.val), soltab1.weight == 0.0)
    vals2 = soltab2.val[:]
    weights2 = soltab2.weight[:]
    vals2[flagged_indx] = np.nan
    weights2[flagged_indx] = 0.0

    # Save the new flags
    soltab2.setValues(vals2)
    soltab2.setValues(weights2, weight=True)


def main(h5parmfile, solsetname='sol000', ampsoltabname='amplitude000',
         phasesoltabname='phase000', ref_id=None, smooth=False, normalize=False,
         flag=False, lowampval=None, highampval=None, max_station_delta=0.0,
         scale_delta_with_dist=False, phase_center=None):
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
        Index of reference station for the phases. If None, a reference station
        is chosen automatically
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
    max_station_delta : float, optional
        The maximum allowed fractional difference between core and remote station
        normalizations.
    scale_delta_with_dist : bool, optional
        If True, max_station_delta is scaled (linearly) with the distance from the
        patch direction to the phase center, allowing larger deltas for more
        distant sources
    phase_center : tuple of floats, optional
        The phase center of the observation as (RA, Dec) in degrees. Required when
        scale_delta_with_dist = True
    """
    smooth = misc.string2bool(smooth)
    normalize = misc.string2bool(normalize)
    flag = misc.string2bool(flag)
    scale_delta_with_dist = misc.string2bool(scale_delta_with_dist)

    # Read in solutions
    H = h5parm(h5parmfile, readonly=False)
    solset = H.getSolset(solsetname)
    ampsoltab = solset.getSoltab(ampsoltabname)
    phasesoltab = solset.getSoltab(phasesoltabname)
    if ref_id is None:
        ref_id = misc.get_reference_station(phasesoltab, 10)

    # Process the solutions
    if flag:
        flag_amps(ampsoltab, lowampval=lowampval, highampval=highampval)
        transfer_flags(ampsoltab, phasesoltab)
    if smooth:
        smooth_solutions(ampsoltab, phasesoltab=phasesoltab, ref_id=ref_id)
    if normalize:
        normalize_direction(ampsoltab, max_station_delta=max_station_delta,
                            scale_delta_with_dist=scale_delta_with_dist,
                            phase_center=phase_center)
    H.close()


if __name__ == '__main__':
    descriptiontext = "Process gain solutions.\n"

    parser = ArgumentParser(description=descriptiontext, formatter_class=RawTextHelpFormatter)
    parser.add_argument('h5parmfile', help='Filename of input h5parm')
    parser.add_argument('--solsetname', help='Solset name', type=str, default='sol000')
    parser.add_argument('--ampsoltabname', help='Amplitude soltab name', type=str, default='amplitude000')
    parser.add_argument('--phasesoltabname', help='Phase soltab name', type=str, default='phase000')
    parser.add_argument('--ref_id', help='Reference station', type=int, default=0)
    parser.add_argument('--normalize', help='Normalize amplitude solutions', type=str, default='False')
    parser.add_argument('--smooth', help='Smooth amplitude solutions', type=str, default='False')
    parser.add_argument('--flag', help='Flag amplitude solutions', type=str, default='False')
    parser.add_argument('--lowampval', help='Low threshold for amplitude flagging', type=float, default=None)
    parser.add_argument('--highampval', help='High threshold for amplitude flagging', type=float, default=None)
    parser.add_argument('--max_station_delta', help='Max difference of median from unity allowed '
                        'for station normalizations', type=float, default=0.0)
    parser.add_argument('--scale_delta_with_dist', help='Scale max difference with distance', type=str, default='False')
    parser.add_argument('--phase_center_ra', help='RA of phase center in degrees', type=float, default=0.0)
    parser.add_argument('--phase_center_dec', help='Dec of phase center in degrees', type=float, default=0.0)
    args = parser.parse_args()
    phase_center = (args.phase_center_ra, args.phase_center_dec)
    main(args.h5parmfile, solsetname=args.solsetname, ampsoltabname=args.ampsoltabname,
         phasesoltabname=args.phasesoltabname, ref_id=args.ref_id, smooth=args.smooth,
         normalize=args.normalize, flag=args.flag, lowampval=args.lowampval,
         highampval=args.highampval, max_station_delta=args.max_station_delta,
         scale_delta_with_dist=args.scale_delta_with_dist, phase_center=phase_center)
