#! /usr/bin/env python3
"""
Script to make a-term images from solutions
"""
import argparse
from argparse import RawTextHelpFormatter
from losoto.h5parm import h5parm
import lsmtool
import os
import numpy as np
from rapthor.lib import miscellaneous as misc
from astropy.io import fits as pyfits
from astropy import wcs
from shapely.geometry import Point
from scipy.spatial import Voronoi
import shapely.geometry
import shapely.ops
import scipy.ndimage as ndimage
import scipy.interpolate as si
from losoto.operations import reweight, stationscreen


def interpolate_amps(soltab_amp, soltab_ph, interp_kind='nearest'):
    """
    Interpolate slow amplitudes to time and frequency grid of fast phases

    Note: interpolation is done in log space.

    Parameters
    ----------
    soltab_amp : soltab
        Soltab with slow amplitudes
    soltab_ph : soltab
        Soltab with fast phases
    interp_kind : str, optional
        Kind of interpolation to use. Can be any supported by scipy.interpolate.interp1d

    Returns
    -------
    vals : array
        Array of interpolated amplitudes

    """
    vals = soltab_amp.val
    times_slow = soltab_amp.time
    freqs_slow = soltab_amp.freq
    times_fast = soltab_ph.time
    freqs_fast = soltab_ph.freq

    # Interpolate the slow amps to the fast times and frequencies
    axis_names = soltab_amp.getAxesNames()
    time_ind = axis_names.index('time')
    freq_ind = axis_names.index('freq')
    fast_axis_names = soltab_ph.getAxesNames()
    fast_time_ind = fast_axis_names.index('time')
    fast_freq_ind = fast_axis_names.index('freq')
    if len(times_slow) == 1:
        # If just a single time, we just repeat the values as needed
        new_shape = list(vals.shape)
        new_shape[time_ind] = vals_ph.shape[fast_time_ind]
        new_shape[freq_ind] = vals_ph.shape[fast_freq_ind]
        vals = np.resize(vals, new_shape)
    else:
        # Interpolate (in log space)
        logvals = np.log10(vals)
        if vals.shape[time_ind] != vals_ph.shape[fast_time_ind]:
            f = si.interp1d(times_slow, logvals, axis=time_ind, kind=interp_kind, fill_value='extrapolate')
            logvals = f(times_fast)
        if vals.shape[freq_ind] != vals_ph.shape[fast_freq_ind]:
            f = si.interp1d(freqs_slow, logvals, axis=freq_ind, kind=interp_kind, fill_value='extrapolate')
            logvals = f(freqs_fast)
        vals = 10**(logvals)

    return vals


def main(h5parmfile, soltabname='phase000', screen_type='voronoi', outroot='',
         bounds_deg=None, bounds_mid_deg=None, skymodel=None,
         solsetname='sol000', padding_fraction=1.4, cellsize_deg=0.1,
         smooth_deg=0, time_avg_factor=1, interp_kind='nearest'):
    """
    Make a-term FITS images

    Parameters
    ----------
    h5parmfile : str
        Filename of h5parm
    soltabname : str, optional
        Name of soltab to use. If "gain" is in the name, phase and amplitudes are used
    screen_type : str, optional
        Kind of screen to use: 'voronoi' (simple Voronoi tessellation) or 'kl' (Karhunen-
        Lo`eve transform)
    outroot : str, optional
        Root of filename of output FITS file (root+'_0.fits')
    bounds_deg : list, optional
        List of [maxRA, minDec, minRA, maxDec] for image bounds
    bounds_mid_deg : list, optional
        List of [RA, Dec] for midpoint of image bounds
    skymodel : str, optional
        Filename of calibration sky model (needed for patch positions)
    solsetname : str, optional
        Name of solset
    padding_fraction : float, optional
        Fraction of total size to pad with (e.g., 0.2 => 20% padding all around)
    cellsize_deg : float, optional
        Cellsize of output image
    smooth_deg : float, optional
        Size of smoothing kernel in degrees to apply
    time_avg_factor : int, optional
        Averaging factor in time for fast-phase corrections
    interp_kind : str, optional
        Kind of interpolation to use. Can be any supported by scipy.interpolate.interp1d

    Returns
    -------
    result : dict
        Dict with list of FITS files
    """
    # Read in solutions
    H = h5parm(h5parmfile)
    solset = H.getSolset(solsetname)
    if 'gain' in soltabname:
        # We have scalarphase and XX+YY amplitudes
        soltab_amp = solset.getSoltab(soltabname.replace('gain', 'amplitude'))
        soltab_ph = solset.getSoltab(soltabname.replace('gain', 'phase'))
    else:
        # We have scalarphase only
        soltab_amp = None
        soltab_ph = solset.getSoltab(soltabname)

    if type(bounds_deg) is str:
        bounds_deg = [float(f.strip()) for f in bounds_deg.strip('[]').split(';')]
    if type(bounds_mid_deg) is str:
        bounds_mid_deg = [float(f.strip()) for f in bounds_mid_deg.strip('[]').split(';')]
    if padding_fraction is not None:
        padding_fraction = float(padding_fraction)
        padding_ra = (bounds_deg[2] - bounds_deg[0]) * (padding_fraction - 1.0)
        padding_dec = (bounds_deg[3] - bounds_deg[1]) * (padding_fraction - 1.0)
        bounds_deg[0] -= padding_ra
        bounds_deg[1] -= padding_dec
        bounds_deg[2] += padding_ra
        bounds_deg[3] += padding_dec
    cellsize_deg = float(cellsize_deg)
    smooth_deg = float(smooth_deg)
    smooth_pix = smooth_deg / cellsize_deg
    time_avg_factor = int(time_avg_factor)

    # Check whether we just have one direction. If so, force screen_type to 'voronoi'
    source_names = soltab_ph.dir[:]
    if len(source_names) == 1:
        screen_type = 'voronoi'

    if screen_type == 'kl':
        # Do Karhunen-Lo`eve transform
        # Reweight the solutions by the scatter after detrending
        reweight.run(soltab_ph, mode='window', nmedian=3, nstddev=251)
        if soltab_amp is not None:
            reweight.run(soltab_amp, mode='window', nmedian=3, nstddev=21)

        # Now call LoSoTo's stationscreen operation to do the fitting
        stationscreen.run(soltab_ph, 'phase_screen000')
        soltab_ph_screen = solset.getSoltab('phase_screen000')
        soltab_ph_screen_resid = solset.getSoltab('phase_screen000resid')
        if soltab_amp is not None:
            stationscreen.run(soltab_amp, 'amplitude_screen000')
            soltab_amp_screen = solset.getSoltab('amplitude_screen000')
            soltab_amp_screen_resid = solset.getSoltab('amplitude_screen000resid')
        else:
            soltab_amp_screen = None
            soltab_amp_screen_resid = None

        # Transform the screens into FITS images
        make_kl_screen_images(soltab_ph_screen, soltab_ph_screen_resid,
                              soltab_amp=soltab_amp_screen,
                              resSoltab_amp=soltab_amp_screen_resid,
                              prefix='', ncpu=0)

    elif screen_type == 'voronoi':
        # Do Voronoi tessellation + smoothing
        make_voronoi_screen_images(soltab_ph, soltab_amp, bounds_mid_deg)



if __name__ == '__main__':
    descriptiontext = "Make a-term images from solutions.\n"

    parser = argparse.ArgumentParser(description=descriptiontext, formatter_class=RawTextHelpFormatter)
    parser.add_argument('h5parmfile', help='Filename of input h5parm')
    parser.add_argument('--soltabname', help='Name of soltab', type=str, default='phase000')
    parser.add_argument('--screen_type', help='Type of screen', type=str, default='voronoi')
    parser.add_argument('--outroot', help='Root of output images', type=str, default='')
    parser.add_argument('--bounds_deg', help='Bounds list in deg', type=str, default=None)
    parser.add_argument('--bounds_mid_deg', help='Bounds mid list in deg', type=str, default=None)
    parser.add_argument('--skymodel', help='Filename of sky model', type=str, default=None)
    parser.add_argument('--solsetname', help='Solset name', type=str, default='sol000')
    parser.add_argument('--padding_fraction', help='Padding fraction', type=float, default=1.4)
    parser.add_argument('--cellsize_deg', help='Cell size in deg', type=float, default=0.1)
    parser.add_argument('--smooth_deg', help='Smooth scale in degree', type=float, default=0.0)
    parser.add_argument('--time_avg_factor', help='Averaging factor in time', type=int, default=1)
    args = parser.parse_args()
    main(args.h5parmfile, soltabname=args.soltabname, screen_type=args.screen_type,
         outroot=args.outroot, bounds_deg=args.bounds_deg,
         bounds_mid_deg=args.bounds_mid_deg, skymodel=args.skymodel,
         solsetname=args.solsetname, padding_fraction=args.padding_fraction,
         cellsize_deg=args.cellsize_deg, smooth_deg=args.smooth_deg,
         time_avg_factor=args.time_avg_factor)
