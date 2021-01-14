#! /usr/bin/env python3
"""
Script to make a-term images from solutions
"""
import argparse
from argparse import RawTextHelpFormatter
import os
from rapthor.lib.screen import KLScreen, VoronoiScreen
from losoto.h5parm import h5parm


def main(h5parmfile, soltabname='phase000', screen_type='tessellated', outroot='',
         bounds_deg=None, bounds_mid_deg=None, skymodel=None,
         solsetname='sol000', padding_fraction=1.4, cellsize_deg=0.2,
         smooth_deg=0, interp_kind='nearest', ncpu=0):
    """
    Make a-term FITS images

    Parameters
    ----------
    h5parmfile : str
        Filename of h5parm
    soltabname : str, optional
        Name of soltab to use. If "gain" is in the name, phase and amplitudes are used
    screen_type : str, optional
        Kind of screen to use: 'tessellated' (simple Voronoi tessellation) or 'kl'
        (Karhunen-Lo`eve transform)
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
    interp_kind : str, optional
        Kind of interpolation to use. Can be any supported by scipy.interpolate.interp1d
    ncpu : int, optional
        Number of CPUs to use (0 means all)

    Returns
    -------
    result : dict
        Dict with list of FITS files
    """
    if 'gain' in soltabname:
        # We have scalarphase and XX+YY amplitudes
        soltab_amp = soltabname.replace('gain', 'amplitude')
        soltab_ph = soltabname.replace('gain', 'phase')
    else:
        # We have scalarphase only
        soltab_amp = None
        soltab_ph = soltabname

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
    if screen_type == 'kl':
        # No need to smooth KL screens
        smooth_pix = 0.0

    # Check whether we just have one direction. If so, force screen_type to 'tessellated'
    # as it can handle this case and KL screens can't
    H = h5parm(h5parmfile)
    solset = H.getSolset(solsetname)
    soltab = solset.getSoltab(soltab_ph)
    source_names = soltab.dir[:]
    if len(source_names) == 1:
        screen_type = 'tessellated'
    H.close()

    # Fit screens and make a-term images
    width_deg = bounds_deg[3] - bounds_deg[1]  # Use Dec difference and force square images
    rootname = os.path.basename(outroot)
    if screen_type == 'kl':
        screen = KLScreen(rootname, h5parmfile, skymodel, bounds_mid_deg[0], bounds_mid_deg[1],
                          width_deg, width_deg, solset_name=solsetname,
                          phase_soltab_name=soltab_ph, amplitude_soltab_name=soltab_amp)
    elif screen_type == 'tessellated':
        screen = VoronoiScreen(rootname, h5parmfile, skymodel, bounds_mid_deg[0], bounds_mid_deg[1],
                               width_deg, width_deg, solset_name=solsetname,
                               phase_soltab_name=soltab_ph, amplitude_soltab_name=soltab_amp)
    screen.process(ncpu=ncpu)
    outdir = os.path.dirname(outroot)
    screen.write(outdir, cellsize_deg, smooth_pix=smooth_pix, interp_kind=interp_kind, ncpu=ncpu)


if __name__ == '__main__':
    descriptiontext = "Make a-term images from solutions.\n"

    parser = argparse.ArgumentParser(description=descriptiontext, formatter_class=RawTextHelpFormatter)
    parser.add_argument('h5parmfile', help='Filename of input h5parm')
    parser.add_argument('--soltabname', help='Name of soltab', type=str, default='phase000')
    parser.add_argument('--screen_type', help='Type of screen', type=str, default='tessellated')
    parser.add_argument('--outroot', help='Root of output images', type=str, default='')
    parser.add_argument('--bounds_deg', help='Bounds list in deg', type=str, default=None)
    parser.add_argument('--bounds_mid_deg', help='Bounds mid list in deg', type=str, default=None)
    parser.add_argument('--skymodel', help='Filename of sky model', type=str, default=None)
    parser.add_argument('--solsetname', help='Solset name', type=str, default='sol000')
    parser.add_argument('--padding_fraction', help='Padding fraction', type=float, default=1.4)
    parser.add_argument('--cellsize_deg', help='Cell size in deg', type=float, default=0.2)
    parser.add_argument('--smooth_deg', help='Smooth scale in degree', type=float, default=0.0)
    parser.add_argument('--ncpu', help='Number of CPUs to use', type=int, default=0)
    args = parser.parse_args()
    main(args.h5parmfile, soltabname=args.soltabname, screen_type=args.screen_type,
         outroot=args.outroot, bounds_deg=args.bounds_deg,
         bounds_mid_deg=args.bounds_mid_deg, skymodel=args.skymodel,
         solsetname=args.solsetname, padding_fraction=args.padding_fraction,
         cellsize_deg=args.cellsize_deg, smooth_deg=args.smooth_deg, ncpu=args.ncpu)
