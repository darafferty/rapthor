#!/usr/bin/env python3
"""
Script to make a source catalog from an image cube
"""
from argparse import ArgumentParser, RawTextHelpFormatter
import ast
import bdsf
import os
import tempfile


def main(cube_image, cube_beams, cube_frequencies, output_catalog, threshisl=3.0,
         threshpix=5.0, rmsbox=(150, 50), rmsbox_bright=(35, 7),
         adaptive_thresh=75.0, ncores=8):
    """
    Make a source catalog from an image cube

    Parameters
    ----------
    cube_image : str
        Filename of input FITS cube image to use to detect sources
    cube_beams : str
        Filename of input text file with cube beam parameters. The file should
        give the beams as written as "(major axis, minor axis, position angle)"
        in degrees, one per cube channel. The beams for all channels should be
        given on a single line, separated by commas. E.g.:
            (0.0091, 0.0073, 38.1526), (0.0090, 0.0074, 39.1030), ...
    cube_frequencies : str
        Filename of input text file with cube frequency parameters. The file
        should give the frequencies in Hz, one per cube channel. The frequencies
        for all channels should be given on a single line, separated by commas.
        E,g.:
            23143005.3710, 129002380.3710, ...
    output_catalog : str
        Filename of output FITS source catalog
    threshisl : float, optional
        Value of thresh_isl PyBDSF parameter
    threshpix : float, optional
        Value of thresh_pix PyBDSF parameter
    rmsbox : tuple of floats, optional
        Value of rms_box PyBDSF parameter
    rmsbox_bright : tuple of floats, optional
        Value of rms_box_bright PyBDSF parameter
    adaptive_thresh : float, optional
        This value sets the threshold above which a source will use the small
        rms box
    ncores : int, optional
        Maximum number of cores to use
    """
    if rmsbox is not None and isinstance(rmsbox, str):
        rmsbox = ast.literal_eval(rmsbox)
    if isinstance(rmsbox_bright, str):
        rmsbox_bright = ast.literal_eval(rmsbox_bright)

    # Try to set the TMPDIR env var to a short path (/tmp, /var/tmp, or
    # /usr/tmp), to try to avoid hitting the length limits for socket paths
    # (used by the mulitprocessing module) in the PyBDSF calls
    os.environ["TMPDIR"] = tempfile.gettempdir()  # note: no effect if TMPDIR already set

    # Read in beams and frequencies
    with open(cube_beams, 'r') as f:
        lines = f.readlines()
    if not lines:
        raise RuntimeError(f'No beam parameters found in {cube_beams}')
    beams = ast.literal_eval(lines[0])
    with open(cube_frequencies, 'r') as f:
        lines = f.readlines()
    if not lines:
        raise RuntimeError(f'No frequencies found in {cube_frequencies}')
    frequencies = ast.literal_eval(lines[0])

    # Run PyBDSF on the image cube
    img = bdsf.process_image(cube_image, mean_map='zero', rms_box=rmsbox,
                             thresh_pix=threshpix, thresh_isl=threshisl,
                             thresh='hard', adaptive_rms_box=True,
                             adaptive_thresh=adaptive_thresh,
                             rms_box_bright=rmsbox_bright, atrous_do=False,
                             rms_map=True, quiet=True,
                             spectralindex_do=True, beam_spectrum=beams,
                             frequency_sp=frequencies, ncores=ncores,
                             outdir='.')
    img.write_catalog(outfile=output_catalog, format='fits', catalog_type='srl',
                      incl_chan=True, clobber=True)


if __name__ == '__main__':
    descriptiontext = "Make a source catalog from an image cube.\n"

    parser = ArgumentParser(description=descriptiontext, formatter_class=RawTextHelpFormatter)
    parser.add_argument('cube_image', help='Filename of input FITS image cube')
    parser.add_argument('cube_beams', help='Filename of input text file with cube beam parameters')
    parser.add_argument('cube_frequencies', help='Filename of input text file with cube frequency parameters')
    parser.add_argument('output_catalog', help='Filename of output FITS catalog')
    parser.add_argument('--threshisl', help='Island threshold', type=float, default=3.0)
    parser.add_argument('--threshpix', help='Peak pixel threshold', type=float, default=5.0)
    parser.add_argument('--rmsbox', help='Rms box width and step (e.g., "(60, 20)")',
                        type=str, default='(150, 50)')
    parser.add_argument('--rmsbox_bright', help='Rms box for bright sources, width and step (e.g., "(60, 20)")',
                        type=str, default='(35, 7)')
    parser.add_argument('--adaptive_thresh', help='Adaptive threshold', type=float, default=75.0)
    parser.add_argument('--ncores', help='Max number of cores to use', type=int, default=8)

    args = parser.parse_args()
    main(args.cube_image, args.cube_beams, args.cube_frequencies, args.output_catalog,
         threshisl=args.threshisl, threshpix=args.threshpix, rmsbox=args.rmsbox,
         rmsbox_bright=args.rmsbox_bright, adaptive_thresh=args.adaptive_thresh,
         ncores=args.ncores)
