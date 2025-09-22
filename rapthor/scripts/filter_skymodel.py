#!/usr/bin/env python3
"""
Script to filter and group a sky model with an image
"""
import ast
import json
import os
from argparse import ArgumentParser, RawTextHelpFormatter

from astropy.utils import iers
from rapthor.lib import miscellaneous as misc
from lsmtool.filter_skymodel import filter_skymodel

# Turn off astropy's IERS downloads to fix problems in cases where compute
# node does not have internet access
iers.conf.auto_download = False


def main(flat_noise_image, true_sky_image, true_sky_skymodel, apparent_sky_skymodel,
         output_root, vertices_file, beamMS, bright_true_sky_skymodel=None, threshisl=5.0,
         threshpix=7.5, rmsbox=(150, 50), rmsbox_bright=(35, 7), adaptive_thresh=75.0,
         filter_by_mask=True, ncores=8, source_finder='bdsf'):
    """
    Filter the input sky model

    Note: If no islands of emission are detected in the input image, a
    blank sky model is made. If any islands are detected in the input image,
    filtered true-sky and apparent-sky models are made, as well as a FITS clean
    mask (with the filename img_true_sky+'.mask'). Various diagnostics are also
    derived and saved in JSON format.

    Parameters
    ----------
    flat_noise_image : str
        Filename of input image to use to detect sources for filtering. Ideally, this
        should be a flat-noise image (i.e., without primary-beam correction)
    true_sky_image : str
        Filename of input image to use to measure the flux densities sources. This
        should be a true-sky image (i.e., with primary-beam correction)
    true_sky_skymodel : str
        Filename of input makesourcedb sky model, with primary-beam correction.
        Note:
            if this file does not exist, steps related to the input sky model are skipped
            but all other processing is still done
    apparent_sky_skymodel : str
        Filename of input makesourcedb sky model, without primary-beam correction.
        Note:
            if this file does not exist, it is generated from true_sky_skymodel by
            applying the beam attenuation
    output_root : str
        Root of filenames of output makesourcedb sky models, images, and image diagnostics
        files. Output filenames will be:
            output_root+'.apparent_sky.txt'
            output_root+'.true_sky.txt'
            output_root+'.flat_noise_rms.fits'
            output_root+'.true_sky_rms.fits'
            output_root+'.source_catalog.fits'
            output_root+'.image_diagnostics.json'
    vertices_file : str
        Filename of file with vertices
    beamMS : list of str
        The list of MS files to use to derive the beam attenuation and theorectical
        image noise
    bright_true_sky_skymodel : str, optional
        Filename of input makesourcedb sky model of bright sources only, with primary-
        beam correction
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
    filter_by_mask : bool, optional
        If True, filter the input sky model by the PyBDSF-derived mask,
        removing sources that lie in unmasked regions
    ncores : int, optional
        Maximum number of cores to use
    source_finder : str, optional
        The source finder to use, either "sofia" or "bdsf"
    """
    # Check that the true- and apparent-sky models exist (they may have been
    # set in the CWL workflow to dummy names; the bright-sky model will never
    # have a dummy name). If not, set to None as expected by filter_skymodel()
    true_sky_skymodel = true_sky_skymodel if os.path.exists(true_sky_skymodel) else None
    apparent_sky_skymodel = apparent_sky_skymodel if os.path.exists(apparent_sky_skymodel) else None

    nsources = filter_skymodel(
        flat_noise_image,
        true_sky_image,
        true_sky_skymodel,
        apparent_sky_skymodel,
        beam_ms=beamMS,
        vertices_file=vertices_file,
        input_bright_skymodel=bright_true_sky_skymodel,
        output_apparent_sky=f'{output_root}.apparent_sky.txt',
        output_true_sky=f'{output_root}.true_sky.txt',
        output_flat_noise_rms=f'{output_root}.flat_noise_rms.fits',
        output_true_rms=f'{output_root}.true_sky_rms.fits',
        output_catalog=f'{output_root}.source_catalog.fits',
        source_finder=source_finder,
        thresh_isl=threshisl,
        thresh_pix=threshpix,
        rmsbox=rmsbox,
        rmsbox_bright=rmsbox_bright,
        adaptive_thresh=adaptive_thresh,
        filter_by_mask=filter_by_mask,
        ncores=ncores)

    # Write out number of sources found by PyBDSF for later use
    output_diagnostics = f'{output_root}.image_diagnostics.json'
    cwl_output = {'nsources': nsources}
    with open(output_diagnostics, 'w') as fp:
        json.dump(cwl_output, fp)


if __name__ == '__main__':
    descriptiontext = "Filter and group a sky model with an image.\n"

    parser = ArgumentParser(description=descriptiontext, formatter_class=RawTextHelpFormatter)
    parser.add_argument('flat_noise_image',
                        help='Filename of input flat-noise (non-primary-beam-corrected) image')
    parser.add_argument(
        'true_sky_image', help='Filename of input true-sky (primary-beam-corrected) image')
    parser.add_argument('true_sky_skymodel', help='Filename of input true-sky sky model')
    parser.add_argument('apparent_sky_skymodel', help='Filename of input apparent-sky sky model')
    parser.add_argument('output_root', help='Root of output files')
    parser.add_argument('vertices_file', help='Filename of vertices file')
    parser.add_argument('beamMS', help='MS filename(s) to use for beam attenuation')
    parser.add_argument('--bright_true_sky_skymodel', help='Filename of input bright-source true-sky sky model',
                        type=str, default=None)
    parser.add_argument('--threshisl', help='Island threshold', type=float, default=3.0)
    parser.add_argument('--threshpix', help='Peak pixel threshold', type=float, default=5.0)
    parser.add_argument('--rmsbox', help='Rms box width and step (e.g., "(60, 20)")',
                        type=str, default='(150, 50)')
    parser.add_argument('--rmsbox_bright', help='Rms box for bright sources, width and step (e.g., "(60, 20)")',
                        type=str, default='(35, 7)')
    parser.add_argument('--adaptive_thresh', help='Adaptive threshold', type=float, default=75.0)
    parser.add_argument('--filter_by_mask', help='Filter sources by mask',
                        type=ast.literal_eval, default=True)
    parser.add_argument('--ncores', help='Max number of cores to use', type=int, default=8)
    parser.add_argument('--source_finder', help='Source finder to use, either "sofia" or "bdsf"', type=str, default='bdsf')

    args = parser.parse_args()
    main(args.flat_noise_image, args.true_sky_image, args.true_sky_skymodel,
         args.apparent_sky_skymodel, args.output_root, args.vertices_file,
         misc.string2list(args.beamMS),
         bright_true_sky_skymodel=args.bright_true_sky_skymodel, threshisl=args.threshisl,
         threshpix=args.threshpix, rmsbox=args.rmsbox, rmsbox_bright=args.rmsbox_bright,
         adaptive_thresh=args.adaptive_thresh, filter_by_mask=args.filter_by_mask,
         ncores=args.ncores, source_finder=args.source_finder)
