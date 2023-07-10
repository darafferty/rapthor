#!/usr/bin/env python3
"""
Script to filter and group a sky model with an image
"""
import argparse
from argparse import RawTextHelpFormatter
import lsmtool
import numpy as np
import bdsf
from rapthor.lib import miscellaneous as misc
import casacore.tables as pt
import astropy.io.ascii
from astropy.io import fits as pyfits
from astropy import wcs
from astropy.utils import iers
import os
import json
from rapthor.lib.observation import Observation
from scipy.interpolate import interp1d
import subprocess
import tempfile


# Turn off astropy's IERS downloads to fix problems in cases where compute
# node does not have internet access
iers.conf.auto_download = False


def main(flat_noise_image, true_sky_image, true_sky_skymodel, output_root,
         vertices_file, beamMS, bright_true_sky_skymodel=None, threshisl=5.0,
         threshpix=7.5, rmsbox=(150, 50), rmsbox_bright=(35, 7),
         adaptive_rmsbox=True, adaptive_thresh=75.0, filter_by_mask=True,
         remove_negative=False):
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
        Filename of input makesourcedb sky model, with primary-beam correction
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
    adaptive_rmsbox : bool, optional
        Value of adaptive_rms_box PyBDSF parameter
    adaptive_thresh : float, optional
        If adaptive_rmsbox is True, this value sets the threshold above
        which a source will use the small rms box
    filter_by_mask : bool, optional
        If True, filter the input sky model by the PyBDSF-derived mask,
        removing sources that lie in unmasked regions
    remove_negative : bool, optional
        If True, remove negative sky model components
    """
    if rmsbox is not None and isinstance(rmsbox, str):
        rmsbox = eval(rmsbox)
    if isinstance(rmsbox_bright, str):
        rmsbox_bright = eval(rmsbox_bright)
    adaptive_rmsbox = misc.string2bool(adaptive_rmsbox)
    if isinstance(beamMS, str):
        beamMS = misc.string2list(beamMS)

    # Try to set the TMPDIR evn var to a short path, to ensure we do not hit the length
    # limits for socket paths (used by the mulitprocessing module). We try a number of
    # standard paths (the same ones used in the tempfile Python library)
    try:
        old_tmpdir = os.environ["TMPDIR"]
    except KeyError:
        old_tmpdir = None
    for tmpdir in ['/tmp', '/var/tmp', '/usr/tmp']:
        if os.path.exists(tmpdir):
            os.environ["TMPDIR"] = tmpdir
            break

    # Run PyBDSF first on the true-sky image to determine its properties and
    # measure source fluxes. The background RMS map is saved for later use in
    # the image diagnostics step
    img_true_sky = bdsf.process_image(true_sky_image, mean_map='zero', rms_box=rmsbox,
                                      thresh_pix=threshpix, thresh_isl=threshisl,
                                      thresh='hard', adaptive_rms_box=adaptive_rmsbox,
                                      adaptive_thresh=adaptive_thresh,
                                      rms_box_bright=rmsbox_bright, atrous_do=True,
                                      atrous_jmax=3, rms_map=True, quiet=True)
    catalog_filename = output_root+'.source_catalog.fits'
    img_true_sky.write_catalog(outfile=catalog_filename, format='fits', catalog_type='srl',
                               clobber=True)
    true_sky_rms_filename = output_root+'.true_sky_rms.fits'
    img_true_sky.export_image(outfile=true_sky_rms_filename, img_type='rms', clobber=True)
    nsources = img_true_sky.nsrc

    # Run PyBDSF again on the flat-noise image and save the RMS map for later
    # use in the image diagnostics step
    img_flat_noise = bdsf.process_image(flat_noise_image, mean_map='zero', rms_box=rmsbox,
                                        thresh_pix=threshpix, thresh_isl=threshisl,
                                        thresh='hard', adaptive_rms_box=adaptive_rmsbox,
                                        adaptive_thresh=adaptive_thresh, rms_box_bright=rmsbox_bright,
                                        rms_map=True, stop_at='isl', quiet=True)
    flat_noise_rms_filename = output_root+'.flat_noise_rms.fits'
    img_flat_noise.export_image(outfile=flat_noise_rms_filename, img_type='rms', clobber=True)
    del(img_flat_noise)  # helps reduce memory usage

    emptysky = False
    if img_true_sky.nisl > 0:
        maskfile = true_sky_image + '.mask'
        img_true_sky.export_image(outfile=maskfile, clobber=True, img_type='island_mask')

        # Construct polygon needed to trim the mask to the sector
        header = pyfits.getheader(maskfile, 0)
        w = wcs.WCS(header)
        RAind = w.axis_type_names.index('RA')
        Decind = w.axis_type_names.index('DEC')
        vertices = misc.read_vertices(vertices_file)
        RAverts = vertices[0]
        Decverts = vertices[1]
        verts = []
        for RAvert, Decvert in zip(RAverts, Decverts):
            ra_dec = np.array([[0.0, 0.0, 0.0, 0.0]])
            ra_dec[0][RAind] = RAvert
            ra_dec[0][Decind] = Decvert
            verts.append((w.wcs_world2pix(ra_dec, 0)[0][RAind], w.wcs_world2pix(ra_dec, 0)[0][Decind]))

        hdu = pyfits.open(maskfile, memmap=False)
        data = hdu[0].data

        # Rasterize the poly
        data_rasertize = data[0, 0, :, :]
        data_rasertize = misc.rasterize(verts, data_rasertize)
        data[0, 0, :, :] = data_rasertize

        hdu[0].data = data
        hdu.writeto(maskfile, overwrite=True)

        # Select the best MS for the beam attenuation
        if len(beamMS) > 1:
            ms_times = []
            for ms in beamMS:
                tab = pt.table(ms, ack=False)
                ms_times.append(np.mean(tab.getcol('TIME')))
                tab.close()
            ms_times_sorted = sorted(ms_times)
            mid_time = ms_times_sorted[int(len(ms_times)/2)]
            beam_ind = ms_times.index(mid_time)
        else:
            beam_ind = 0
        with tempfile.TemporaryDirectory() as temp_ms_dir:
            # Copy the beam MS file to TMPDIR, as this is likely to have faster
            # I/O (important for EveryBeam, which is used by LSMTool)
            # TODO: for now we copy the full file, but it may be possible to cache
            # just the parts of the MS that EveryBeam needs
            beam_ms = os.path.join(temp_ms_dir, os.path.basename(beamMS[beam_ind]))
            subprocess.check_call(['cp', '-r', '-L', '--no-preserve=mode', beamMS[beam_ind], beam_ms])

            # Load the sky model with the associated beam MS
            try:
                s_in = lsmtool.load(true_sky_skymodel, beamMS=beam_ms)
            except astropy.io.ascii.InconsistentTableError:
                emptysky = True

            # If bright sources were peeled before imaging, add them back
            if bright_true_sky_skymodel is not None:
                try:
                    s_bright = lsmtool.load(bright_true_sky_skymodel)

                    # Rename the bright sources, removing the '_sector_*' added previously
                    # (otherwise the '_sector_*' text will be added every iteration,
                    # eventually making for very long source names)
                    new_names = [name.split('_sector')[0] for name in s_bright.getColValues('Name')]
                    s_bright.setColValues('Name', new_names)
                    if not emptysky:
                        s_in.concatenate(s_bright)
                    else:
                        s_in = s_bright
                        emptysky = False
                except astropy.io.ascii.InconsistentTableError:
                    pass

            # Do final filtering and write out the sky models
            if not emptysky:
                if remove_negative:
                    # Keep only those sources with positive flux densities
                    s_in.select('I > 0.0')
                if s_in and filter_by_mask:
                    # Keep only those sources in PyBDSF masked regions
                    s_in.select('{} == True'.format(maskfile))
                if s_in:
                    # Write out apparent- and true-sky models
                    del(img_true_sky)  # helps reduce memory usage
                    s_in.group(maskfile)  # group the sky model by mask islands
                    s_in.write(output_root+'.true_sky.txt', clobber=True)
                    s_in.write(output_root+'.apparent_sky.txt', clobber=True, applyBeam=True)
                else:
                    emptysky = True
    else:
        emptysky = True

    if emptysky:
        # No sources cleaned/found in image, so just make a dummy sky model with single,
        # very faint source at center
        nsources = 0
        dummylines = ["Format = Name, Type, Patch, Ra, Dec, I, SpectralIndex, LogarithmicSI, "
                      "ReferenceFrequency='100000000.0', MajorAxis, MinorAxis, Orientation\n"]
        ra, dec = img_true_sky.pix2sky((img_true_sky.shape[-2]/2.0, img_true_sky.shape[-1]/2.0))
        if ra < 0.0:
            ra += 360.0
        ra = misc.ra2hhmmss(ra)
        sra = str(ra[0]).zfill(2)+':'+str(ra[1]).zfill(2)+':'+str("%.6f" % (ra[2])).zfill(6)
        dec = misc.dec2ddmmss(dec)
        decsign = ('-' if dec[3] < 0 else '+')
        sdec = decsign+str(dec[0]).zfill(2)+'.'+str(dec[1]).zfill(2)+'.'+str("%.6f" % (dec[2])).zfill(6)
        dummylines.append(',,p1,{0},{1}\n'.format(sra, sdec))
        dummylines.append('s0c0,POINT,p1,{0},{1},0.00000001,'
                          '[0.0,0.0],false,100000000.0,,,\n'.format(sra, sdec))
        with open(output_root+'.apparent_sky.txt', 'w') as f:
            f.writelines(dummylines)
        with open(output_root+'.true_sky.txt', 'w') as f:
            f.writelines(dummylines)

    # Write out number of sources found by PyBDSF for later use
    cwl_output = {'nsources': nsources}
    with open(output_root+'.image_diagnostics.json', 'w') as fp:
        json.dump(cwl_output, fp)

    # Set the TMPDIR env var back to its original value
    if old_tmpdir is not None:
        os.environ["TMPDIR"] = old_tmpdir


if __name__ == '__main__':
    descriptiontext = "Filter and group a sky model with an image.\n"

    parser = argparse.ArgumentParser(description=descriptiontext, formatter_class=RawTextHelpFormatter)
    parser.add_argument('flat_noise_image', help='Filename of input flat-noise (non-primary-beam-corrected) image')
    parser.add_argument('true_sky_image', help='Filename of input true-sky (primary-beam-corrected) image')
    parser.add_argument('true_sky_skymodel', help='Filename of input true-sky sky model')
    parser.add_argument('output_root', help='Root of output files')
    parser.add_argument('vertices_file', help='Filename of vertices file')
    parser.add_argument('beamMS', help='MS filename to use for beam attenuation')
    parser.add_argument('--bright_true_sky_skymodel', help='Filename of input bright-source true-sky sky model',
                        type=str, default=None)
    parser.add_argument('--threshisl', help='Island threshold', type=float, default=3.0)
    parser.add_argument('--threshpix', help='Peak pixel threshold', type=float, default=5.0)
    parser.add_argument('--rmsbox', help='Rms box width and step (e.g., "(60, 20)")',
                        type=str, default='(150, 50)')
    parser.add_argument('--rmsbox_bright', help='Rms box for bright sources, width and step (e.g., "(60, 20)")',
                        type=str, default='(35, 7)')
    parser.add_argument('--adaptive_rmsbox', help='Use an adaptive rms box', type=str, default='True')

    args = parser.parse_args()
    main(args.flat_noise_image, args.true_sky_image, args.true_sky_skymodel, args.output_root,
         args.vertices_file, args.beamMS, bright_true_sky_skymodel=args.bright_true_sky_skymodel,
         threshisl=args.threshisl, threshpix=args.threshpix, rmsbox=args.rmsbox,
         rmsbox_bright=args.rmsbox_bright, adaptive_rmsbox=args.adaptive_rmsbox)
