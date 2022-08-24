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
import os


def main(input_image, input_skymodel_pb, input_bright_skymodel_pb, output_root,
         vertices_file, threshisl=5.0, threshpix=7.5, rmsbox=(150, 50),
         rmsbox_bright=(35, 7), adaptive_rmsbox=True,
         use_adaptive_threshold=False, adaptive_thresh=75.0, beamMS=None,
         peel_bright=False):
    """
    Filter the input sky model so that they lie in islands in the image

    Parameters
    ----------
    input_image : str
        Filename of input image to use to detect sources for filtering. Ideally, this
        should be a flat-noise image (i.e., without primary-beam correction)
    input_skymodel_pb : str
        Filename of input makesourcedb sky model, with primary-beam correction
    input_bright_skymodel_pb : str
        Filename of input makesourcedb sky model of bright sources only, with primary-
        beam correction
    output_root : str
        Root of filename of output makesourcedb sky models. Output filenames will be
        output_root+'.apparent_sky.txt' and output_root+'.true_sky.txt'
    vertices_file : str
        Filename of file with vertices
    threshisl : float, optional
        Value of thresh_isl PyBDSF parameter
    threshpix : float, optional
        Value of thresh_pix PyBDSF parameter
    rmsbox : tuple of floats, optional
        Value of rms_box PyBDSF parameter
    rmsbox_bright : tuple of floats, optional
        Value of rms_box_bright PyBDSF parameter
    adaptive_rmsbox : tuple of floats, optional
        Value of adaptive_rms_box PyBDSF parameter
    use_adaptive_threshold : bool, optional
        If True, use an adaptive threshold estimated from the negative values in
        the image
    adaptive_thresh : float, optional
        If adaptive_rmsbox is True, this value sets the threshold above
        which a source will use the small rms box
    peel_bright : bool, optional
        If True, bright sources were peeled, so add then back before filtering
    """
    if rmsbox is not None and isinstance(rmsbox, str):
        rmsbox = eval(rmsbox)
    if isinstance(rmsbox_bright, str):
        rmsbox_bright = eval(rmsbox_bright)
    adaptive_rmsbox = misc.string2bool(adaptive_rmsbox)
    use_adaptive_threshold = misc.string2bool(use_adaptive_threshold)
    if isinstance(beamMS, str):
        beamMS = misc.string2list(beamMS)
    peel_bright = misc.string2bool(peel_bright)

    # Try to set the TMPDIR evn var to a short path, to ensure we do not hit the length
    # limits for socket paths (used by the mulitprocessing module). We try a number of
    # standard paths (the same ones used in the tempfile Python library)
    old_tmpdir = os.environ["TMPDIR"]
    for tmpdir in ['/tmp', '/var/tmp', '/usr/tmp']:
        if os.path.exists(tmpdir):
            os.environ["TMPDIR"] = tmpdir
            break

    # Run PyBDSF to make a mask for grouping
    if use_adaptive_threshold:
        # Get an estimate of the rms by running PyBDSF to make an rms map
        img = bdsf.process_image(input_image, mean_map='zero', rms_box=rmsbox,
                                 thresh_pix=threshpix, thresh_isl=threshisl,
                                 thresh='hard', adaptive_rms_box=adaptive_rmsbox,
                                 adaptive_thresh=adaptive_thresh, rms_box_bright=rmsbox_bright,
                                 rms_map=True, quiet=True, stop_at='isl')

        # Find min and max pixels
        max_neg_val = abs(np.min(img.ch0_arr))
        max_neg_pos = np.where(img.ch0_arr == np.min(img.ch0_arr))
        max_pos_val = abs(np.max(img.ch0_arr))
        max_pos_pos = np.where(img.ch0_arr == np.max(img.ch0_arr))

        # Estimate new thresh_isl from min pixel value's sigma, but don't let
        # it get higher than 1/2 of the peak's sigma
        threshisl_neg = 2.0 * max_neg_val / img.rms_arr[max_neg_pos][0]
        max_sigma = max_pos_val / img.rms_arr[max_pos_pos][0]
        if threshisl_neg > max_sigma / 2.0:
            threshisl_neg = max_sigma / 2.0

        # Use the new threshold only if it is larger than the user-specified one
        if threshisl_neg > threshisl:
            threshisl = threshisl_neg

    img = bdsf.process_image(input_image, mean_map='zero', rms_box=rmsbox,
                             thresh_pix=threshpix, thresh_isl=threshisl,
                             thresh='hard', adaptive_rms_box=adaptive_rmsbox,
                             adaptive_thresh=adaptive_thresh, rms_box_bright=rmsbox_bright,
                             atrous_do=True, atrous_jmax=3, rms_map=True, quiet=True)

    emptysky = False
    if img.nisl > 0:
        maskfile = input_image + '.mask'
        img.export_image(outfile=maskfile, clobber=True, img_type='island_mask')

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

        # Now filter the sky model using the mask made above
        if len(beamMS) > 1:
            # Select the best MS for the beam attenuation
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
        try:
            s = lsmtool.load(input_skymodel_pb, beamMS=beamMS[beam_ind])
        except astropy.io.ascii.InconsistentTableError:
            emptysky = True
        if peel_bright:
            try:
                # If bright sources were peeled before imaging, add them back
                s_bright = lsmtool.load(input_bright_skymodel_pb, beamMS=beamMS[beam_ind])

                # Rename the bright sources, removing the '_sector_*' added previously
                # (otherwise the '_sector_*' text will be added every iteration,
                # eventually making for very long source names)
                new_names = [name.split('_sector')[0] for name in s_bright.getColValues('Name')]
                s_bright.setColValues('Name', new_names)
                if not emptysky:
                    s.concatenate(s_bright)
                else:
                    s = s_bright
                    emptysky = False
            except astropy.io.ascii.InconsistentTableError:
                pass
        if not emptysky:
            s.select('{} == True'.format(maskfile))  # keep only those in PyBDSF masked regions
            if len(s) == 0:
                emptysky = True
            else:
                # Write out apparent and true-sky models
                del(img)  # helps reduce memory usage
                s.group(maskfile)  # group the sky model by mask islands
                s.write(output_root+'.true_sky.txt', clobber=True)
                s.write(output_root+'.apparent_sky.txt', clobber=True, applyBeam=True)
    else:
        emptysky = True

    if emptysky:
        # No sources cleaned/found in image, so just make a dummy sky model with single,
        # very faint source at center
        dummylines = ["Format = Name, Type, Patch, Ra, Dec, I, SpectralIndex, LogarithmicSI, "
                      "ReferenceFrequency='100000000.0', MajorAxis, MinorAxis, Orientation\n"]
        ra, dec = img.pix2sky((img.shape[-2]/2.0, img.shape[-1]/2.0))
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

    # Set the TMPDIR env var back to its original value
    os.environ["TMPDIR"] = old_tmpdir


if __name__ == '__main__':
    descriptiontext = "Filter and group a sky model with an image.\n"

    parser = argparse.ArgumentParser(description=descriptiontext, formatter_class=RawTextHelpFormatter)
    parser.add_argument('input_image', help='Filename of input image')
    parser.add_argument('input_skymodel_pb', help='Filename of input sky model')
    parser.add_argument('input_bright_skymodel_pb', help='Filename of input bright-source sky model')
    parser.add_argument('output_skymodel', help='Filename of output sky model')
    parser.add_argument('vertices_file', help='Filename of vertices file')
    parser.add_argument('--threshisl', help='Island threshold', type=float, default=3.0)
    parser.add_argument('--threshpix', help='Peak pixel threshold', type=float, default=5.0)
    parser.add_argument('--rmsbox', help='Rms box width and step (e.g., "(60, 20)")',
                        type=str, default='(60, 20)')
    parser.add_argument('--rmsbox_bright', help='Rms box for bright sources, width and step (e.g., "(60, 20)")',
                        type=str, default='(60, 20)')
    parser.add_argument('--adaptive_rmsbox', help='Use an adaptive rms box', type=str, default='False')
    parser.add_argument('--beamMS', help='MS filename to use for beam attenuation', type=str, default=None)
    parser.add_argument('--peel_bright', help='Bright sources were peeling before imaging',
                        type=str, default='False')

    args = parser.parse_args()
    main(args.input_image, args.input_skymodel_pb, args.input_bright_skymodel_pb,
         args.output_skymodel, args.vertices_file, threshisl=args.threshisl,
         threshpix=args.threshpix, rmsbox=args.rmsbox,
         rmsbox_bright=args.rmsbox_bright, adaptive_rmsbox=args.adaptive_rmsbox,
         beamMS=args.beamMS, peel_bright=args.peel_bright)
