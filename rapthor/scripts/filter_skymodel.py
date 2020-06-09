#! /usr/bin/env python3
"""
Script to filter a sky model with an image
"""
import argparse
from argparse import RawTextHelpFormatter
import lsmtool
import numpy as np
import bdsf
from rapthor.lib import miscellaneous as misc
import casacore.tables as pt
import astropy.io.ascii


def ra2hhmmss(deg):
    """Convert RA coordinate (in degrees) to HH MM SS"""

    from math import modf
    if deg < 0:
        deg += 360.0
    x, hh = modf(deg/15.)
    x, mm = modf(x*60)
    ss = x*60

    return (int(hh), int(mm), ss)


def dec2ddmmss(deg):
    """Convert DEC coordinate (in degrees) to DD MM SS"""

    from math import modf
    sign = (-1 if deg < 0 else 1)
    x, dd = modf(abs(deg))
    x, ma = modf(x*60)
    sa = x*60

    return (int(dd), int(ma), sa, sign)


def main(input_image, input_skymodel_nonpb, input_skymodel_pb, output_root,
         threshisl=5.0, threshpix=7.5, rmsbox=(150, 50), rmsbox_bright=(35, 7),
         adaptive_rmsbox=True, use_adaptive_threshold=False, adaptive_thresh=75.0,
         beamMS=None):
    """
    Filter the input sky model so that they lie in islands in the image

    Parameters
    ----------
    input_image : str
        Filename of input image to use to detect sources for filtering. Ideally, this
        should be a flat-noise image (i.e., without primary-beam correction)
    input_skymodel_nonpb : str
        Filename of input makesourcedb sky model, without primary-beam correction
    input_skymodel_pb : str, optional
        Filename of input makesourcedb sky model, with primary-beam correction
    output_root : str
        Root of filename of output makesourcedb sky models. Output filenames will be
        output_root+'.apparent_sky' and output_root+'.true_sky'
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
    """
    if rmsbox is not None and isinstance(rmsbox, str):
        rmsbox = eval(rmsbox)
    if isinstance(rmsbox_bright, str):
        rmsbox_bright = eval(rmsbox_bright)
    adaptive_rmsbox = misc.string2bool(adaptive_rmsbox)
    use_adaptive_threshold = misc.string2bool(use_adaptive_threshold)
    if isinstance(beamMS, str):
        beamMS = misc.string2list(beamMS)

    if use_adaptive_threshold:
        # Get an estimate of the rms
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
        del img

        # TODO: remove the following attenuation once WSClean correctly produces
        # non-pb-corrected sky models
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
        try:
    #         s = lsmtool.load(input_skymodel_nonpb)  # normally, load nonpb model and don't attenuate!
            s = lsmtool.load(input_skymodel_pb, beamMS=beamMS[beam_ind])
            s.select('{} == True'.format(maskfile))  # keep only those in PyBDSF masked regions
            if len(s) == 0:
                emptysky = True
            else:
                s.group(maskfile)  # group the sky model by mask islands
        #         s.write(output_root+'.apparent_sky', clobber=True)  # normally don't attenuate!
                s.write(output_root+'.apparent_sky', clobber=True, applyBeam=True)

                s = lsmtool.load(input_skymodel_pb)
                s.select('{} == True'.format(maskfile))  # keep only those in PyBDSF masked regions
                s.group(maskfile)  # group the sky model by mask islands
                s.write(output_root+'.true_sky', clobber=True)
        except astropy.io.ascii.InconsistentTableError:
            emptysky = True
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
        ra = ra2hhmmss(ra)
        sra = str(ra[0]).zfill(2)+':'+str(ra[1]).zfill(2)+':'+str("%.6f" % (ra[2])).zfill(6)
        dec = dec2ddmmss(dec)
        decsign = ('-' if dec[3] < 0 else '+')
        sdec = decsign+str(dec[0]).zfill(2)+'.'+str(dec[1]).zfill(2)+'.'+str("%.6f" % (dec[2])).zfill(6)
        dummylines.append(',,p1,{0},{1}\n'.format(sra, sdec))
        dummylines.append('s0c0,POINT,p1,{0},{1},0.00000001,'
                          '[0.0,0.0],false,100000000.0,,,\n'.format(sra, sdec))
        with open(output_root+'.apparent_sky', 'w') as f:
            f.writelines(dummylines)
        with open(output_root+'.true_sky', 'w') as f:
            f.writelines(dummylines)


if __name__ == '__main__':
    descriptiontext = "Filter a sky model with an image.\n"

    parser = argparse.ArgumentParser(description=descriptiontext, formatter_class=RawTextHelpFormatter)
    parser.add_argument('input_image', help='Filename of input image')
    parser.add_argument('input_skymodel_nonpb', help='Filename of input sky model')
    parser.add_argument('input_skymodel_pb', help='Filename of input sky model')
    parser.add_argument('output_skymodel', help='Filename of output sky model')
    parser.add_argument('--threshisl', help='', type=float, default=3.0)
    parser.add_argument('--threshpix', help='', type=float, default=5.0)
    parser.add_argument('--rmsbox', help='rms box width and step (e.g., "(60, 20)")',
                        type=str, default='(60, 20)')
    parser.add_argument('--rmsbox_bright', help='rms box for bright sources, width and step (e.g., "(60, 20)")',
                        type=str, default='(60, 20)')
    parser.add_argument('--adaptive_rmsbox', help='use an adaptive rms box', type=str, default='False')
    parser.add_argument('--beamMS', help='', type=str, default=None)

    args = parser.parse_args()
    main(args.input_image, args.input_skymodel_nonpb, args.input_skymodel_pb, args.output_skymodel,
         threshisl=args.threshisl, threshpix=args.threshpix, rmsbox=args.rmsbox,
         rmsbox_bright=args.rmsbox_bright, adaptive_rmsbox=args.adaptive_rmsbox,
         beamMS=args.beamMS)
