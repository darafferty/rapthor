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
import sys
import tempfile

# Turn off astropy's IERS downloads to fix problems in cases where compute
# node does not have internet access
iers.conf.auto_download = False


def calc_theoretical_noise(mslist, w_factor=1.5):
    """
    Return the expected theoretical image noise for a dataset

    Note: the calculations follow those of SKA Memo 113 (see
    http://www.skatelescope.org/uploaded/59513_113_Memo_Nijboer.pdf) and
    assume no tapering. International stations are not included.

    Parameters
    ----------
    mslist : list of str
        List of the filenames of the input MS files
    w_factor : float, optional
        Factor for increase of noise due to the weighting scheme used
        in imaging (typically ranges from 1.3 - 2)

    Returns
    -------
    noise, unflagged_fraction : tuple of floats
        Estimate of the expected theoretical noise in Jy/beam and the
        unflagged fraction of the input data
    """
    nobs = len(mslist)
    if nobs == 0:
        # If no MS files, just return zero for the noise as we cannot
        # estimate it
        return 0.0

    # Find the total time and the average total bandwidth, average frequency,
    # average unflagged fraction, and average number of core and remote stations
    # (for the averages, assume each observation has equal weight)
    total_time = 0
    total_bandwidth = 0
    ncore = 0
    nremote = 0
    mid_freq = 0
    unflagged_fraction = 0
    for ms in mslist:
        obs = Observation(ms)
        total_time += obs.endtime - obs.starttime  # sec
        total_bandwidth += obs.endfreq - obs.startfreq  # Hz
        ncore += len([stat for stat in obs.stations if stat.startswith('CS')])
        nremote += len([stat for stat in obs.stations if stat.startswith('RS')])
        mid_freq += (obs.endfreq + obs.startfreq) / 2 / 1e6  # MHz
        unflagged_fraction += find_unflagged_fraction(ms)
    total_bandwidth /= nobs
    ncore = int(np.round(ncore / nobs))
    nremote = int(np.round(nremote / nobs))
    mean_freq = mid_freq / nobs
    unflagged_fraction /= nobs

    # Define table of system equivalent flux densities and interpolate
    # to get the values at the mean frequency of the input observations.
    # Note: values were taken from Table 9 of SKA Memo 113
    sefd_freq_MHz = np.array([15, 30, 45, 60, 75, 120, 150, 180, 210, 240])
    sefd_core_kJy = np.array([483, 89, 48, 32, 51, 3.6, 2.8, 3.2, 3.7, 4.1])
    sefd_remote_kJy = np.array([483, 89, 48, 32, 51, 1.8, 1.4, 1.6, 1.8, 2.0])
    f_core = interp1d(sefd_freq_MHz, sefd_core_kJy)
    f_remote = interp1d(sefd_freq_MHz, sefd_remote_kJy)
    sefd_core = f_core(mean_freq) * 1e3  # Jy
    sefd_remote = f_remote(mean_freq) * 1e3  # Jy

    # Calculate the theoretical noise, adjusted for the unflagged fraction
    core_term = ncore * (ncore - 1) / 2 / sefd_core**2
    remote_term = nremote * (nremote - 1) / 2 / sefd_remote**2
    mixed_term = ncore * nremote / (sefd_core * sefd_remote)
    noise = w_factor / np.sqrt(2 * (2 * total_time * total_bandwidth) *
                               (core_term + mixed_term + remote_term))  # Jy
    noise /= np.sqrt(unflagged_fraction)

    return (noise, unflagged_fraction)


def find_unflagged_fraction(ms_file):
    """
    Finds the fraction of data that is unflagged

    Parameters
    ----------
    ms_file : str
        Filename of input MS

    Returns
    -------
    unflagged_fraction : float
        Fraction of unflagged data
    """
    # Call taql. Note that we do not use pt.taql(), as pt.taql() can cause
    # hanging/locking issues on some systems
    if (sys.version_info.major, sys.version_info.minor) >= (3, 7):
        # Note: the capture_output argument was added in Python 3.7
        result = subprocess.run("taql 'CALC sum([select nfalse(FLAG) from {0}]) / "
                                "sum([select nelements(FLAG) from {0}])'".format(ms_file),
                                shell=True, capture_output=True, check=True)
        unflagged_fraction = float(result.stdout)
    else:
        p = subprocess.Popen("taql 'CALC sum([select nfalse(FLAG) from {0}]) / "
                             "sum([select nelements(FLAG) from {0}])'".format(ms_file),
                             shell=True, stdout=subprocess.PIPE)
        r = p.communicate()
        if p.returncode != 0:
            raise subprocess.CalledProcessError(p.returncode, p.args)
        unflagged_fraction = float(r[0])

    return unflagged_fraction


def main(input_image, input_skymodel_pb, output_root, vertices_file, beamMS,
         input_bright_skymodel_pb=None, threshisl=5.0, threshpix=7.5,
         rmsbox=(150, 50), rmsbox_bright=(35, 7), adaptive_rmsbox=True,
         use_adaptive_threshold=False, adaptive_thresh=75.0,
         comparison_skymodel=None, filter_by_mask=True, remove_negative=False):
    """
    Filter the input sky model

    Note: If no islands of emission are detected in the input image, a
    blank sky model is made. If any islands are detected in the input image,
    filtered true-sky and apparent-sky models are made, as well as a FITS clean
    mask (with the filename input_image+'.mask'). Various diagnostics are also
    derived and saved in JSON format.

    Parameters
    ----------
    input_image : str
        Filename of input image to use to detect sources for filtering. Ideally, this
        should be a flat-noise image (i.e., without primary-beam correction)
    input_skymodel_pb : str
        Filename of input makesourcedb sky model, with primary-beam correction
    output_root : str
        Root of filenames of output makesourcedb sky models and image diagnostics
        files. Output filenames will be output_root+'.apparent_sky.txt',
        output_root+'.true_sky.txt', and output_root+'.image_diagnostics.json'
    vertices_file : str
        Filename of file with vertices
    beamMS : list of str
        The list of MS files to use to derive the beam attenuation and theorectical
        image noise
    input_bright_skymodel_pb : str, optional
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
    adaptive_rmsbox : tuple of floats, optional
        Value of adaptive_rms_box PyBDSF parameter
    use_adaptive_threshold : bool, optional
        If True, use an adaptive threshold estimated from the negative values in
        the image
    adaptive_thresh : float, optional
        If adaptive_rmsbox is True, this value sets the threshold above
        which a source will use the small rms box
    comparison_skymodel : str, optional
        The filename of the sky model to use for flux scale and astrometry
        comparisons
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
    use_adaptive_threshold = misc.string2bool(use_adaptive_threshold)
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

    # Collect some diagnostic numbers for later reporting. Note: we ensure all
    # non-integer numbers are float, as, e.g., np.float32 is not supported by json.dump()
    theoretical_rms, unflagged_fraction = calc_theoretical_noise(beamMS)  # Jy/beam
    min_rms = float(np.min(img.rms_arr))  # Jy/beam
    max_rms = float(np.max(img.rms_arr))  # Jy/beam
    mean_rms = float(np.mean(img.rms_arr))  # Jy/beam
    median_rms = float(np.median(img.rms_arr))  # Jy/beam
    nsources = img.nsrc
    dynamic_range_global = float(np.max(img.ch0_arr) / min_rms)
    dynamic_range_local = float(np.max(img.ch0_arr / img.rms_arr))
    beam_fwhm = [float(img.beam[0]), float(img.beam[1]), float(img.beam[2])]  # (maj, min, pa), all in deg
    freq = float(img.frequency)  # Hz
    cwl_output = {'theoretical_rms': theoretical_rms,
                  'unflagged_data_fraction': unflagged_fraction,
                  'min_rms': min_rms,
                  'max_rms': max_rms,
                  'mean_rms': mean_rms,
                  'median_rms': median_rms,
                  'dynamic_range_global': dynamic_range_global,
                  'dynamic_range_local': dynamic_range_local,
                  'nsources': nsources,
                  'freq': freq,
                  'beam_fwhm': beam_fwhm}

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
                s_in = lsmtool.load(input_skymodel_pb, beamMS=beam_ms)
            except astropy.io.ascii.InconsistentTableError:
                emptysky = True

            # If bright sources were peeled before imaging, add them back
            if input_bright_skymodel_pb is not None:
                try:
                    s_bright = lsmtool.load(input_bright_skymodel_pb)

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
                    del(img)  # helps reduce memory usage
                    s_in.group(maskfile)  # group the sky model by mask islands
                    s_in.write(output_root+'.true_sky.txt', clobber=True)
                    s_in.write(output_root+'.apparent_sky.txt', clobber=True, applyBeam=True)
                else:
                    emptysky = True
    else:
        emptysky = True

    # Get the flux scale and astrometry diagnostics if possible and save all the image
    # diagnostics to the output JSON file
    if not emptysky:
        if comparison_skymodel is None:
            # Download a TGSS sky model around the midpoint of the input sky model,
            # using a 5-deg radius to ensure the field is fully covered
            _, _, midRA, midDec = s_in._getXY()
            try:
                s_comp = lsmtool.load('tgss', VOPosition=[midRA, midDec], VORadius=5.0)
            except OSError:
                # Problem encountered when downloading model from the TGSS server, so
                # skip the comparison
                s_comp = None
        else:
            s_comp = lsmtool.load(comparison_skymodel)

        # Group the comparison sky model into sources and select only those sources
        # that are composed entirely of type "POINT", as the comparison method in
        # LSMTool works reliably only for this type
        if s_comp is not None:
            # Group using FWHM of 40 arcsec, the approximate TGSS resolution
            s_comp.group('threshold', FWHM='40.0 arcsec', threshold=0.05)

            # Keep POINT-only sources
            source_type = s_comp.getColValues('Type')
            patch_names = s_comp.getColValues('Patch')
            non_point_patch_names = set(patch_names[np.where(source_type != 'POINT')])
            ind = []
            for patch_name in non_point_patch_names:
                ind.extend(s_comp.getRowIndex(patch_name))
            s_comp.remove(np.array(ind))

            # Check if there is a sufficient number of sources to do the comparison with.
            # If there is, do it and append the resulting diagnostics dict to the
            # existing one
            #
            # Note: the various ratios are all calculated as (s_in / s_comp) and the
            # differences as (s_in - s_comp). If there are no successful matches,
            # the compare() method returns None
            if (s_comp and len(s_in.getPatchNames()) >= 10 and len(s_comp.getPatchNames()) >= 10):
                flux_astrometry_diagnostics = s_in.compare(s_comp, radius='5 arcsec',
                                                           excludeMultiple=True, make_plots=False)
                if flux_astrometry_diagnostics is not None:
                    cwl_output.update(flux_astrometry_diagnostics)
    with open(output_root+'.image_diagnostics.json', 'w') as fp:
        json.dump(cwl_output, fp)

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
    if old_tmpdir is not None:
        os.environ["TMPDIR"] = old_tmpdir


if __name__ == '__main__':
    descriptiontext = "Filter and group a sky model with an image.\n"

    parser = argparse.ArgumentParser(description=descriptiontext, formatter_class=RawTextHelpFormatter)
    parser.add_argument('input_image', help='Filename of input image')
    parser.add_argument('input_skymodel_pb', help='Filename of input sky model')
    parser.add_argument('output_skymodel', help='Filename of output sky model')
    parser.add_argument('vertices_file', help='Filename of vertices file')
    parser.add_argument('--input_bright_skymodel_pb', help='Filename of input bright-source sky model',
                        type=str, default=None)
    parser.add_argument('--threshisl', help='Island threshold', type=float, default=3.0)
    parser.add_argument('--threshpix', help='Peak pixel threshold', type=float, default=5.0)
    parser.add_argument('--rmsbox', help='Rms box width and step (e.g., "(60, 20)")',
                        type=str, default='(150, 50)')
    parser.add_argument('--rmsbox_bright', help='Rms box for bright sources, width and step (e.g., "(60, 20)")',
                        type=str, default='(35, 7)')
    parser.add_argument('--adaptive_rmsbox', help='Use an adaptive rms box', type=str, default='True')
    parser.add_argument('--beamMS', help='MS filename to use for beam attenuation', type=str, default=None)

    args = parser.parse_args()
    main(args.input_image, args.input_skymodel_pb, args.output_skymodel,
         args.vertices_file, input_bright_skymodel_pb=args.input_bright_skymodel_pb,
         threshisl=args.threshisl, threshpix=args.threshpix, rmsbox=args.rmsbox,
         rmsbox_bright=args.rmsbox_bright, adaptive_rmsbox=args.adaptive_rmsbox,
         beamMS=args.beamMS)
