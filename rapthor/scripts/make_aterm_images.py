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


def main(h5parmfile, soltabname='phase000', outroot='', bounds_deg=None,
         bounds_mid_deg=None, skymodel=None, solsetname='sol000',
         padding_fraction=1.4, cellsize_deg=0.1, smooth_deg=0,
         time_avg_factor=1, interp_kind='nearest', screen_type='voronoi')):
    """
    Make a-term FITS images

    Parameters
    ----------
    h5parmfile : str
        Filename of h5parm
    soltabname : str, optional
        Name of soltab to use. If "gain" is in the name, phase and amplitudes are used
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
    screen_type : str, optional
        Kind of screen to use: 'voronoi' (simple Voronoi tessellation) or 'kl' (Karhunen-
        Lo`eve transform)

    Returns
    -------
    result : dict
        Dict with list of FITS files
    """
    # Read in solutions
    H = h5parm(h5parmfile)
    solset = H.getSolset(solsetname)
    if 'gain' in soltabname:
        soltab = solset.getSoltab(soltabname.replace('gain', 'amplitude'))
        soltab_ph = solset.getSoltab(soltabname.replace('gain', 'phase'))
    else:
        soltab = solset.getSoltab(soltabname)

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

    # Read in solutions
    if 'amplitude' in soltab.getType():
        # scalarphases and XX+YY amplitudes
        vals_ph = soltab_ph.val
        vals = soltab.val
    else:
        # scalarphase -> set amplitudes to unity
        vals_ph = soltab.val
        vals = np.ones_like(vals_ph)
    times = soltab.time
    freqs = soltab.freq
    ants = soltab.ant
    axis_names = soltab.getAxesNames()
    source_names = soltab.dir[:]

    # If needed, combine fast-scalarphase solutions with slow amps by interpolating
    # the slow amps to the fast time and frequency grid
    if 'amplitude' in soltab.getType():
        soltab_fast = soltab_ph
        times_fast = soltab_fast.time
        freqs_fast = soltab_fast.freq

        # Interpolate the slow gains to the fast times and frequencies
        axis_names = soltab.getAxesNames()
        time_ind = axis_names.index('time')
        freq_ind = axis_names.index('freq')
        fast_axis_names = soltab_fast.getAxesNames()
        fast_time_ind = fast_axis_names.index('time')
        fast_freq_ind = fast_axis_names.index('freq')
        if len(times) == 1:
            # If just a single time, we just repeat the values as needed
            new_shape = list(vals.shape)
            new_shape[time_ind] = val_ph.shape[fast_time_ind]
            new_shape[freq_ind] = val_ph.shape[fast_freq_ind]
            vals = np.resize(vals, new_shape)
        else:
            # Interpolate (in log space)
            logvals = np.log10(vals)
            if vals.shape[time_ind] != vals_ph.shape[fast_time_ind]:
                f = si.interp1d(times, logvals, axis=time_ind, kind=interp_kind, fill_value='extrapolate')
                logvals = f(times_fast)
            if vals.shape[freq_ind] != vals_ph.shape[fast_freq_ind]:
                f = si.interp1d(freqs, logvals, axis=freq_ind, kind=interp_kind, fill_value='extrapolate')
                logvals = f(freqs_fast)
            vals = 10**(logvals)
        freqs = freqs_fast
        times = times_fast

    if screen_type == 'kl':
        # Do Karhunen-Lo`eve transform

        # First, make a new h5parm file with the solutions we want to fit (for now, fit
        # phases and amp_XX and amp_YY; later, we can look into more optimal ways to
        # fit the amplitudes)
        if os.path.exists(outh5parm):
            os.remove(outh5parm)
        ho = h5parm(outh5parm, readonly=False)
        sso = ho.makeSolset(solsetName='sol000', addTables=False)
        if 'amplitude' in soltab.getType():
            # Scalarphases and XX+YY amplitudes

            # Copy phase soltab as is
            soltab_ph.obj._f_copy_children(sso.obj, recursive=True, overwrite=True)

            # Make soltab for amplitudes (axes are ['time', 'freq', 'ant', 'dir', 'pol'])
            ants = soltab_ph.ant[:]
            dirs = soltab_ph.dir[:]
            pols = soltab_ph.pol[:]
            soltab_amp = sso.makeSoltab('amplitude', 'amplitude000',
                                        axesNames=['time', 'freq', 'ant', 'dir', 'pol'],
                                        axesVals=[times, freqs, ants, dirs, pols],
                                        vals=vals, weights=np.ones_like(vals))
        else:
            # Phase only, so just copy input soltab to new solset
            soltab.obj._f_copy_children(sso.obj, recursive=True, overwrite=True)
        ho.close()

        # Reweight the solutions by the scatter after detrending
        H_screen = h5parm(outh5parm)
        solset_screen = H.getSolset('sol000')
        soltab_phase = H.getSoltab('phase000')
        reweight.run(soltab_phase, mode='window', nmedian=3, nstddev=251)
        if 'amplitude' in soltab.getType():
            soltab_amp = H.getSoltab('amplitude000')
            reweight.run(soltab_amp, mode='window', nmedian=3, nstddev=21)

        # Now call LoSoTo's stationscreen operation to do the fitting
        stationscreen.run(soltab_phase, 'phase_screen000')
        if 'amplitude' in soltab.getType():
            stationscreen.run(soltab_amp, 'amplitude_screen000')

        # Transform the screens into FITS images


    elif screen_type == 'voronoi':
        # Do Voronoi tessellation + smoothing

        # Make blank output FITS file (type does not matter at this point)
        midRA = bounds_mid_deg[0]
        midDec = bounds_mid_deg[1]
        temp_image = outroot + '.tmp'
        imsize = (bounds_deg[3] - bounds_deg[1])  # deg
        imsize = int(imsize / cellsize_deg)  # pix
        misc.make_template_image(temp_image, midRA, midDec, ximsize=imsize,
                                 yimsize=imsize, cellsize_deg=cellsize_deg, freqs=freqs,
                                 times=[0.0], antennas=soltab.ant, aterm_type='tec')
        hdu = pyfits.open(temp_image, memmap=False)
        data = hdu[0].data
        w = wcs.WCS(hdu[0].header)
        RAind = w.axis_type_names.index('RA')
        Decind = w.axis_type_names.index('DEC')

        # Get x, y coords for directions in pixels. We use the input calibration sky
        # model for this, as the patch positions written to the h5parm file by DPPP may
        # be different
        skymod = lsmtool.load(skymodel)
        source_dict = skymod.getPatchPositions()
        source_positions = []
        for source in source_names:
            radecpos = source_dict[source.strip('[]')]
            source_positions.append([radecpos[0].value, radecpos[1].value])
        source_positions = np.array(source_positions)
        ra_deg = source_positions.T[0]
        dec_deg = source_positions.T[1]
        xy = []
        for RAvert, Decvert in zip(ra_deg, dec_deg):
            ra_dec = np.array([[0.0, 0.0, 0.0, 0.0, 0.0]])
            ra_dec[0][RAind] = RAvert
            ra_dec[0][Decind] = Decvert
            xy.append((w.wcs_world2pix(ra_dec, 0)[0][RAind], w.wcs_world2pix(ra_dec, 0)[0][Decind]))

        # Get boundary of tessellation region in pixels
        ra_dec = np.array([[0.0, 0.0, 0.0, 0.0, 0.0]])
        ra_dec[0][RAind] = max(bounds_deg[0], np.max(ra_deg))
        ra_dec[0][Decind] = min(bounds_deg[1], np.min(dec_deg))
        field_minxy = (w.wcs_world2pix(ra_dec, 0)[0][RAind], w.wcs_world2pix(ra_dec, 0)[0][Decind])
        ra_dec[0][RAind] = min(bounds_deg[2], np.min(ra_deg))
        ra_dec[0][Decind] = max(bounds_deg[3], np.max(dec_deg))
        field_maxxy = (w.wcs_world2pix(ra_dec, 0)[0][RAind], w.wcs_world2pix(ra_dec, 0)[0][Decind])

        if len(xy) == 1:
            # If there is only a single direction, just make a single rectangular polygon
            box = [field_minxy, (field_minxy[0], field_maxxy[1]), field_maxxy, (field_maxxy[0], field_minxy[1]), field_minxy]
            polygons = [shapely.geometry.Polygon(box)]
        else:
            # For more than one direction, tessellate
            # Generate array of outer points used to constrain the facets
            nouter = 64
            means = np.ones((nouter, 2)) * np.array(xy).mean(axis=0)
            offsets = []
            angles = [np.pi/(nouter/2.0)*i for i in range(0, nouter)]
            for ang in angles:
                offsets.append([np.cos(ang), np.sin(ang)])
            radius = 2.0*np.sqrt( (field_maxxy[0]-field_minxy[0])**2 + (field_maxxy[1]-field_minxy[1])**2 )
            scale_offsets = radius * np.array(offsets)
            outer_box = means + scale_offsets

            # Tessellate and clip
            points_all = np.vstack([xy, outer_box])
            vor = Voronoi(points_all)
            lines = [
                shapely.geometry.LineString(vor.vertices[line])
                for line in vor.ridge_vertices
                if -1 not in line
            ]
            polygons = [poly for poly in shapely.ops.polygonize(lines)]

        # Index polygons to directions
        ind = []
        for i, xypos in enumerate(xy):
            for poly in polygons:
                if poly.contains(Point(xypos)):
                    poly.index = i
    #     polygons = [poly for poly in polygons if hasattr(poly, 'index')]

        # Rasterize the polygons to an array, with the value being equal to the
        # polygon's index+1
        data_template = np.ones(data[0, 0, 0, :, :].shape)
        data_rasertize_template = np.zeros(data[0, 0, 0, :, :].shape)
        for poly in polygons:
            verts_xy = poly.exterior.xy
            verts = []
            for x, y in zip(verts_xy[0], verts_xy[1]):
                verts.append((x, y))
            poly_raster = misc.rasterize(verts, data_template.copy()) * (poly.index+1)
            filled = np.where(poly_raster > 0)
            data_rasertize_template[filled] = poly_raster[filled]
        zeroind = np.where(data_rasertize_template == 0)
        if len(zeroind[0]) > 0:
            nonzeroind = np.where(data_rasertize_template != 0)
            data_rasertize_template[zeroind] = si.griddata((nonzeroind[0], nonzeroind[1]), data_rasertize_template[nonzeroind],
                                                           (zeroind[0], zeroind[1]), method='nearest')

        # Identify any duplicate times and remove
        delta_times = times[1:] - times[:-1]  # time at center of solution interval
        nodupind = np.where(delta_times > 0.1)
        times = times[nodupind]
        if 'pol' in axis_names:
            vals = np.squeeze(vals[nodupind, :, :, :, :], axis=0)
        else:
            vals = np.squeeze(vals[nodupind, :, :, :], axis=0)
        vals_ph = np.squeeze(vals_ph[nodupind, :, :, :], axis=0)

        # Identify any gaps in time (frequency gaps are not allowed), as we need to
        # output a separate FITS file for each time chunk
        delta_times = times[1:] - times[:-1]  # time at center of solution interval
        timewidth = np.min(delta_times)
        gaps = np.where(delta_times > timewidth*1.2)
        gaps_ind = gaps[0] + 1
        gaps_ind = np.append(gaps_ind, np.array([len(times)]))

        # Add additional breaks to gaps_ind to keep memory use within that available
        # From experience, making a (30, 46, 62, 4, 146, 146) aterm image needs around
        # 30 GB of memory
        if soltab.getType() == 'tec':
            max_ntimes = 15 * 46 * 4
        else:
            max_ntimes = 15
        # TODO: adjust max_ntimes depending on available memory and time_avg_factor
        check_gaps = True
        while check_gaps:
            check_gaps = False
            g_start = 0
            gaps_ind_copy = gaps_ind.copy()
            for gnum, g_stop in enumerate(gaps_ind_copy):
                if g_stop - g_start > max_ntimes:
                    new_gap = g_start + int((g_stop - g_start) / 2)
                    gaps_ind = np.insert(gaps_ind, gnum, np.array([new_gap]))
                    check_gaps = True
                    break
                g_start = g_stop

        # Input data are [time, freq, ant, dir, pol] for slow amplitudes
        # and [time, freq, ant, dir] for fast phases (scalarphase).
        # Output data are [RA, DEC, MATRIX, ANTENNA, FREQ, TIME].T
        # Loop over stations, frequencies, and times and fill in the correct
        # matrix values (matrix dimension has 4 elements: real XX, imaginary XX,
        # real YY and imaginary YY)
        outfiles = []
        g_start = 0
        for gnum, g_stop in enumerate(gaps_ind):
            outfile = '{0}_{1}.fits'.format(outroot, gnum)
            misc.make_template_image(temp_image, midRA, midDec, ximsize=imsize,
                                     yimsize=imsize, cellsize_deg=cellsize_deg,
                                     times=times[g_start:g_stop],
                                     freqs=freqs, antennas=soltab.ant,
                                     aterm_type='gain')
            hdu = pyfits.open(temp_image, memmap=False)
            data = hdu[0].data
            w = wcs.WCS(hdu[0].header)
            for t, time in enumerate(times[g_start:g_stop]):
                for f, freq in enumerate(freqs):
                    for s, stat in enumerate(ants):
                        for p, poly in enumerate(polygons):
                            ind = np.where(data_rasertize_template == poly.index+1)
                            if 'pol' in axis_names:
                                val_amp_xx = vals[t+g_start, f, s, poly.index, 0]
                                val_amp_yy = vals[t+g_start, f, s, poly.index, 1]
                            else:
                                val_amp_xx = vals[t+g_start, f, s, poly.index]
                                val_amp_yy = val_amp_xx
                            val_phase = vals_ph[t+g_start, f, s, poly.index]
                            data[t, f, s, 0, ind[0], ind[1]] = val_amp_xx * np.cos(val_phase)
                            data[t, f, s, 2, ind[0], ind[1]] = val_amp_yy * np.cos(val_phase)
                            data[t, f, s, 1, ind[0], ind[1]] = val_amp_xx * np.sin(val_phase)
                            data[t, f, s, 3, ind[0], ind[1]] = val_amp_yy * np.sin(val_phase)

                        # Smooth if desired
                        if smooth_pix > 0:
                            data[t, f, s, :, :, :] = ndimage.gaussian_filter(data[t, f, s, :, :, :],
                                                                             sigma=(0, smooth_pix,
                                                                                    smooth_pix),
                                                                             order=0)

            # If averaging in time, make a new template image with
            # fewer times and write to that instead
            if time_avg_factor > 1:
                times_avg = times[g_start:g_stop:time_avg_factor]
                ntimes = len(times_avg)
                misc.make_template_image(temp_image+'.avg', midRA, midDec, ximsize=imsize,
                                         yimsize=imsize, cellsize_deg=cellsize_deg,
                                         times=times_avg,
                                         freqs=freqs, antennas=soltab.ant,
                                         aterm_type='gain')
                hdu = pyfits.open(temp_image+'.avg', memmap=False)
                data_avg = hdu[0].data

                # Average
                for t, time in enumerate(times_avg):
                    incr = min(time_avg_factor, len(times[g_start:g_stop])-t*time_avg_factor)
                    data_avg[t, :, :, :, :, :] = np.nanmean(data[t:t+incr, :, :, :, :, :], axis=0)
                data = data_avg
            else:
                ntimes = len(times[g_start:g_stop])

            # Ensure there are no NaNs in the images, as WSClean will produced uncorrected,
            # uncleaned images if so. We replace NaNs with 1.0 and 0.0 for real and
            # imaginary parts, respectively
            # Note: we iterate over time to reduce memory usage
            for t in range(ntimes):
                for p in range(4):
                    if p % 2:
                        # Imaginary elements
                        nanval = 0.0
                    else:
                        # Real elements
                        nanval = 1.0
                    data[t, :, :, p, :, :][np.isnan(data[t, :, :, p, :, :])] = nanval

            # Write FITS file
            hdu[0].data = data
            hdu.writeto(outfile, overwrite=True)
            outfiles.append(outfile)
            os.remove(temp_image)
            hdu = None
            data = None

            # Update start time index
            g_start = g_stop

            outfile = open(outroot+'.txt', 'w')
            outfile.writelines([o+'\n' for o in outfiles])
            outfile.close()


if __name__ == '__main__':
    descriptiontext = "Make a-term images from solutions.\n"

    parser = argparse.ArgumentParser(description=descriptiontext, formatter_class=RawTextHelpFormatter)
    parser.add_argument('h5parmfile', help='Filename of input h5parm')
    parser.add_argument('--soltabname', help='Name of soltab', type=str, default='phase000')
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
    main(args.h5parmfile, soltabname=args.soltabname, outroot=args.outroot,
         bounds_deg=args.bounds_deg, bounds_mid_deg=args.bounds_mid_deg,
         skymodel=args.skymodel, solsetname=args.solsetname,
         padding_fraction=args.padding_fraction,
         cellsize_deg=args.cellsize_deg, smooth_deg=args.smooth_deg,
         time_avg_factor=args.time_avg_factor)
