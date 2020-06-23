#! /usr/bin/env python3
"""
Script to process gain solutions
"""
import argparse
from argparse import RawTextHelpFormatter
from losoto.h5parm import h5parm
import numpy as np
import scipy.interpolate as si
from rapthor.lib import miscellaneous as misc


def normalize_full(soltab):
    """
    Normalize values so that mean is equal to unity for amplitudes and zero for phases

    Parameters
    ----------
    soltab : solution table
        Input table with solutions

    Returns
    -------
    parms, weights : arrays
        The normalized parameters and weights
    """
    soltype = soltab.getType()
    parms = soltab.val[:]  # ['time', 'freq', 'ant', 'dir', 'pol']
    weights = soltab.weight[:]
    initial_flagged_indx = np.logical_or(np.isnan(parms), weights == 0.0)
    initial_unflagged_indx = np.logical_and(~np.isnan(parms), weights != 0.0)
    parms[initial_flagged_indx] = np.nan

    # Normalize each station, time, and frequency separately to unity, but
    # maintain the relative offset between polarizations. Note that we work in log space
    # for these operations if the solutions are amplitudes
    if soltype == 'amplitude':
        parms = np.log10(parms)
    for dir in range(len(soltab.dir[:])):
        for s in range(len(soltab.ant[:])):
            for t in range(len(soltab.time[:])):
                for f in range(len(soltab.freq[:])):
                    norm_rapthor = np.nanmean(parms[t, f, s, dir, :][initial_unflagged_indx[t, f, s, dir, :]])
                    parms[t, f, s, dir, :] -= norm_rapthor

    # Average the XX-YY offsets over all stations and directions for each time, freq and pol
    # Only do this for phases, as we do it for amplitudes in the bandpass correction
    if soltype == 'phase':
        for t in range(len(soltab.time[:])):
            for f in range(len(soltab.freq[:])):
                for p in range(2):
                    parms[t, f, :, :, p] = np.nanmedian(parms[t, f, :, :, p])

    # Make sure flagged solutions are still flagged
    if soltype == 'amplitude':
        parms = 10**parms
    parms[initial_flagged_indx] = np.nan
    weights[initial_flagged_indx] = 0.0

    return parms, weights


def normalize_station(soltab):
    """
    Normalize values so that mean is equal to unity for amplitudes per direction per
    station

    Parameters
    ----------
    soltab : solution table
        Input table with solutions

    Returns
    -------
    parms, weights : arrays
        The normalized parameters and weights
    """
    soltype = soltab.getType()
    parms = soltab.val[:]  # ['time', 'freq', 'ant', 'dir', 'pol']
    weights = soltab.weight[:]
    initial_flagged_indx = np.logical_or(np.isnan(parms), weights == 0.0)
    initial_unflagged_indx = np.logical_and(~np.isnan(parms), weights != 0.0)
    parms[initial_flagged_indx] = np.nan

    # Normalize each station per direction separately to have a mean of unity over all
    # times, frequencies, and pols. Note that we work in log space
    # for these operations if the solutions are amplitudes
    if soltype == 'amplitude':
        parms = np.log10(parms)
    for dir in range(len(soltab.dir[:])):
        for s in range(len(soltab.ant[:])):
            norm_rapthor = np.nanmean(parms[:, :, s, dir, :][initial_unflagged_indx[:, :, s, dir, :]])
            parms[:, :, s, dir, :] -= norm_rapthor

    # Make sure flagged solutions are still flagged
    if soltype == 'amplitude':
        parms = 10**parms
    parms[initial_flagged_indx] = np.nan
    weights[initial_flagged_indx] = 0.0

    return parms, weights


def normalize_direction(soltab):
    """
    Normalize values so that mean is equal to unity for amplitudes per direction

    Parameters
    ----------
    soltab : solution table
        Input table with solutions

    Returns
    -------
    parms, weights : arrays
        The normalized parameters and weights
    """
    soltype = soltab.getType()
    parms = soltab.val[:]  # ['time', 'freq', 'ant', 'dir', 'pol']
    weights = soltab.weight[:]
    initial_flagged_indx = np.logical_or(np.isnan(parms), weights == 0.0)
    initial_unflagged_indx = np.logical_and(~np.isnan(parms), weights != 0.0)
    parms[initial_flagged_indx] = np.nan

    # Normalize each station per direction separately to have a mean of unity over all
    # times, frequencies, and pols. Note that we work in log space
    # for these operations if the solutions are amplitudes
    if soltype == 'amplitude':
        parms = np.log10(parms)
    for dir in range(len(soltab.dir[:])):
        norm_rapthor = np.nanmean(parms[:, :, :, dir, :][initial_unflagged_indx[:, :, :, dir, :]])
        parms[:, :, :, dir, :] -= norm_rapthor

    # Make sure flagged solutions are still flagged
    if soltype == 'amplitude':
        parms = 10**parms
    parms[initial_flagged_indx] = np.nan
    weights[initial_flagged_indx] = 0.0

    return parms, weights


def normalize_field(soltab):
    """
    Normalize values so that mean is equal to unity for amplitudes over entire field

    Parameters
    ----------
    soltab : solution table
        Input table with solutions

    Returns
    -------
    parms, weights : arrays
        The normalized parameters and weights
    """
    soltype = soltab.getType()
    parms = soltab.val[:]  # ['time', 'freq', 'ant', 'dir', 'pol']
    weights = soltab.weight[:]
    initial_flagged_indx = np.logical_or(np.isnan(parms), weights == 0.0)
    initial_unflagged_indx = np.logical_and(~np.isnan(parms), weights != 0.0)
    parms[initial_flagged_indx] = np.nan

    # Normalize everything to have a mean of unity over all
    # times, frequencies, directions, and pols. Note that we work in log space
    # for these operations if the solutions are amplitudes
    if soltype == 'amplitude':
        parms = np.log10(parms)
        norm_rapthor = np.nanmean(parms[initial_unflagged_indx])
        parms -= norm_rapthor

    # Make sure flagged solutions are still flagged
    if soltype == 'amplitude':
        parms = 10**parms
    parms[initial_flagged_indx] = np.nan
    weights[initial_flagged_indx] = 0.0

    return parms, weights


def find_bandpass_correction(soltab, parms_normalized):
    """
    Find the correction to the bandpass for each station

    Parameters
    ----------
    soltab : solution table
        Input table with solutions
    parms_normalized : array
        The normalized parameters

    Returns
    -------
    parms, weights : arrays
        The parameters with bandpass corrections and weights
    """
    from loess import loess_2d, loess_1d

    parms = soltab.val[:]  # ['time', 'freq', 'ant', 'dir', 'pol']
    parms /= parms_normalized  # divide out XX-YY offsets
    weights = soltab.weight[:]
    initial_flagged_indx = np.logical_or(np.isnan(parms), weights == 0.0)
    initial_unflagged_indx = np.logical_and(~np.isnan(parms), weights != 0.0)
    parms[initial_flagged_indx] = np.nan

    # Normalize each direction, time, and station to have a mean of 1 over all frequencies
    # and polarizations. Note that we work in log space for these operations
    parms = np.log10(parms)
    for dir in range(len(soltab.dir[:])):
        for s in range(len(soltab.ant[:])):
            for t in range(len(soltab.time[:])):
                norm_rapthor = np.nanmean(parms[t, :, s, dir, :][initial_unflagged_indx[t, :, s, dir, :]])
                parms[t, :, s, dir, :] -= norm_rapthor

    # Take the median over all directions, times, and pols for each station at each freq
    # to reduce the noise on the bandpass correction, then smooth in frequency with LOESS.
    for s in range(len(soltab.ant[:])):
        for f in range(len(soltab.freq[:])):
            parms[:, f, s, :, :] = np.nanmedian(parms[:, f, s, :, :])
        nanind = np.where(~np.isnan(parms[0, :, s, 0, 0]))
        fs, ps, ws = loess_1d.loess_1d(soltab.freq[nanind]/100e8, parms[0, :, s, 0, 0][nanind], frac=0.5, degree=2)
        parms[0, :, s, 0, 0][nanind] = ps
    parms = 10**parms

    # Average the XX-YY offsets over all stations and directions for each time, freq and pol
    parms_normalized = np.log10(parms_normalized)
    for t in range(len(soltab.time[:])):
        for f in range(len(soltab.freq[:])):
            for p in range(2):
                parms_normalized[t, f, :, :, p] = np.nanmedian(parms_normalized[t, f, :, :, p])
    parms_normalized = 10**parms_normalized
    parms *= parms_normalized  # add in averaged XX-YY offsets

    # Make sure flagged solutions are still flagged
    parms[initial_flagged_indx] = np.nan
    weights[initial_flagged_indx] = 0.0

    return parms, weights


def smooth(soltab, stddev_threshold=0.25, freq_sampling=5, time_sampling=2, ref_id=0,
           debug=False):
    """
    Smooth scalarphases. The smoothing is done in real and imaginary space

    Parameters
    ----------
    soltab : solution table
        Input table with solutions
    stddev_threshold : float, optional
        The threshold stddev below which no smoothing is done
    freq_sampling : int, optoinal
        Sampling stride to use for frequency when doing LOESS smooth
    time_sampling : int, optoinal
        Sampling stride to use for time when doing LOESS smooth
    ref_id : int, optional
        Index of reference station

    Returns
    -------
    parms, weights : arrays
        The parameters with bandpass corrections and weights
    """
    from loess import loess_2d, loess_1d

    parms = soltab.val[:]  # ['time', 'freq', 'ant', 'dir']
    parms_ref = parms[:, :, ref_id, :].copy()
    for i in range(len(soltab.ant)):
        parms[:, :, i, :] -= parms_ref
    weights = soltab.weight[:]
    initial_flagged_indx = np.logical_or(np.isnan(parms), weights == 0.0)

    # Find gaps in time and treat each block separately
    times = soltab.time[:]
    delta_times = times[1:] - times[:-1]  # time at center of solution interval
    timewidth = np.min(delta_times)
    gaps = np.where(delta_times > timewidth*1.2)
    gaps_ind = gaps[0] + 1
    gaps_ind = np.append(gaps_ind, np.array([len(times)]))

    for dir in range(len(soltab.dir[:])):
        # Find standard deviation of the real part of a core station and determine
        # whether we need to smooth this direction or not
        csindx = 2  # should always be a core station
        sdev = np.std(np.cos(parms[:, :, csindx, dir]))
        if sdev >= stddev_threshold:
            for s in range(len(soltab.ant[:])):
                # For core stations, just set phase to 0
                if 'CS' in soltab.ant[s]:
                    parms[:, :, s, dir] = 0.0
                    continue

                # For remote stations, set smoothing parameter (frac) depending on sdev
                frac = min(0.8, 0.3 * sdev / stddev_threshold)
                g_start = 0
                for gnum, g_stop in enumerate(gaps_ind):
                    # Define slices for frequency and time sampling
                    freq_slice = slice(0, -1, freq_sampling)
                    time_slice = slice(g_start, g_stop, time_sampling)

                    # Do the smoothing with LOESS
                    yv, xv = np.meshgrid(soltab.freq[freq_slice], times[time_slice])
                    yv /= np.min(yv)
                    xv -= np.min(xv)
                    zreal = np.cos(parms[time_slice, freq_slice, s, dir])
                    nanind = np.where(~np.isnan(zreal))
                    if len(nanind[0]) == 0:
                        g_start = g_stop
                        continue
                    zsreal, wreal = loess_2d.loess_2d(xv[nanind].flatten(),
                                                      yv[nanind].flatten(),
                                                      zreal[nanind].flatten(),
                                                      rescale=True, frac=frac, degree=2)
                    zimag = np.sin(parms[time_slice, freq_slice, s, dir])
                    zsimag, wimag = loess_2d.loess_2d(xv[nanind].flatten(),
                                                      yv[nanind].flatten(),
                                                      zimag[nanind].flatten(),
                                                      rescale=True, frac=frac, degree=2)

                    if debug:
                        from plotbin.plot_velfield import plot_velfield
                        import matplotlib.pyplot as plt
                        plt.clf()
                        plt.subplot(121)
                        plot_velfield(xv[nanind].flatten(), yv[nanind].flatten()*1000, zreal[nanind].flatten(), vmin=-1.0, vmax=1.0)
                        plt.title("Input Real Values")
                        plt.subplot(122)
                        plot_velfield(xv[nanind].flatten(), yv[nanind].flatten()*1000, zsreal, vmin=-1.0, vmax=1.0)
                        plt.title("LOESS Recovery")
                        plt.tick_params(labelleft=False)
                        plt.show()

                    # Interpolate back to original grid
                    zr = zsreal.reshape((len(times[time_slice][np.array(list(set(nanind[0])))]), len(soltab.freq[freq_slice][np.array(list(set(nanind[1])))])))
                    f = si.interp1d(times[time_slice][np.array(list(set(nanind[0])))], zr, axis=0, kind='linear', fill_value='extrapolate')
                    zr1 = f(times[g_start:g_stop])
                    f = si.interp1d(soltab.freq[freq_slice][np.array(list(set(nanind[1])))], zr1, axis=1, kind='linear', fill_value='extrapolate')
                    zr = f(soltab.freq)
                    zi = zsimag.reshape((len(times[time_slice][np.array(list(set(nanind[0])))]), len(soltab.freq[freq_slice][np.array(list(set(nanind[1])))])))
                    f = si.interp1d(times[time_slice][np.array(list(set(nanind[0])))], zi, axis=0, kind='linear', fill_value='extrapolate')
                    zi1 = f(times[g_start:g_stop])
                    f = si.interp1d(soltab.freq[freq_slice][np.array(list(set(nanind[1])))], zi1, axis=1, kind='linear', fill_value='extrapolate')
                    zi = f(soltab.freq)
                    parms[g_start:g_stop, :, s, dir] = np.arctan2(zi, zr)
                    g_start = g_stop

    # Make sure flagged solutions are still flagged
    parms[initial_flagged_indx] = np.nan
    weights[initial_flagged_indx] = 0.0

    return parms, weights


def remove_soltabs(solset, soltabnames):
    """
    Remove soltab
    """
    for soltabname in soltabnames:
        try:
            soltab = solset.getSoltab(soltabname)
            soltab.delete()
        except:
            pass


def main(h5parmfile, solsetname='sol000', ampsoltabname=None,
         phsoltabname=None, ref_id=0, smooth_phases=False,
         normalize=False, find_bandpass=False):
    """
    Fit screens to gain solutions

    Parameters
    ----------
    h5parmfile : str
        Filename of h5parm
    solsetname : str, optional
        Name of solset
    ampsoltabname : str, optional
        Name of TEC soltab
    phsoltabname : str, optional
        Name of error soltab
    ref_id : int, optional
        Index of reference station
    smooth_phases : bool, optional
        Smooth phase solutions
    normalize : bool, optional
        Normalize amp solutions
    find_bandpass : bool, optional
        Find and apply bandpass corrections
    """
    ref_id = int(ref_id)
    normalize = misc.string2bool(normalize)
    smooth_phases = misc.string2bool(smooth_phases)
    find_bandpass = misc.string2bool(find_bandpass)

    # Read in solutions
    H = h5parm(h5parmfile, readonly=False)
    solset = H.getSolset(solsetname)
    if ampsoltabname is not None:
        ampsoltab = solset.getSoltab(ampsoltabname)
        amp = np.array(ampsoltab.val)
        damp = np.ones(amp.shape)

        if ampsoltabname != 'origamplitude000':
            ampsoltab.rename('origamplitude000', overwrite=True)
        if normalize:
#             amp, damp = normalize_full(ampsoltab)
#             amp, damp = normalize_station(ampsoltab)
            amp, damp = normalize_direction(ampsoltab)
            if find_bandpass:
                amp, damp = find_bandpass_correction(ampsoltab, amp)
        remove_soltabs(solset, 'amplitude000')
        solset.makeSoltab('amplitude', 'amplitude000', axesNames=['time', 'freq', 'ant', 'dir', 'pol'],
                          axesVals=[ampsoltab.time[:], ampsoltab.freq[:], ampsoltab.ant[:],
                          ampsoltab.dir[:], ampsoltab.pol[:]], vals=amp, weights=damp)

    if phsoltabname is not None:
        phsoltab = solset.getSoltab(phsoltabname)
        ph = np.array(phsoltab.val)
        dph = np.ones(ph.shape)
        axis_names = phsoltab.getAxesNames()

        # Identify any duplicate times and remove
        times = phsoltab.time[:]
        delta_times = times[1:] - times[:-1]
        nodupind = np.where(delta_times > 0.1)
        if len(nodupind[0]) < len(times)-1:
            times = times[nodupind]
            if 'pol' in axis_names:
                ph = np.squeeze(ph[nodupind, :, :, :, :])
                dph = np.squeeze(dph[nodupind, :, :, :, :])
            else:
                ph = np.squeeze(ph[nodupind, :, :, :])
                dph = np.squeeze(dph[nodupind, :, :, :])
            if phsoltabname != 'origphase000':
                phsoltab.rename('origphase000', overwrite=True)
            if 'pol' in axis_names:
                solset.makeSoltab('phase', 'phase000', axesNames=['time', 'freq', 'ant', 'dir', 'pol'],
                                  axesVals=[times, phsoltab.freq[:], phsoltab.ant[:],
                                  phsoltab.dir[:], phsoltab.pol[:]], vals=ph, weights=dph)
            else:
                solset.makeSoltab('phase', 'phase000', axesNames=['time', 'freq', 'ant', 'dir'],
                                  axesVals=[times, phsoltab.freq[:], phsoltab.ant[:],
                                  phsoltab.dir[:]], vals=ph, weights=dph)
            phsoltab = solset.getSoltab('phase000')
            ph = np.array(phsoltab.val)
            dph = np.ones(ph.shape)

        if phsoltabname != 'origphase000':
            phsoltab.rename('origphase000', overwrite=True)
        if smooth_phases:
            ph, dph = smooth(phsoltab, ref_id=ref_id)
        if normalize:
#             ph, dph = normalize_full(phsoltab)
            pass
        remove_soltabs(solset, 'phase000')
        if 'pol' in axis_names:
            solset.makeSoltab('phase', 'phase000', axesNames=['time', 'freq', 'ant', 'dir', 'pol'],
                              axesVals=[times, phsoltab.freq[:], phsoltab.ant[:],
                              phsoltab.dir[:], phsoltab.pol[:]], vals=ph, weights=dph)
        else:
            solset.makeSoltab('phase', 'phase000', axesNames=['time', 'freq', 'ant', 'dir'],
                              axesVals=[times, phsoltab.freq[:], phsoltab.ant[:],
                              phsoltab.dir[:]], vals=ph, weights=dph)
    H.close()


if __name__ == '__main__':
    descriptiontext = "Process gain solutions.\n"

    parser = argparse.ArgumentParser(description=descriptiontext, formatter_class=RawTextHelpFormatter)
    parser.add_argument('h5parmfile', help='Filename of input h5parm')
    parser.add_argument('--solsetname', help='Solset name', type=str, default='sol000')
    parser.add_argument('--ampsoltabname', help='Amplitude soltab name', type=str, default=None)
    parser.add_argument('--phsoltabname', help='Phase soltab name', type=str, default=None)
    parser.add_argument('--ref_id', help='Reference station', type=int, default=0)
    parser.add_argument('--smooth_phases', help='Smooth phase solutions', type=str, default='False')
    parser.add_argument('--normalize', help='Normalize amplitude solutions', type=str, default='False')
    parser.add_argument('--find_bandpass', help='Find bandpass corrections', type=str, default='False')
    args = parser.parse_args()
    main(args.h5parmfile, solsetname=args.solsetname, ampsoltabname=args.ampsoltabname,
         phsoltabname=args.phsoltabname, ref_id=args.ref_id,
         smooth_phases=args.smooth_phases, normalize=args.normalize,
         find_bandpass=args.find_bandpass)
