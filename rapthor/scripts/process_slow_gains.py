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
from scipy.optimize import curve_fit


def get_ant_dist(ant_xyz, ref_xyz):
    """
    Returns distance between ant and ref in m

     Parameters
    ----------
    ant_xyz : array
        Array of station position
    ref_xyz : array
        Array of reference position

    Returns
    -------
    dist : float
        Distance between station and reference positions

    """
    import numpy as np

    return np.sqrt((ref_xyz[0] - ant_xyz[0])**2 + (ref_xyz[1] - ant_xyz[1])**2 + (ref_xyz[2] - ant_xyz[2])**2)


def func(x, m, c):
    return m * x + c


def normalize_direction(soltab, remove_core_gradient=True, solset=None, ref_id=0,
                        parms=None, weights=None):
    """
    Normalize amplitudes so that mean is equal to unity, per direction

    Parameters
    ----------
    soltab : solution table
        Input table with solutions
    remove_core_gradient : bool, optional
        If True, remove any gradient with distance from the core stations
    solset : solution set, optional
        Input set, needed if remove_core_gradient is True
    ref_id : int, optional
        Index of reference station, needed if remove_core_gradient is True

    Returns
    -------
    parms, weights : arrays
        The normalized parameters and weights
    """
    if parms is None:
        parms = soltab.val[:]  # axes are ['time', 'freq', 'ant', 'dir', 'pol']
    if weights is None:
        weights = soltab.weight[:]
    initial_flagged_indx = np.logical_or(np.isnan(parms), weights == 0.0)
    initial_unflagged_indx = np.logical_and(~np.isnan(parms), weights != 0.0)
    parms[initial_flagged_indx] = np.nan

    # Work in log space, as required for amplitudes
    parms = np.log10(parms)

    # Find and remove any gradient for each direction separately
    if remove_core_gradient:
        dist = []
        station_names = soltab.ant[:]
        if type(station_names) is not list:
            station_names = station_names.tolist()
        station_dict = solset.getAnt()
        station_positions = []
        for station in station_names:
            station_positions.append(station_dict[station.encode()])
        for s in range(len(station_names)):
            if s == ref_id:
                dist.append(1.0)
            else:
                dist.append(get_ant_dist(station_positions[s], station_positions[ref_id]))
        for dir in range(len(soltab.dir[:])):
            dist_vals = []
            mean_vals = []
            stat_names = []
            for s in range(len(station_names)):
                if 'CS' in station_names[s] and s != ref_id:
                    if not np.all(np.isnan(parms[:, :, s, dir, :])):
                        mean_vals.append(np.nanmean(parms[:, :, s, dir, :]))
                        dist_vals.append(dist[s])
                        stat_names.append(station_names[s])

            # Find best-fit gradient for core only
            x = np.log10(np.array(dist_vals))
            y = np.array(mean_vals)
            w = np.ones_like(y)
            popt, pcov = curve_fit(func, x, y, sigma=w, p0=[0.0, 1.0])

            # Divide out the gradient, assuming the values at the largest distances
            # are more likely correct (and so should be around 1.0, after normalization)
            for s in range(len(station_names)):
                if 'CS' in station_names[s]:
                    if s == ref_id:
                        # For the reference station, take as the distance that of a
                        # neighboring station, to avoid large extrapolations to zero
                        # distance
                        parms[:, :, s, dir, :] -= popt[0]*np.log10(dist[s+1]) + popt[1] - (popt[0]*np.log10(np.max(dist_vals)) + popt[1])
                    else:
                        parms[:, :, s, dir, :] -= popt[0]*np.log10(dist[s]) + popt[1] - (popt[0]*np.log10(np.max(dist_vals)) + popt[1])

    # Normalize each direction separately to have a mean of unity over all
    # times, frequencies, and pols
    for dir in range(len(soltab.dir[:])):
        norm_factor = np.nanmean(parms[:, :, :, dir, :][initial_unflagged_indx[:, :, :, dir, :]])
        parms[:, :, :, dir, :] -= norm_factor

    # Convert back to non-log values and make sure flagged solutions are still flagged
    parms = 10**parms
    parms[initial_flagged_indx] = np.nan
    weights[initial_flagged_indx] = 0.0

    return parms, weights


def smooth_amps(soltab, stddev_threshold=0.1, freq_sampling=1, time_sampling=1,
                smooth_over_gaps=True, parms=None, weights=None, debug=False):
    """
    Smooth amplitudes. The smoothing is done in log space

    Parameters
    ----------
    soltab : solution table
        Input table with solutions
    stddev_threshold : float, optional
        The threshold stddev below which no smoothing is done
    freq_sampling : int, optional
        Sampling stride to use for frequency when doing LOESS smooth
    time_sampling : int, optional
        Sampling stride to use for time when doing LOESS smooth
    smooth_over_gaps : bool, optional
        If True, ignore gaps in time when smoothing

    Returns
    -------
    parms, weights : arrays
        The smoothed parameters and associated weights
    """
    from loess import loess_2d

    # Work in log space, as required for amplitudes
    if parms is None:
        parms = soltab.val[:]  # axes are ['time', 'freq', 'ant', 'dir', 'pol']
    parms = np.log10(parms)
    if weights is None:
        weights = soltab.weight[:]
    initial_flagged_indx = np.logical_or(np.isnan(parms), weights == 0.0)
    times = soltab.time[:]
    if smooth_over_gaps:
        # Ignore any gaps in time
        gaps_ind = [soltab.time.shape[0]]
    else:
        # Find gaps in time and treat each block separately
        delta_times = times[1:] - times[:-1]  # time at center of solution interval
        timewidth = np.min(delta_times)
        gaps = np.where(delta_times > timewidth*1.2)
        gaps_ind = gaps[0] + 1
        gaps_ind = np.append(gaps_ind, np.array([len(times)]))

    # Find a core station that is not completely flagged
    for s in range(len(soltab.ant[:])):
        if 'CS' in soltab.ant[s]:
            if not np.all(initial_flagged_indx[:, :, s, :, :]):
                csindx = s
                break

    for dir in range(len(soltab.dir[:])):
        # Find standard deviation of a core station and determine
        # whether we need to smooth this direction or not
        sdev = np.std(parms[:, :, csindx, dir])
        if sdev >= stddev_threshold:
            for s in range(len(soltab.ant[:])):
                # Set smoothing parameter (frac) depending on sdev, smoothing more
                # when the sdev is higher. Allow more smoothing on the core
                # stations than the remote ones (which tend to be less noisy and
                # show more structure)
                if 'CS' in soltab.ant[s]:
                    frac = min(0.5, 0.2 * sdev / stddev_threshold)
                else:
                    frac = min(0.4, 0.1 * sdev / stddev_threshold)
                g_start = 0
                for gnum, g_stop in enumerate(gaps_ind):
                    # Define slices for frequency and time sampling
                    freq_slice = slice(0, soltab.freq.shape[0], freq_sampling)
                    time_slice = slice(g_start, g_stop, time_sampling)

                    # Do the smoothing with LOESS
                    for pol in [0, 1]:
                        yv, xv = np.meshgrid(soltab.freq[freq_slice], times[time_slice])
                        yv /= np.min(yv)
                        xv -= np.min(xv)
                        z = parms[time_slice, freq_slice, s, dir, pol]
                        nanind = np.where(~np.isnan(z))
                        if len(nanind[0]) == 0:
                            # All solutions are flagged, so skip processing
                            g_start = g_stop
                            continue
                        zs, w = loess_2d.loess_2d(xv[nanind].flatten(),
                                                  yv[nanind].flatten(),
                                                  z[nanind].flatten(),
                                                  rescale=True, frac=frac, degree=1)

                        if debug:
                            from plotbin.plot_velfield import plot_velfield
                            import matplotlib.pyplot as plt
                            plt.clf()
                            plt.subplot(121)
                            plot_velfield(xv[nanind].flatten(), yv[nanind].flatten()*1000, z[nanind].flatten(), vmin=-1.0, vmax=1.0)
                            plt.title("Input Values")
                            plt.subplot(122)
                            plot_velfield(xv[nanind].flatten(), yv[nanind].flatten()*1000, zs, vmin=-1.0, vmax=1.0)
                            plt.title("LOESS Recovery")
                            plt.tick_params(labelleft=False)
                            plt.show()

                        # Interpolate back to original grid
                        zr = zs.reshape((len(times[time_slice][np.array(list(set(nanind[0])))]), len(soltab.freq[freq_slice][np.array(list(set(nanind[1])))])))
                        f = si.interp1d(times[time_slice][np.array(list(set(nanind[0])))], zr, axis=0, kind='linear', fill_value='extrapolate')
                        zr1 = f(times[g_start:g_stop])
                        f = si.interp1d(soltab.freq[freq_slice][np.array(list(set(nanind[1])))], zr1, axis=1, kind='linear', fill_value='extrapolate')
                        zr = f(soltab.freq)
                        parms[time_slice, freq_slice, s, dir, pol] = zr
                    g_start = g_stop

    # Convert back to non-log values and make sure flagged solutions are still flagged
    parms = 10**parms
    parms[initial_flagged_indx] = np.nan
    weights[initial_flagged_indx] = 0.0

    return parms, weights


def smooth_phases(soltab, stddev_threshold=0.1, freq_sampling=1, time_sampling=1,
                  ref_id=0, smooth_over_gaps=True, parms=None, weights=None, debug=False):
    """
    Smooth phases. The smoothing is done in real/imag space

    Parameters
    ----------
    soltab : solution table
        Input table with solutions
    stddev_threshold : float, optional
        The threshold stddev below which no smoothing is done
    freq_sampling : int, optional
        Sampling stride to use for frequency when doing LOESS smooth
    time_sampling : int, optional
        Sampling stride to use for time when doing LOESS smooth
    ref_id : int, optional
        Index of reference station
    smooth_over_gaps : bool, optional
        If True, ignore gaps in time when smoothing

    Returns
    -------
    parms, weights : arrays
        The smoothed parameters and associated weights
    """
    from loess import loess_2d

    # Work in real/image space, as required for phases
    if parms is None:
        parms = soltab.val[:]  # axes are ['time', 'freq', 'ant', 'dir', 'pol']
    parms_ref = parms[:, :, ref_id, :, :].copy()
    for i in range(len(soltab.ant)):
        parms[:, :, i, :, :] -= parms_ref
    if weights is None:
        weights = soltab.weight[:]
    initial_flagged_indx = np.logical_or(np.isnan(parms), weights == 0.0)
    times = soltab.time[:]
    if smooth_over_gaps:
        # Ignore any gaps in time
        gaps_ind = [soltab.time.shape[0]]
    else:
        # Find gaps in time and treat each block separately
        delta_times = times[1:] - times[:-1]  # time at center of solution interval
        timewidth = np.min(delta_times)
        gaps = np.where(delta_times > timewidth*1.2)
        gaps_ind = gaps[0] + 1
        gaps_ind = np.append(gaps_ind, np.array([len(times)]))

    # Find a core station that is not completely flagged
    for s in range(len(soltab.ant[:])):
        if s == ref_id:
            continue
        if 'CS' in soltab.ant[s]:
            if not np.all(initial_flagged_indx[:, :, s, :, :]):
                csindx = s
                break

    for dir in range(len(soltab.dir[:])):
        # Find standard deviation of the real part of a a core station and determine
        # whether we need to smooth this direction or not
        sdev = np.std(np.cos(parms[:, :, csindx, dir]))
        if sdev >= stddev_threshold:
            for s in range(len(soltab.ant[:])):
                if s == ref_id:
                    continue

                # Set smoothing parameter (frac) depending on sdev, smoothing more
                # when the sdev is higher
                frac = min(0.5, 0.1 * sdev / stddev_threshold)
                g_start = 0
                for gnum, g_stop in enumerate(gaps_ind):
                    # Define slices for frequency and time sampling
                    freq_slice = slice(0, soltab.freq.shape[0], freq_sampling)
                    time_slice = slice(g_start, g_stop, time_sampling)

                    # Do the smoothing with LOESS
                    for pol in [0, 1]:
                        yv, xv = np.meshgrid(soltab.freq[freq_slice], times[time_slice])
                        yv /= np.min(yv)
                        xv -= np.min(xv)
                        zreal = np.cos(parms[time_slice, freq_slice, s, dir, pol])
                        nanind = np.where(~np.isnan(zreal))
                        if len(nanind[0]) == 0:
                            # All solutions are flagged, so skip processing
                            g_start = g_stop
                            continue
                        zsreal, wreal = loess_2d.loess_2d(xv[nanind].flatten(),
                                                          yv[nanind].flatten(),
                                                          zreal[nanind].flatten(),
                                                          rescale=True, frac=frac, degree=1)
                        zimag = np.sin(parms[time_slice, freq_slice, s, dir, pol])
                        zsimag, wimag = loess_2d.loess_2d(xv[nanind].flatten(),
                                                          yv[nanind].flatten(),
                                                          zimag[nanind].flatten(),
                                                          rescale=True, frac=frac, degree=1)

                        if debug:
                            from plotbin.plot_velfield import plot_velfield
                            import matplotlib.pyplot as plt
                            plt.clf()
                            plt.subplot(121)
                            plot_velfield(xv[nanind].flatten(), yv[nanind].flatten()*1000, zreal[nanind].flatten(), vmin=-1.0, vmax=1.0)
                            plt.title("Input Values")
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
                        parms[time_slice, freq_slice, s, dir, pol] = np.arctan2(zi, zr)
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


def main(h5parmfile, solsetname='sol000', ampsoltabname='amplitude000',
         phasesoltabname='phase000', ref_id=0, smooth=False, normalize=True):
    """
    Fit screens to gain solutions

    Parameters
    ----------
    h5parmfile : str
        Filename of h5parm
    solsetname : str, optional
        Name of solset
    ampsoltabname : str, optional
        Name of amplitude soltab
    phasesoltabname : str, optional
        Name of phase soltab
    ref_id : int, optional
        Index of reference station
    smooth : bool, optional
        Smooth amp solutions
    normalize : bool, optional
        Normalize amp solutions
    """
    ref_id = int(ref_id)
    normalize = misc.string2bool(normalize)
    smooth = misc.string2bool(smooth)

    # Read in solutions
    H = h5parm(h5parmfile, readonly=False)
    solset = H.getSolset(solsetname)
    ampsoltab = solset.getSoltab(ampsoltabname)
    amp = np.array(ampsoltab.val)
    damp = np.ones(amp.shape)
    phasesoltab = solset.getSoltab(phasesoltabname)
    phase = np.array(phasesoltab.val)
    dphase = np.ones(phase.shape)

    # Smooth and normalize if desired
    if smooth:
        amp, damp = smooth_amps(ampsoltab, parms=amp, weights=damp)
        phase, dphase = smooth_phases(phasesoltab, parms=phase, weights=dphase,
                                      ref_id=ref_id)
    if normalize:
        amp, damp = normalize_direction(ampsoltab, remove_core_gradient=True,
                                        solset=solset, ref_id=ref_id, parms=amp,
                                        weights=damp)

    # Write the solutions back
    ampsoltab.setValues(amp)
    ampsoltab.setValues(damp, weight=True)
    phasesoltab.setValues(phase)
    phasesoltab.setValues(dphase, weight=True)
    H.close()


if __name__ == '__main__':
    descriptiontext = "Process gain solutions.\n"

    parser = argparse.ArgumentParser(description=descriptiontext, formatter_class=RawTextHelpFormatter)
    parser.add_argument('h5parmfile', help='Filename of input h5parm')
    parser.add_argument('--solsetname', help='Solset name', type=str, default='sol000')
    parser.add_argument('--ampsoltabname', help='Amplitude soltab name', type=str, default='amplitude000')
    parser.add_argument('--phasesoltabname', help='Phase soltab name', type=str, default='phase000')
    parser.add_argument('--ref_id', help='Reference station', type=int, default=0)
    parser.add_argument('--normalize', help='Normalize amplitude solutions', type=str, default='False')
    parser.add_argument('--smooth', help='Smooth amplitude solutions', type=str, default='False')
    args = parser.parse_args()
    main(args.h5parmfile, solsetname=args.solsetname, ampsoltabname=args.ampsoltabname,
         phasesoltabname=args.phasesoltabname, ref_id=args.ref_id, smooth=args.smooth,
         normalize=args.normalize)
