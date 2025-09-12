#!/usr/bin/env python3
"""
Script to combine two h5parms
"""
from argparse import ArgumentParser, RawTextHelpFormatter
from losoto.h5parm import h5parm
import logging
import os
import numpy as np
import scipy.interpolate as si
from astropy.stats import circmean
from rapthor.lib import miscellaneous as misc
import shutil
import tempfile
import losoto.operations


def expand_array(array, new_shape, new_axis_ind):
    """
    Expands an array along an axis

    Parameters
    ----------
    array : array
        Array to expand
    new_shape : list
        New shape of expanded array
    new_axis_ind : int
        Index of the axis to expand

    Returns
    -------
    new_array : array
        The expanded array
    """
    new_array = np.zeros(new_shape)
    slc = [slice(None)] * len(new_shape)
    for i in range(new_shape[new_axis_ind]):
        slc[new_axis_ind] = i
        new_array[tuple(slc)] = array

    return new_array


def average_polarizations(soltab):
    """
    Average solutions over XX and YY polarizations

    Note: phases are averaged using the circular mean and
    amplitudes are averaged in log space

    Parameters
    ----------
    soltab : soltab
        Solution table with the solutions to be averaged

    Returns
    -------
    vals : array
        The averaged values
    weights : array
        The averaged weights
    """
    # Read in various values
    pol_ind = soltab.getAxesNames().index('pol')
    vals = soltab.val[:]
    weights = soltab.weight[:]

    # Make sure flagged solutions are NaN and have zero weight
    flagged_ind = np.logical_or(~np.isfinite(vals), weights == 0.0)
    vals[flagged_ind] = np.nan
    weights[flagged_ind] = 0.0

    # Average the values over the polarization axis if desired. For the weights,
    # take the min value over the polarization axis
    if soltab.getType() == 'phase':
        # Use the circmean to get the average (circmean does not ignore NaNs,
        # so set flagged values to zero first)
        vals[flagged_ind] = 0.0
        vals = circmean(vals, axis=pol_ind, weights=weights)
    elif soltab.getType() == 'amplitude':
        # Take the average in log space. Use nanmean to get the average
        # (nanmean ignores NaNs)
        vals = np.log10(vals)
        vals = np.nanmean(vals, axis=pol_ind)
        vals = 10**vals
    else:
        # Use nanmean to get the average (nanmean ignores NaNs)
        vals = np.nanmean(vals, axis=pol_ind)
    weights = np.min(weights, axis=pol_ind)

    return vals, weights


def interpolate_solutions(fast_soltab, slow_soltab, final_axes_shapes,
                          slow_vals=None, slow_weights=None):
    """
    Interpolates slow phases or amplitudes to the fast time and frequency grid

    Parameters
    ----------
    fast_soltab : soltab
        Solution table with fast solutions
    slow_soltab : soltab
        Solution table with slow solutions
    final_axes_shapes : list of int
        Final shape of the output arrays
    slow_vals : array, optional
        Array of values to use for the slow solutions (useful if averaging
        has been done)
    slow_weights : array, optional
        Array of weights to use for the slow solutions (useful if averaging
        has been done)

    Returns
    -------
    vals_interp : array
        The interpolated values
    weights_interp : array
        The interpolated weights
    """
    # Read in various values
    time_ind = fast_soltab.getAxesNames().index('time')
    freq_ind = fast_soltab.getAxesNames().index('freq')
    if slow_vals is None:
        slow_vals = slow_soltab.val[:]
    if slow_weights is None:
        slow_weights = slow_soltab.weight[:]

    # Make sure flagged solutions have zero weight and fill them with 0 for
    # phases and 1 for amplitudes to avoid NaNs
    flagged_ind2 = np.logical_or(~np.isfinite(slow_vals), slow_weights == 0.0)
    if slow_soltab.getType() == 'phase':
        slow_vals[flagged_ind2] = 0.0
    elif slow_soltab.getType() == 'amplitude':
        slow_vals[flagged_ind2] = 1.0
    else:
        slow_vals[flagged_ind2] = 0.0
    slow_weights[flagged_ind2] = 0.0

    # Interpolate the values and weights
    if len(slow_soltab.time) > 1:
        f = si.interp1d(slow_soltab.time, slow_vals, axis=time_ind, kind='nearest', fill_value='extrapolate')
        vals_time_intep = f(fast_soltab.time)
        f = si.interp1d(slow_soltab.time, slow_weights, axis=time_ind, kind='nearest', fill_value='extrapolate')
        weights_time_intep = f(fast_soltab.time)
    else:
        # Just duplicate the single time to all times, without altering the freq axis
        axes_shapes_time_interp = final_axes_shapes[:]
        axes_shapes_time_interp[freq_ind] = slow_vals.shape[freq_ind]
        vals_time_intep = expand_array(slow_vals, axes_shapes_time_interp, time_ind)
        weights_time_intep = expand_array(slow_weights, axes_shapes_time_interp, time_ind)
    if len(slow_soltab.freq) > 1:
        f = si.interp1d(slow_soltab.freq, vals_time_intep, axis=freq_ind, kind='nearest', fill_value='extrapolate')
        vals_interp = f(fast_soltab.freq)
        f = si.interp1d(slow_soltab.freq, weights_time_intep, axis=freq_ind, kind='nearest', fill_value='extrapolate')
        weights_interp = f(fast_soltab.freq)
    else:
        # Just duplicate the single frequency to all frequencies
        vals_interp = expand_array(vals_time_intep, final_axes_shapes, freq_ind)
        weights_interp = expand_array(weights_time_intep, final_axes_shapes, freq_ind)

    # Set all flagged values to NaNs
    vals_interp[weights_interp == 0.0] = np.nan

    return vals_interp, weights_interp


def combine_phase1_amp2(ss1, ss2, sso):
    """
    Take phases from 1 and amplitudes from 2

    Parameters
    ----------
    ss1 : solset
        Solution set #1
    ss2 : solset
        Solution set #2
    sso : solset
        Output solution set

    Returns
    -------
    sso : solset
        Updated output solution set
    """
    # Remove unneeded soltabs from 1 and 2, then copy
    if 'amplitude000' in ss1.getSoltabNames():
        st = ss1.getSoltab('amplitude000')
        st.delete()
    if 'phase000' in ss2.getSoltabNames():
        st = ss2.getSoltab('phase000')
        st.delete()
    ss1.obj._f_copy_children(sso.obj, recursive=True, overwrite=True)
    ss2.obj._f_copy_children(sso.obj, recursive=True, overwrite=True)

    return sso


def combine_phase1_amp1_amp2(ss1, ss2, sso):
    """
    Take phases and amplitudes from 1 and amplitudes from 2 (amplitudes 1 and 2
    are multiplied to create combined values)

    Note: solset ss1 is assumed to have the coarser time grid. The frequency grid
    of the two solsets is assumed to be the same.

    Parameters
    ----------
    ss1 : solset
        Solution set #1
    ss2 : solset
        Solution set #2
    sso : solset
        Output solution set

    Returns
    -------
    sso : solset
        Updated output solution set
    """
    # First, copy metadata from solset #1 to the output solset
    ss1.obj._f_copy_children(sso.obj, recursive=True, overwrite=True)

    # Next, make the axes and their values for the output soltabs.
    # The ss2 solset has the faster time axis (frequency axis is identical),
    # so use it to derive the output axes shapes
    st1 = ss1.getSoltab('amplitude000')
    st2 = ss2.getSoltab('amplitude000')
    axes_names = st2.getAxesNames()
    axes_vals = []
    for axis in axes_names:
        axis_vals = st2.getAxisValues(axis)
        axes_vals.append(axis_vals)
    axes_shapes = [len(axis) for axis in axes_vals]

    # Interpolate the slow amplitudes in st1 to the fast grid and multiply them
    # with the fast ones
    vals, weights = interpolate_solutions(st2, st1, axes_shapes)
    vals *= st2.val
    weights *= st2.weight
    if 'amplitude000' in sso.getSoltabNames():
        st = sso.getSoltab('amplitude000')
        st.delete()
    sto = sso.makeSoltab(soltype='amplitude', soltabName='amplitude000', axesNames=axes_names,
                         axesVals=axes_vals, vals=vals, weights=weights)

    # Interpolate the slow phases in st1 to the fast grid.
    # Note: the output axes and their values are the same as for the amplitude solutions
    st1 = ss1.getSoltab('phase000')
    st2 = ss2.getSoltab('phase000')
    vals, weights = interpolate_solutions(st2, st1, axes_shapes)
    if 'phase000' in sso.getSoltabNames():
        st = sso.getSoltab('phase000')
        st.delete()
    sto = sso.makeSoltab(soltype='phase', soltabName='phase000', axesNames=axes_names,
                         axesVals=axes_vals, vals=vals, weights=weights)

    return sso


def combine_phase1_phase2_scalar(ss1, ss2, sso):
    """
    Take phases from 1 and phases from 2 (phases 2 are interpolated to time
    grid of 1 and summed)

    Parameters
    ----------
    ss1 : solset
        Solution set #1
    ss2 : solset
        Solution set #2
    sso : solset
        Output solution set

    Returns
    -------
    sso : solset
        Updated output solution set
    """
    # First, copy phases from 1
    ss1.obj._f_copy_children(sso.obj, recursive=True, overwrite=True)

    # Next, make the axes and their values for the output soltab
    st1 = ss1.getSoltab('phase000')
    st2 = ss2.getSoltab('phase000')
    axes_names = st1.getAxesNames()
    axes_vals = []
    for axis in axes_names:
        axis_vals = st1.getAxisValues(axis)
        axes_vals.append(axis_vals)
    axes_shapes = [len(axis) for axis in axes_vals]

    # Average and interpolate the slow phases, then add them to the fast ones
    vals, weights = interpolate_solutions(st1, st2, axes_shapes)
    vals += st1.val
    weights *= st1.weight
    if 'phase000' in sso.getSoltabNames():
        st = sso.getSoltab('phase000')
        st.delete()
    sto = sso.makeSoltab(soltype='phase', soltabName='phase000', axesNames=axes_names,
                         axesVals=axes_vals, vals=vals, weights=weights)

    return sso


def combine_phase1_phase2_amp2(ss1, ss2, sso):
    """
    Take phases from 1 and phases and amplitudes from 2 (phases 2 are averaged
    over XX and YY, then interpolated to time grid of 1 and summed)

    Parameters
    ----------
    ss1 : solset
        Solution set #1
    ss2 : solset
        Solution set #2
    sso : solset
        Output solution set

    Returns
    -------
    sso : solset
        Updated output solution set
    """
    # First, copy phases from 1
    ss1.obj._f_copy_children(sso.obj, recursive=True, overwrite=True)

    # Next, make the axes and their values for the output soltab
    st1 = ss1.getSoltab('phase000')
    st2 = ss2.getSoltab('phase000')
    axes_names = st1.getAxesNames()
    axes_vals = []
    for axis in axes_names:
        axis_vals = st1.getAxisValues(axis)
        axes_vals.append(axis_vals)
    axes_shapes = [len(axis) for axis in axes_vals]

    # Average and interpolate the slow phases, then add them to the fast ones
    vals, weights = average_polarizations(st2)
    vals, weights = interpolate_solutions(st1, st2, axes_shapes, slow_vals=vals,
                                          slow_weights=weights)
    vals += st1.val
    weights *= st1.weight
    if 'phase000' in sso.getSoltabNames():
        st = sso.getSoltab('phase000')
        st.delete()
    sto = sso.makeSoltab(soltype='phase', soltabName='phase000', axesNames=axes_names,
                         axesVals=axes_vals, vals=vals, weights=weights)

    # Copy amplitudes from 2
    # Remove unneeded phase soltab from 2, then copy
    if 'phase000' in ss2.getSoltabNames():
        st = ss2.getSoltab('phase000')
        st.delete()
    ss2.obj._f_copy_children(sso.obj, recursive=True, overwrite=True)

    return sso


def combine_phase1_phase2_amp2_diagonal(ss1, ss2, sso):
    """
    Take phases from 1 and phases and amplitudes from 2, XX and YY for both

    Parameters
    ----------
    ss1 : solset
        Solution set #1
    ss2 : solset
        Solution set #2
    sso : solset
        Output solution set

    Returns
    -------
    sso : solset
        Updated output solution set
    """
    # First, copy phases from 1
    ss1.obj._f_copy_children(sso.obj, recursive=True, overwrite=True)

    # Next, make the axes and their values for the output soltab
    st1 = ss1.getSoltab('phase000')
    st2 = ss2.getSoltab('phase000')
    axes_names = st2.getAxesNames()
    axes_vals = []
    for axis in axes_names:
        if axis == 'time' or axis == 'freq':
            # Take time and frequency values from 1
            axis_vals = st1.getAxisValues(axis)
        else:
            # Take other values from 2
            axis_vals = st2.getAxisValues(axis)
        axes_vals.append(axis_vals)
    axes_shapes = [len(axis) for axis in axes_vals]

    # Interpolate the slow phases, then add them to the fast ones (after
    # expanding them to include the pol axis)
    vals, weights = interpolate_solutions(st1, st2, axes_shapes)
    pol_ind = axes_names.index('pol')
    st1_vals = expand_array(st1.val, axes_shapes, pol_ind)
    vals += st1_vals
    st1_weights = expand_array(st1.weight, axes_shapes, pol_ind)
    weights *= st1_weights
    if 'phase000' in sso.getSoltabNames():
        st = sso.getSoltab('phase000')
        st.delete()
    sto = sso.makeSoltab(soltype='phase', soltabName='phase000', axesNames=axes_names,
                         axesVals=axes_vals, vals=vals, weights=weights)

    # Copy amplitudes from 2
    # Remove unneeded phase soltab from 2, then copy
    if 'phase000' in ss2.getSoltabNames():
        st = ss2.getSoltab('phase000')
        st.delete()
    ss2.obj._f_copy_children(sso.obj, recursive=True, overwrite=True)

    return sso


def combine_phase1_phase2_amp2_scalar(ss1, ss2, sso):
    """
    Take phases from 1 and phases and amplitudes from 2, scalar for both

    Parameters
    ----------
    ss1 : solset
        Solution set #1
    ss2 : solset
        Solution set #2
    sso : solset
        Output solution set

    Returns
    -------
    sso : solset
        Updated output solution set
    """
    # First, copy phases from 1
    ss1.obj._f_copy_children(sso.obj, recursive=True, overwrite=True)

    # Next, make the axes and their values for the output soltab
    st1 = ss1.getSoltab('phase000')
    st2 = ss2.getSoltab('phase000')
    axes_names = st1.getAxesNames()
    axes_names2 = st2.getAxesNames()
    axes_vals = []
    for axis in axes_names:
        axis_vals = st1.getAxisValues(axis)
        axes_vals.append(axis_vals)
    axes_shapes = [len(axis) for axis in axes_vals]

    # Average and interpolate the slow phases, then add them to the fast ones
    vals, weights = average_polarizations(st2)
    vals, weights = interpolate_solutions(st1, st2, axes_shapes, slow_vals=vals,
                                          slow_weights=weights)
    vals += st1.val
    weights *= st1.weight
    if 'phase000' in sso.getSoltabNames():
        st = sso.getSoltab('phase000')
        st.delete()
    sto = sso.makeSoltab(soltype='phase', soltabName='phase000', axesNames=axes_names,
                         axesVals=axes_vals, vals=vals, weights=weights)

    # Average the amplitudes (no interpolation needed)
    st2 = ss2.getSoltab('amplitude000')
    vals, weights = average_polarizations(st2)
    if 'amplitude000' in sso.getSoltabNames():
        st = sso.getSoltab('amplitude000')
        st.delete()
    axes_vals = []
    axes_names2.pop(axes_names2.index('pol'))  # remove pol axis in output axis names
    for axis in axes_names2:
        axis_vals = st2.getAxisValues(axis)
        axes_vals.append(axis_vals)
    sto = sso.makeSoltab(soltype='amplitude', soltabName='amplitude000', axesNames=axes_names2,
                         axesVals=axes_vals, vals=vals, weights=weights)

    return sso


def copy_solset(ss1, ss2):
    """
    Copies ss1 to ss2

    Parameters
    ----------
    ss1 : solset
        Solution set #1
    ss2 : solset
        Solution set #2

    Returns
    -------
    ss2 : solset
        Updated solution set
    """
    ss1.obj._f_copy_children(ss2.obj, recursive=True, overwrite=True)

    return ss2


def main(h5parm1, h5parm2, outh5parm, mode, solset1='sol000', solset2='sol000',
         reweight=False, cal_names=None, cal_fluxes=None):
    """
    Combines two h5parms

    Parameters
    ----------
    h5parm1 : str
        Filename of h5parm 1. Solution axes are assumed to be in the
        standard DDECal order of ['time', 'freq', 'ant', 'dir']
    h5parm2 : str
        Filename of h5parm 2. Solution axes are assumed to be in the
        standard DDECal order of ['time', 'freq', 'ant', 'dir', 'pol']
    outh5parm : str
        Filename of the output h5parm
    mode : str
        Mode to use when combining:
        'p1a2' - phases from 1 and amplitudes from 2
        'p1a1a2' - phases and amplitudes from 1 and amplitudes from 2 (amplitudes 1 and 2
        are multiplied to create combined amplitudes)
        'p1p2a2' - phases from 1 and phases and amplitudes from 2 (phases 2 are averaged
        over XX and YY, then interpolated to time grid of 1 and summed)
        'p1p2a2_diagonal' - phases from 1 and phases and amplitudes from 2, XX and YY for
        both
        'p1p2a2_scalar' - phases from 1 and phases and amplitudes from 2, scalar for both
        'separate' - no sum or multiplication; solutions are copied as separate solsets
    solset1 : str, optional
        Name of solset for h5parm1
    solset2 : str, optional
        Name of solset for h5parm2
    reweight : bool, optional
        If True, reweight the solutions by their detrended noise
    cal_names : str or list, optional
        List of calibrator names (for use in reweighting)
    cal_fluxes : str or list, optional
        List of calibrator flux densities (for use in reweighting)
    """
    reweight = misc.string2bool(reweight)
    cal_names = misc.string2list(cal_names)
    cal_fluxes = misc.string2list(cal_fluxes)

    known_modes = ('p1a2', 'p1p2_scalar', 'p1a1a2', 'p1p2a2', 'p1p2a2_diagonal', 'p1p2a2_scalar', 'separate')
    if mode not in known_modes:
        raise ValueError(f'Mode {mode} unknown. Supported modes are: {", ".join(known_modes)}')

    # Make copies of the input h5parms (since they may be altered by steps below) and
    # open them
    with tempfile.TemporaryDirectory() as tmpdir:
        h5parm1_copy = shutil.copy(h5parm1, tmpdir)
        h5parm2_copy = shutil.copy(h5parm2, tmpdir)
        if os.path.exists(outh5parm):
            os.remove(outh5parm)

        with (
            h5parm(h5parm1_copy, readonly=False) as h1,
            h5parm(h5parm2_copy, readonly=False) as h2,
            h5parm(outh5parm, readonly=False) as ho
        ):

            ss1 = h1.getSolset(solset=solset1)
            ss2 = h2.getSolset(solset=solset2)
            sso = ho.makeSolset(solsetName='sol000', addTables=False)

            if mode == 'p1a2':
                # Take phases from 1 and amplitudes from 2
                sso = combine_phase1_amp2(ss1, ss2, sso)

            if mode == 'p1p2_scalar':
                # Take phases from 1 and phases from 2
                sso = combine_phase1_phase2_scalar(ss1, ss2, sso)

            elif mode == 'p1a1a2':
                # Take phases and amplitudes from 1 and amplitudes from 2
                sso = combine_phase1_amp1_amp2(ss1, ss2, sso)

            elif mode == 'p1p2a2':
                # Take phases from 1 and phases and amplitudes from 2
                sso = combine_phase1_phase2_amp2(ss1, ss2, sso)

            elif mode == 'p1p2a2_diagonal':
                # Take phases from 1 and phases and amplitudes from 2, diagonal
                sso = combine_phase1_phase2_amp2_diagonal(ss1, ss2, sso)

            elif mode == 'p1p2a2_scalar':
                # Take phases from 1 and phases and amplitudes from 2, scalar
                sso = combine_phase1_phase2_amp2_scalar(ss1, ss2, sso)

            elif mode == 'separate':
                # No sum or multiplication is done. The solutions from 1 and 2 are copied
                # to the output as separate solsets (named sol000 for 1 and sol001 for 2)
                sso = copy_solset(ss1, sso)
                sso2 = ho.makeSolset(solsetName='sol001', addTables=False)
                sso2 = copy_solset(ss2, sso2)

    # Reweight
    if reweight:
        # Use the scatter on the solutions for weighting, with an additional scaling
        # by the calibrator flux densities in each direction
        with h5parm(outh5parm, readonly=False) as ho:
            sso = ho.getSolset(solset='sol000')

            # Reweight the phases. Reweighting doesn't work when there are too few samples,
            # so check there are at least 10
            soltab_ph = sso.getSoltab('phase000')
            if len(soltab_ph.time) > 10:
                # Set window size for std. dev. calculation. We try to get one of around
                # 30 minutes, as that is roughly the timescale on which the global properties
                # of the ionosphere are expected to change
                delta_times = soltab_ph.time[1:] - soltab_ph.time[:-1]
                timewidth = np.min(delta_times)
                nstddev = min(251, max(11, int(1800/timewidth)))
                if nstddev % 2 == 0:
                    # Ensure window is odd
                    nstddev += 1
                losoto.operations.reweight.run(soltab_ph, mode='window', nmedian=3, nstddev=nstddev)

            # Reweight the amplitudes
            soltab_amp = sso.getSoltab('amplitude000')
            if len(soltab_amp.time) > 10:
                # Set window size for std. dev. calculation. We try to get one of around
                # 90 minutes, as that is roughly the timescale on which the global properties
                # of the beam errors are expected to change
                delta_times = soltab_amp.time[1:] - soltab_amp.time[:-1]
                timewidth = np.min(delta_times)
                nstddev = min(251, max(11, int(5400/timewidth)))
                if nstddev % 2 == 0:
                    # Ensure window is odd
                    nstddev += 1
                losoto.operations.reweight.run(soltab_amp, mode='window', nmedian=5, nstddev=nstddev)

            # Use the input calibrator flux densities to adjust the weighting done above
            # to ensure that the average weights are proportional to the square of the
            # calibrator flux densities
            sso = ho.getSolset(solset='sol000')
            soltab_ph = sso.getSoltab('phase000')
            soltab_amp = sso.getSoltab('amplitude000')
            dir_names = [d.strip('[]') for d in soltab_ph.dir[:]]
            cal_weights = []
            for dir_name in dir_names:
                cal_weights.append(cal_fluxes[cal_names.index(dir_name)])
            cal_weights = [float(c) for c in cal_weights]
            cal_weights = np.array(cal_weights)**2

            # Convert weights to float64 from float16 to avoid clipping in the
            # intermediate steps, and set flagged (weight = 0) ones to NaN
            # so they are not included in the calculations
            weights_ph = np.array(soltab_ph.weight, dtype=float)
            weights_amp = np.array(soltab_amp.weight, dtype=float)
            weights_ph[weights_ph == 0.0] = np.nan
            weights_amp[weights_amp == 0.0] = np.nan

            # Reweight, keeping the median value of the weights the same (to avoid
            # changing the overall normalization, which should be the inverse square of the
            # uncertainty (scatter) in the solutions).
            global_median_ph = np.nanmedian(weights_ph)
            global_median_amp = np.nanmedian(weights_amp)
            for d in range(len(dir_names)):
                # Input data are [time, freq, ant, dir, pol] for slow amplitudes
                # and [time, freq, ant, dir] for fast phases (scalarphase)
                norm_factor = cal_weights[d] / np.nanmedian(weights_ph[:, :, :, d])
                weights_ph[:, :, :, d] *= norm_factor
                if mode == 'p1p2a2_scalar':
                    norm_factor = cal_weights[d] / np.nanmedian(weights_amp[:, :, :, d])
                    weights_amp[:, :, :, d] *= norm_factor
                else:
                    norm_factor = cal_weights[d] / np.nanmedian(weights_amp[:, :, :, d, :])
                    weights_amp[:, :, :, d, :] *= norm_factor
            weights_ph *= global_median_ph / np.nanmedian(weights_ph)
            weights_amp *= global_median_amp / np.nanmedian(weights_amp)
            weights_ph[~np.isfinite(weights_ph)] = 0.0
            weights_amp[~np.isfinite(weights_amp)] = 0.0

            # Clip to fit in float16 (required by LoSoTo)
            float16max = 65504.0
            weights_ph[weights_ph > float16max] = float16max
            weights_amp[weights_amp > float16max] = float16max

            # Write new weights
            soltab_ph.setValues(weights_ph, weight=True)
            soltab_amp.setValues(weights_amp, weight=True)


if __name__ == '__main__':
    descriptiontext = "Combine two h5parms.\n"

    parser = ArgumentParser(description=descriptiontext, formatter_class=RawTextHelpFormatter)
    parser.add_argument('h51', help='Filename of input h5 1')
    parser.add_argument('h52', help='Filename of input h5 2')
    parser.add_argument('outh5', help='Filename of the output h5')
    parser.add_argument('mode', help='Mode to use')
    parser.add_argument('--reweight', help='Reweight solutions', type=str, default='False')
    parser.add_argument('--cal_names', help='Names of calibrators', type=str, default='')
    parser.add_argument('--cal_fluxes', help='Flux densities of calibrators', type=str, default='')
    args = parser.parse_args()

    try:
        main(args.h51, args.h52, args.outh5, args.mode, reweight=args.reweight,
             cal_names=args.cal_names, cal_fluxes=args.cal_fluxes)
    except ValueError as e:
        log = logging.getLogger('rapthor:combine_h5parms')
        log.critical(e)
