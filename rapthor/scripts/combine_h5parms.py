#! /usr/bin/env python3
"""
Script to combine two h5parms
"""
import argparse
from argparse import RawTextHelpFormatter
from losoto.h5parm import h5parm
import os
import sys
import numpy as np
import scipy.interpolate as si
from astropy.stats import circmean
from rapthor.lib import miscellaneous as misc


def main(h5parm1, h5parm2, outh5parm, mode, solset1='sol000', solset2='sol000',
         reweight=False):
    """
    Combines two h5parms

    Parameters
    ----------
    h5parm1 : str
        Filename of h5parm 1
    h5parm2 : str
        Filename of h5parm 2
    outh5parm : str
        Filename of the output h5parm
    mode : str
        Mode to use when combining:
        'p1a2' - phases from 1 and amplitudes from 2
        'p1a1a2' - phases and amplitudes from 1 and amplitudes from 2 (amplitudes 1 and 2
        are multiplied to create combined amplitudes)
        'p1p2a2' - phases from 1 and phases and amplitudes from 2 (phases 2 are averaged
        over XX and YY, then interpolated to time grid of 1 and summed)
    solset1 : str, optional
        Name of solset for h5parm1
    solset2 : str, optional
        Name of solset for h5parm2
    reweight : bool, optional
        If True, reweight the solutions by their detrended noise
    """
    reweight = misc.string2bool(reweight)

    # Open the input h5parms
    h1 = h5parm(h5parm1, readonly=False)
    h2 = h5parm(h5parm2, readonly=False)
    ss1 = h1.getSolset(solset=solset1)
    ss2 = h2.getSolset(solset=solset2)

    # Initialize the output h5parm
    if os.path.exists(outh5parm):
        os.remove(outh5parm)
    ho = h5parm(outh5parm, readonly=False)
    sso = ho.makeSolset(solsetName='sol000', addTables=False)

    if mode == 'p1a2':
        # Take phases from 1 and amplitudes from 2
        # Remove unneeded soltabs from 1 and 2, then copy
        if 'amplitude000' in ss1.getSoltabNames():
            st = ss1.getSoltab('amplitude000')
            st.delete()
        if 'phase000' in ss2.getSoltabNames():
            st = ss2.getSoltab('phase000')
            st.delete()
        ss1.obj._f_copy_children(sso.obj, recursive=True, overwrite=True)
        ss2.obj._f_copy_children(sso.obj, recursive=True, overwrite=True)

    elif mode == 'p1a1a2':
        # Take phases and amplitudes from 1 and amplitudes from 2 (amplitudes 1 and 2
        # are multiplied to create combined values)
        # First, copy phases and amplitudes from 1
        ss1.obj._f_copy_children(sso.obj, recursive=True, overwrite=True)

        # Then read amplitudes from 1 and 2, multiply them together, and store
        st1 = ss1.getSoltab('amplitude000')
        st2 = ss2.getSoltab('amplitude000')
        sto = sso.getSoltab('amplitude000')
        sto.setValues(st1.val*st2.val)

    elif mode == 'p1p2a2':
        # Take phases from 1 and phases and amplitudes from 2 (phases 2 are averaged
        # over XX and YY, then interpolated to time grid of 1 and summed)
        # First, copy phases from 1
        ss1.obj._f_copy_children(sso.obj, recursive=True, overwrite=True)

        # Read phases from 2, average XX and YY (using circmean), interpolate to match
        # those from 1, and sum. Note: the interpolation is done in phase space (instead
        # of real/imag space) since phase wraps are not expected to be present in the
        # slow phases
        st1 = ss1.getSoltab('phase000')
        st2 = ss2.getSoltab('phase000')
        axis_names = st1.getAxesNames()
        time_ind = axis_names.index('time')
        freq_ind = axis_names.index('freq')
        axis_names = st2.getAxesNames()
        pol_ind = axis_names.index('pol')
        val2 = circmean(st2.val, axis=pol_ind)  # average over XX and YY
        f = si.interp1d(st2.time, val2, axis=time_ind, kind='nearest', fill_value='extrapolate')
        v1 = f(st1.time)
        f = si.interp1d(st2.freq, v1, axis=freq_ind, kind='linear', fill_value='extrapolate')
        vals = f(st1.freq) + st1.val
        sto = sso.getSoltab('phase000')
        sto.setValues(vals)

        # Copy amplitudes from 2
        # Remove unneeded phase soltab from 2, then copy
        if 'phase000' in ss2.getSoltabNames():
            st = ss2.getSoltab('phase000')
            st.delete()
        ss2.obj._f_copy_children(sso.obj, recursive=True, overwrite=True)

    else:
        print('ERROR: mode not understood')
        sys.exit(1)
    h1.close()
    h2.close()
    ho.close()

    # Reweight
    if reweight:
        ho = h5parm(outh5parm, readonly=False)
        sso = ho.makeSolset(solsetName='sol000', addTables=False)

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
            reweight.run(soltab_ph, mode='window', nmedian=3, nstddev=nstddev)

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
            reweight.run(soltab_amp, mode='window', nmedian=5, nstddev=nstddev)

        ho.close()


if __name__ == '__main__':
    descriptiontext = "Combine two h5parms.\n"

    parser = argparse.ArgumentParser(description=descriptiontext, formatter_class=RawTextHelpFormatter)
    parser.add_argument('h51', help='Filename of input h5 1')
    parser.add_argument('h52', help='Filename of input h5 2')
    parser.add_argument('outh5', help='Filename of the output h5')
    parser.add_argument('mode', help='Mode to use')
    parser.add_argument('--reweight', help='Reweight solutions', type=str, default='False')
    args = parser.parse_args()

    main(args.h51, args.h52, args.outh5, args.mode, reweight=args.reweight)
