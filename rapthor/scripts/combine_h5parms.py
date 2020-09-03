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


def main(h5parm1, h5parm2, outh5parm, mode, solset1='sol000', solset2='sol000'):
    """
    Combines two h5parms

    Parameters
    ----------
    h5parm1 : str
        Filenames of h5parm 1
    h5parm2 : str
        Filenames of h5parm 2
    outh5parm : str
        Filename of the output h5parm
    mode : str
        Mode to use when combining:
        'p1a2' - phases from 1 and amplitudes from 2
        'p1a1a2' - phases and amplitudes from 1 and amplitudes from 2 (amplitudes 1 and 2
        are multiplied to create combined amplitudes)
        'p1a1p2a2' - phases and amplitudes from 1 and from 2 (amplitudes 1 and 2
        are multiplied to create combined amplitudes, phases 2 are averaged over XX and
        YY, then interpolated to time grid of 1 and summed)
    solset1 : str, optional
        Name of solset for h5parm1
    solset2 : str, optional
        Name of solset for h5parm2
    """
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
        #  over XX and YY, then interpolated to time grid of 1 and summed)
        # First, copy phases from 1
        ss1.obj._f_copy_children(sso.obj, recursive=True, overwrite=True)

        # Read phases from 2, average XX and YY, interpolate to match those from 1, and
        # sum
        st1 = ss1.getSoltab('phase000')
        st2 = ss2.getSoltab('phase000')
        axis_names = st1.getAxesNames()
        time_ind = axis_names.index('time')
        freq_ind = axis_names.index('freq')
        axis_names = st2.getAxesNames()
        pol_ind = axis_names.index('pol')
        val2 = np.mean(st2.val, axis=pol_ind)  # average over XX and YY
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

    elif mode == 'p1a1p2a2':
        # Take phases and amplitudes from 1 and from 2 (amplitudes 1 and 2
        # are multiplied to create combined values, phases 2 are averaged over XX and
        # YY, then interpolated to time grid of 1 and summed)
        # First, copy phases and amplitudes from 1
        ss1.obj._f_copy_children(sso.obj, recursive=True, overwrite=True)

        # Then read amplitudes from 1 and 2, multiply them together, and store
        st1 = ss1.getSoltab('amplitude000')
        st2 = ss2.getSoltab('amplitude000')
        sto = sso.getSoltab('amplitude000')
        sto.setValues(st1.val*st2.val)

        # Read phases from 2, average XX and YY, interpolate to match those from 1, and
        # sum
        st1 = ss1.getSoltab('phase000')
        st2 = ss2.getSoltab('phase000')
        axis_names = st1.getAxesNames()
        time_ind = axis_names.index('time')
        freq_ind = axis_names.index('freq')
        axis_names = st2.getAxesNames()
        pol_ind = axis_names.index('pol')
        val2 = np.mean(st2.val, axis=pol_ind)  # average over XX and YY
        f = si.interp1d(st2.time, val2, axis=time_ind, kind='nearest', fill_value='extrapolate')
        v1 = f(st1.time)
        f = si.interp1d(st2.freq, v1, axis=freq_ind, kind='linear', fill_value='extrapolate')
        vals = f(st1.freq) + st1.val
        sto = sso.getSoltab('phase000')
        sto.setValues(vals)

    else:
        print('ERROR: mode not understood')
        sys.exit(1)

    h1.close()
    h2.close()
    ho.close()


if __name__ == '__main__':
    descriptiontext = "Combine two h5parms.\n"

    parser = argparse.ArgumentParser(description=descriptiontext, formatter_class=RawTextHelpFormatter)
    parser.add_argument('h51', help='name of input h5 1')
    parser.add_argument('h52', help='name of input h5 2')
    parser.add_argument('outh5', help='name of the output h5')
    parser.add_argument('mode', help='mode to use')
    args = parser.parse_args()

    main(args.h51, args.h52, args.outh5, args.mode)
