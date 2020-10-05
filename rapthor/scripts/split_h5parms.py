#! /usr/bin/env python3
"""
Script to split an h5parm
"""
import argparse
from argparse import RawTextHelpFormatter
from losoto.h5parm import h5parm
import os
import numpy as np
from rapthor.lib import miscellaneous as misc


def main(inh5parm, outh5parms, soltabname='phase000', insolset='sol000'):
    """
    Combines two h5parms

    Parameters
    ----------
    inh5parm : str
        Filename of h5parm  to split
    outh5parms : str
        Filenames of split h5parms as comma-separated string
    soltabname : str, optional
        Name of soltab to use. If "gain" is in the name, phase and amplitudes are used
    insolset : str, optional
        Name of solset to split
    """
    output_h5parm_list = misc.string2list(outh5parms)
    nchunks = len(output_h5parm_list)

    # Read input table
    h5 = h5parm(inh5parm, readonly=True)
    solset = h5.getSolset(insolset)
    if 'gain' in soltabname:
        soltab_amp = solset.getSoltab(soltabname.replace('gain', 'amplitude'))
        soltab_ph = solset.getSoltab(soltabname.replace('gain', 'phase'))
    else:
        soltab_ph = solset.getSoltab(soltabname)
    pointingNames = []
    antennaNames = []
    pointingDirections = []
    antennaPositions = []
    ants = solset.getAnt()
    sous = solset.getSou()
    for k, v in list(sous.items()):
        if k not in pointingNames:
            pointingNames.append(k)
            pointingDirections.append(v)
    for k, v in list(ants.items()):
        if k not in antennaNames:
            antennaNames.append(k)
            antennaPositions.append(v)

    # Read in times
    times_fast = soltab_ph.time
    if 'gain' in soltabname:
        times_slow = soltab_amp.time

    # Identify any gaps in time and put initial breaks there
    delta_times = times_fast[1:] - times_fast[:-1]  # time at center of solution interval
    timewidth = np.min(delta_times)
    gaps = np.where(delta_times > timewidth*1.2)
    gaps_ind = gaps[0] + 1
    gaps_ind = np.append(gaps_ind, np.array([len(times_fast)]))

    # Add additional breaks to reach the desired number of chunks
    if len(gaps_ind) >= nchunks:
        gaps_ind = gaps_ind[:nchunks]
        check_gaps = False
    else:
        check_gaps = True
    while check_gaps:
        chunk_size = int(len(times_fast) / nchunks)

        check_gaps = False
        g_start = 0
        gaps_ind_copy = gaps_ind.copy()
        for gnum, g_stop in enumerate(gaps_ind_copy):
            if g_stop - g_start > chunk_size:
                new_gap = g_start + int((g_stop - g_start) / 2)
                gaps_ind = np.insert(gaps_ind, gnum, np.array([new_gap]))
                check_gaps = True
                break
            g_start = g_stop
    if len(gaps_ind) >= nchunks:
        gaps_ind = gaps_ind[:nchunks]
        check_gaps = False
    gaps_sec = times_fast[gaps_ind-1]

    # Fill the output files
    for i, outh5file in enumerate(output_h5parm_list):
        if os.path.exists(outh5file):
            os.remove(outh5file)
        outh5 = h5parm(outh5file, readonly=False)
        solsetOut = outh5.makeSolset('sol000')

        # Store phases
        if i == 0:
            startval = times_fast[0]
        else:
            startval = gaps_sec[i-1]
        if i == nchunks-1:
            endval = times_fast[-1]
        else:
            endval = gaps_sec[i] - 0.5  # subtract 0.5 sec to ensure "[)" range
        soltab_ph.setSelection(time={'min': startval, 'max': endval, 'step': 1})
        solsetOut.makeSoltab('phase', 'phase000', axesNames=soltab_ph.getAxesNames(),
                             axesVals=[soltab_ph.getAxisValues(a) for a in soltab_ph.getAxesNames()],
                             vals=soltab_ph.getValues(retAxesVals=False),
                             weights=soltab_ph.getValues(weight=True, retAxesVals=False))
        soltab_ph.clearSelection()

        # Store amps
        if 'gain' in soltabname:
            if i == 0:
                startval = times_slow[0]
            else:
                startval = gaps_sec[i-1]
            if i == nchunks-1:
                endval = times_slow[-1]
            else:
                endval = gaps_sec[i] - 0.5  # subtract 0.5 sec to ensure "[)" range
            soltab_amp.setSelection(time={'min': startval, 'max': endval, 'step': 1})
            solsetOut.makeSoltab('amplitude', 'amplitude000', axesNames=soltab_amp.getAxesNames(),
                                 axesVals=[soltab_amp.getAxisValues(a) for a in soltab_amp.getAxesNames()],
                                 vals=soltab_amp.getValues(retAxesVals=False),
                                 weights=soltab_amp.getValues(weight=True, retAxesVals=False))
            soltab_amp.clearSelection()

        # Store metadata
        sourceTable = solsetOut.obj._f_get_child('source')
        antennaTable = solsetOut.obj._f_get_child('antenna')
        antennaTable.append(list(zip(*(antennaNames, antennaPositions))))
        sourceTable.append(list(zip(*(pointingNames, pointingDirections))))
        outh5.close()


if __name__ == '__main__':
    descriptiontext = "Combine two h5parms.\n"

    parser = argparse.ArgumentParser(description=descriptiontext, formatter_class=RawTextHelpFormatter)
    parser.add_argument('inh5parm', help='name of input h5parm')
    parser.add_argument('outh5parms', help='name of output h5parm')
    parser.add_argument('--soltabname', help='name of the soltab', type=str, default='phase000')
    args = parser.parse_args()

    main(args.inh5parm, args.outh5parms, soltabname=args.soltabname)
