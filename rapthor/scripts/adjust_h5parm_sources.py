#!/usr/bin/env python3
"""
Script to adjust h5parm source coordinates to match those in the sky model
"""
import argparse
from argparse import RawTextHelpFormatter
from rapthor.lib import miscellaneous as misc
import lsmtool
from losoto.h5parm import h5parm
import numpy as np


def main(skymodel, h5parm_file, solset_name='sol000'):
    """
    Adjust the h5parm source coordinates to match those in the sky model

    Parameters
    ----------
    skymodel : str
        Filename of calibration sky model
    h5parm_file : str
        Filename of h5parm with cooresponding solutions to update with
        the facet positions
    solset_name : str, optional
        Name of the solution set to adjust
    """
    # Open the input sky model and h5parm file and match up the source names and
    # positions
    skymod = lsmtool.load(skymodel)
    source_dict = skymod.getPatchPositions()
    H = h5parm(h5parm_file, readonly=False)
    solset = H.getSolset(solset_name)
    soltab = solset.getSoltabs()[0]  # take the first soltab (all soltabs have the same directions)
    source_positions = []
    for source in soltab.dir:
        # For each source in the soltab, find its coordinates in the sky model
        # (stored in the source_dict dictionary)
        radecpos = source_dict[source.strip('[]')]  # degrees
        source_positions.append([misc.normalize_ra(radecpos[0].value),
                                 misc.normalize_dec(radecpos[1].value)])
    source_positions = np.array(source_positions)
    ra_deg = source_positions.T[0]
    dec_deg = source_positions.T[1]

    # Update the source table of the solution set
    sourceTable = solset.obj._f_get_child('source')
    vals = [[ra*np.pi/180.0, dec*np.pi/180.0] for ra, dec in zip(ra_deg, dec_deg)]  # radians
    sourceTable = list(zip(*(soltab.dir, vals)))
    H.close()


if __name__ == '__main__':
    descriptiontext = "Blank regions of an image.\n"

    parser = argparse.ArgumentParser(description=descriptiontext, formatter_class=RawTextHelpFormatter)
    parser.add_argument('skymodel', help='Filename of input sky model')
    parser.add_argument('h5parm_file', help='Filename of h5parm with cooresponding solutions to update')
    parser.add_argument('--solset_name', help='Name of solution set', type=str, default='sol000')
    args = parser.parse_args()
    main(args.skymodel, args.h5parm_file, solset_name=args.solset_name)