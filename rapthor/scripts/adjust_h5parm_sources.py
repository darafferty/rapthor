#!/usr/bin/env python3
"""
Script to adjust h5parm source coordinates to match those in the sky model
"""
from argparse import ArgumentParser, RawTextHelpFormatter
from rapthor.lib import miscellaneous as misc
import lsmtool
from losoto.h5parm import h5parm
import numpy as np
import sys


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
    # positions to get the adjusted values
    skymod = lsmtool.load(skymodel)
    source_dict = skymod.getPatchPositions()
    parms = h5parm(h5parm_file, readonly=False)
    try:
        solset = parms.getSolset(solset_name)
        soltab = solset.getSoltabs()[0]  # take the first soltab (all have the same directions)
        if not len(source_dict) == len(soltab.dir):
            sys.exit('ERROR: The patches in the sky model and the directions in the h5parm '
                     'must have the same length')
        source_positions = []
        for source in soltab.dir:
            # For each source in the soltab, find its coordinates in the sky model
            # (stored in the source_dict dictionary)
            try:
                radecpos = source_dict[source.strip('[]')]  # degrees
            except KeyError:
                sys.exit('ERROR: A direction is present in the h5parm that is not in the sky model')
            ra, dec = misc.normalize_ra_dec(radecpos[0].value, radecpos[1].value)
            source_positions.append([ra, dec])
        source_positions = np.array(source_positions)
        ra_deg = source_positions.T[0]
        dec_deg = source_positions.T[1]
        vals = [[ra*np.pi/180.0, dec*np.pi/180.0] for ra, dec in zip(ra_deg, dec_deg)]  # radians

        # Remove the old source table and make a new empty one
        sourceTable = solset.obj._f_get_child('source')
        sourceTable._f_remove(recursive=True)
        descriptor = np.dtype([('name', np.str_, 128), ('dir', np.float32, 2)])
        snode = parms.H.get_node('/', solset_name)
        _ = parms.H.create_table(snode, 'source', descriptor, title='Source names and directions',
                                 expectedrows=25)

        # Add the adjusted values to the new table
        sourceTable = solset.obj._f_get_child('source')
        sourceTable.append(list(zip(*(soltab.dir, vals))))
    finally:
        # Close the h5parm file
        parms.close()


if __name__ == '__main__':
    descriptiontext = "Adjust the h5parm source coordinates to match those in the sky model.\n"

    parser = ArgumentParser(description=descriptiontext, formatter_class=RawTextHelpFormatter)
    parser.add_argument('skymodel', help='Filename of input sky model')
    parser.add_argument('h5parm_file', help='Filename of h5parm with cooresponding solutions to update')
    parser.add_argument('--solset_name', help='Name of solution set', type=str, default='sol000')
    args = parser.parse_args()
    main(args.skymodel, args.h5parm_file, solset_name=args.solset_name)
