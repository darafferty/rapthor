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
    with h5parm(h5parm_file, readonly=False) as parms:
        solset = parms.getSolset(solset_name)
        soltab = solset.getSoltabs()[0]  # take the first soltab (all have the same directions)
        duplicate_solutions = False
        if not len(source_dict) == len(soltab.dir):
            if len(soltab.dir) == 1:
                print('H5parm has a single direction. These solutions will be duplicated for '
                      'all directions ')
                duplicate_solutions = True
            else:
                sys.exit('ERROR: The patches in the sky model and the directions in the h5parm '
                         'must have the same length')
        source_names = []
        source_positions = []
        for source in soltab.dir:
            # For each source in the soltab, find its coordinates in the sky model
            # (stored in the source_dict dictionary)
            if len(soltab.dir) == 1:
                # For direction-independent solutions, use all the sources in the sky
                # model
                for skymodel_source_name, skymodel_source_position in source_dict.items():
                    source_names.append(f'[{skymodel_source_name}]')
                    ra, dec = misc.normalize_ra_dec(skymodel_source_position[0].value,
                                                    skymodel_source_position[1].value)
                    source_positions.append([ra, dec])
            else:
                try:
                    radecpos = source_dict[source.strip('[]')]  # degrees
                except KeyError:
                    sys.exit('ERROR: A direction is present in the h5parm that is not in the sky model')
                ra, dec = misc.normalize_ra_dec(radecpos[0].value, radecpos[1].value)
                source_names.append(source)
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

        # For direction-indepedent case, duplicate solutions
        # for each new direction
        if duplicate_solutions:
            for soltab in solset.getSoltabs():
                vals, axes = soltab.getValues()
                weights, _ = soltab.getValues(weight=True)
                new_vals = np.zeros((vals.shape, len(source_names)))
                new_weights = np.zeros((weights.shape, len(source_names)))
                for i in range(len(source_names)):
                    new_vals[:, i] = vals
                    new_weights[:, i] = weights
                soltab_name = soltab.name
                soltab.delete()
                soltab = solset.makeSoltab('amplitude', soltab_name, axesNames=['freq', 'dir'],
                                           axesVals=[frequencies, source_names], vals=new_vals,
                                           weights=new_weights)


if __name__ == '__main__':
    descriptiontext = "Adjust the h5parm source coordinates to match those in the sky model.\n"

    parser = ArgumentParser(description=descriptiontext, formatter_class=RawTextHelpFormatter)
    parser.add_argument('skymodel', help='Filename of input sky model')
    parser.add_argument('h5parm_file', help='Filename of h5parm with cooresponding solutions to update')
    parser.add_argument('--solset_name', help='Name of solution set', type=str, default='sol000')
    args = parser.parse_args()
    main(args.skymodel, args.h5parm_file, solset_name=args.solset_name)
