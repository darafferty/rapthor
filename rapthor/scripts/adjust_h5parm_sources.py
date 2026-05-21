#!/usr/bin/env python3
"""
Script to adjust h5parm source coordinates to match those in the sky model
"""
from argparse import ArgumentParser, RawTextHelpFormatter
import lsmtool
from losoto.h5parm import h5parm
import numpy as np
import sys
from lsmtool.operations_lib import normalize_ra_dec

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
        source_names = []
        source_positions = []
        for skymodel_source_name, skymodel_source_position in source_dict.items():
            source_names.append(f'[{skymodel_source_name}]')
            ra, dec = normalize_ra_dec(skymodel_source_position[0].value,
                                       skymodel_source_position[1].value)
            source_positions.append([ra, dec])

        soltab = solset.getSoltabs()[0]  # take the first soltab (all have the same directions)
        axes_names = soltab.getAxesNames()
        if 'dir' not in axes_names:
            print(f'The solutions in solution set {solset_name} of the input H5parm file are '
                  'direction-independent. The solutions will be duplicated for all directions')
        else:
            soltab_sources = [str(source) for source in soltab.getAxisValues('dir')]
            if soltab_sources != source_names:
                if len(soltab_sources) == 1:
                    print(f'The solution directions in solution set {solset_name} do not match '
                          'the sky model. The single direction will be duplicated for all '
                          'sky-model directions')
                elif len(soltab_sources) != len(source_names):
                    sys.exit('ERROR: The patches in the sky model and the directions in the h5parm '
                             'must have the same length')
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
        sourceTable.append(list(zip(*(source_names, vals))))

        for soltab_name in list(solset.getSoltabNames()):
            soltab = solset.getSoltab(soltab_name)
            vals, _ = soltab.getValues()
            weights, _ = soltab.getValues(weight=True)
            soltab_type = soltab.getType()
            axes_names = list(soltab.getAxesNames())
            axes_vals = [soltab.getAxisValues(axis_name) for axis_name in axes_names]

            if 'dir' in axes_names:
                dir_ind = axes_names.index('dir')
                old_dirs = [str(source) for source in axes_vals[dir_ind]]
                if old_dirs == source_names:
                    continue
                if len(old_dirs) != 1 and len(old_dirs) != len(source_names):
                    sys.exit('ERROR: A direction is present in the h5parm that is not in the sky model')
                source_vals = np.take(vals, 0, axis=dir_ind) if len(old_dirs) == 1 else vals
                source_weights = np.take(weights, 0, axis=dir_ind) if len(old_dirs) == 1 else weights
                new_shape = list(vals.shape)
                new_shape[dir_ind] = len(source_names)
                axes_vals[dir_ind] = source_names
            else:
                dir_ind = len(axes_names)
                source_vals = vals
                source_weights = weights
                new_shape = list(vals.shape) + [len(source_names)]
                axes_names.append('dir')
                axes_vals.append(source_names)

            new_vals = np.zeros(new_shape, dtype=vals.dtype)
            new_weights = np.zeros(new_shape, dtype=weights.dtype)
            for i in range(len(source_names)):
                slc = [slice(None)] * len(new_shape)
                slc[dir_ind] = i
                new_vals[tuple(slc)] = source_vals
                new_weights[tuple(slc)] = source_weights
            soltab.delete()
            solset.makeSoltab(soltab_type, soltab_name, axesNames=axes_names,
                              axesVals=axes_vals, vals=new_vals, weights=new_weights)


if __name__ == '__main__':
    descriptiontext = "Adjust the h5parm source coordinates to match those in the sky model.\n"

    parser = ArgumentParser(description=descriptiontext, formatter_class=RawTextHelpFormatter)
    parser.add_argument('skymodel', help='Filename of input sky model')
    parser.add_argument('h5parm_file', help='Filename of h5parm with cooresponding solutions to update')
    parser.add_argument('--solset_name', help='Name of solution set', type=str, default='sol000')
    args = parser.parse_args()
    main(args.skymodel, args.h5parm_file, solset_name=args.solset_name)
