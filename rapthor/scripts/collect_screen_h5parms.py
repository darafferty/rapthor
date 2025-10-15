#!/usr/bin/env python3
"""
Script to collect multiple H5parms containing screen solutions by concatenating
in time
"""
from argparse import ArgumentParser, RawTextHelpFormatter
import h5py
import logging
import os


def main(h5parm_files, outh5parm_file, overwrite):
    """
    Collects multiple screen h5parms into a single h5parm by concatenating in
    time

    Parameters
    ----------
    h5parm_files : list
        Filenames of input screen h5parms. The h5parms should contian solutions
        with the same structure but for different times
    outh5parm_file : str
        Filename of the output h5parm
    overwrite : bool
        If True, overwrite existing output H5parm file
    """
    # Create output file
    if os.path.exists(outh5parm_file):
        if overwrite:
            os.remove(outh5parm_file)
        else:
            raise FileExistsError("The output H5parm file exists and overwrite=False")
    outh5parm = h5py.File(outh5parm_file, "w")

    # Define the visitor function used to populate the output h5 file
    def visitor(itemname, item):
        itembasename = itemname.split('/')[-1]
        if itemname not in outh5parm:
            # Make a new entry for the table
            if isinstance(item, h5py.Dataset):
                # Create a dummy entry with the required shape and then fill
                # with item values. The size of the time axis is set as
                # “unlimited” (using None in maxshape), as its final size will
                # be determined later by the concatenation process done below
                maxshape = item.maxshape
                if itembasename == 'time':
                    maxshape = (None,)
                elif itembasename in ['val', 'weight']:
                    # Time axis has index 2
                    maxshape = (maxshape[0], maxshape[1], None, maxshape[3])
                d = outh5parm.create_dataset_like(itemname, item, chunks=item.shape if all(item.shape) else None, maxshape=maxshape)
                if 'AXES' in item.attrs:
                    d.attrs['AXES'] = item.attrs['AXES']
                d.resize(item.shape)
                d[:] = item[:]
            elif isinstance(item, h5py.Group):
                # Create and populate group metadata
                g = outh5parm.create_group(itemname)
                if 'TITLE' in item.attrs:
                    g.attrs['TITLE'] = item.attrs['TITLE']
                if 'h5parm_version' in item.attrs:
                    g.attrs['h5parm_version'] = item.attrs['h5parm_version']
                else:
                    g.attrs['h5parm_version'] = 1.0
        else:
            # Update the existing entry by expanding (concatenating) it in
            # time. This involves adjusting the shape of the time axis to
            # account for the new values
            if itembasename == 'time':
                new_shape = outh5parm[itemname].shape
                new_shape = (new_shape[0] + item.shape[0],)
                outh5parm[itemname].resize(new_shape)
            elif itembasename in ['val', 'weight']:
                # Time axis has index 2
                new_shape = outh5parm[itemname].shape
                new_shape = (new_shape[0], new_shape[1], new_shape[2] + item.shape[2], new_shape[3])
                outh5parm[itemname].resize(new_shape)
            if itembasename in ['time', 'val', 'weight']:
                slicer = tuple(slice(-i, None) for i in item.shape)
                outh5parm[itemname][slicer] = item[:]

    # Collect values from each input file and place them in the output
    for h5parm_file in h5parm_files:
        h5parm = h5py.File(h5parm_file, "r")
        h5parm.visititems(visitor)
        h5parm.close()
    outh5parm.close()


if __name__ == '__main__':
    descriptiontext = "Collect multiple screen h5parms in time.\n"

    parser = ArgumentParser(description=descriptiontext, formatter_class=RawTextHelpFormatter)
    parser.add_argument('h5parm_files', nargs='+', help='List of input h5parms')
    parser.add_argument('--outh5parm', '-o', default='output.h5', dest='outh5parm', help='Output h5parm name [default: output.h5]')
    parser.add_argument('--overwrite', '-c', default=False, action='store_true', help='Replace exising outh5parm file instead of appending to it (default=False)')
    args = parser.parse_args()

    try:
        if len(args.h5parm_files) == 1 and (',' in args.h5parm_files[0] or
                                            ('[' in args.h5parm_files[0] and
                                             ']' in args.h5parm_files[0])):
            # Treat input as a string with comma-separated values
            args.h5parm_files = args.h5parm_files[0].strip('[]').split(',')
            args.h5parm_files = [f.strip() for f in args.h5parm_files]

        main(args.h5parm_files, args.outh5parm, args.overwrite)
    except ValueError as e:
        log = logging.getLogger('rapthor:collect_screen_h5parms')
        log.critical(e)
