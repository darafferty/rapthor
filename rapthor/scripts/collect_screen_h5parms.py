#!/usr/bin/env python3
"""
Script to collect multiple H5parms containing screen solutions by concatenating
in time and frequency
"""
from argparse import ArgumentParser, RawTextHelpFormatter
from collections import namedtuple
import h5py
import logging
import numpy as np
import os


def main(h5parm_files, outh5parm_file, overwrite):
    """
    Collects multiple screen h5parms into a single h5parm by concatenating in time and
    frequency

    Parameters
    ----------
    h5parm_files : list
        Filenames of input screen h5parms. The h5parms should contain solutions with
        the same structure but for different times or frequencies
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
        itembasename = itemname.split("/")[-1]
        if itemname not in outh5parm:
            # Make a new entry for the output table
            if isinstance(item, h5py.Dataset):
                shape = item.shape
                if itembasename == "time":
                    shape = (shape[0] * nr_time_blocks,)
                if itembasename == "freq":
                    shape = (shape[0] * nr_frequency_blocks,)
                elif itembasename in ["val", "weight"]:
                    # Value and weight array axes are always [freq, ant, time, dir]
                    shape = (
                        shape[0] * nr_frequency_blocks,
                        shape[1],
                        shape[2] * nr_time_blocks,
                        shape[3],
                    )
                d = outh5parm.create_dataset_like(itemname, item, shape=shape)
                if "AXES" in item.attrs:
                    d.attrs["AXES"] = item.attrs["AXES"]
            elif isinstance(item, h5py.Group):
                # Create and populate group metadata
                g = outh5parm.create_group(itemname)
                if "TITLE" in item.attrs:
                    g.attrs["TITLE"] = item.attrs["TITLE"]
                if "h5parmversion" in item.attrs:
                    g.attrs["h5parmversion"] = item.attrs["h5parmversion"]
                else:
                    g.attrs['h5parm_version'] = 1.0
        if isinstance(item, h5py.Dataset):
            # Update the output table by filling in the appropriate time and/or
            # frequency ranges
            if itembasename == "time":
                t_slice = slice(i * item.shape[0], (i + 1) * item.shape[0])
                outh5parm[itemname][t_slice] = item[:]
            elif itembasename == "freq":
                f_slice = slice(j * item.shape[0], (j + 1) * item.shape[0])
                outh5parm[itemname][f_slice] = item[:]
            elif itembasename in ["val", "weight"]:
                t_slice = slice(i * item.shape[2], (i + 1) * item.shape[2])
                f_slice = slice(j * item.shape[0], (j + 1) * item.shape[0])
                outh5parm[itemname][f_slice, :, t_slice, :] = item[:]
            else:
                outh5parm[itemname][:] = item[:]

    # Define the visitor function used to get the minimum frequency
    def get_min_frequency(itemname, item):
        if 'freq' in itemname:
            return np.min(item)

    # Define the visitor function used to get the minimum time
    def get_min_time(itemname, item):
        if 'time' in itemname:
            return np.min(item)

    # Determine frequency and time chunks present in the input files
    frequencies = []
    times = []
    chunks = []
    Chunk = namedtuple('Chunk', ['filename', 'time', 'freq'])
    for h5parm_file in h5parm_files:
        with h5py.File(h5parm_file, "r") as h5parm:
            frequencies.append(h5parm.visititems(get_min_frequency))
            times.append(h5parm.visititems(get_min_time))
            chunks.append(Chunk(h5parm_file, times[-1], frequencies[-1]))
    frequencies = sorted(set(frequencies))
    times = sorted(set(times))
    nr_time_blocks = len(times)
    nr_frequency_blocks = len(frequencies)

    # Collect values from each chunk and place them in the output
    for i, time in enumerate(times):
        for j, freq in enumerate(frequencies):
            chunk = [c for c in chunks if (c.time == time and c.freq == freq)][0]
            with h5py.File(chunk.filename, "r") as h5parm:
                h5parm.visititems(visitor)

    # Close the output file
    outh5parm.close()


if __name__ == '__main__':
    descriptiontext = "Collect multiple screen h5parms in time and frequency.\n"

    parser = ArgumentParser(
        description=descriptiontext, formatter_class=RawTextHelpFormatter
    )
    parser.add_argument("h5parm_files", nargs="+", help="List of input h5parms")
    parser.add_argument("num_time_chunks", help="Number of time chunks ")
    parser.add_argument(
        "--outh5parm",
        "-o",
        default="output.h5",
        dest="outh5parm",
        help="Output h5parm name [default: output.h5]",
    )
    parser.add_argument(
        "--overwrite",
        "-c",
        default=False,
        action="store_true",
        help="Replace exising outh5parm file instead of appending to it (default=False)",
    )
    args = parser.parse_args()

    try:
        if len(args.h5parm_files) == 1 and (
            "," in args.h5parm_files[0]
            or ("[" in args.h5parm_files[0] and "]" in args.h5parm_files[0])
        ):
            # Treat input as a string with comma-separated values
            args.h5parm_files = args.h5parm_files[0].strip("[]").split(",")
            args.h5parm_files = [f.strip() for f in args.h5parm_files]

        main(args.h5parm_files, args.outh5parm, args.overwrite)
    except ValueError as e:
        log = logging.getLogger('rapthor:collect_screen_h5parms')
        log.critical(e)
