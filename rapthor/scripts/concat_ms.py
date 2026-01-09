#!/usr/bin/env python3
"""
Script to concatenate MS files
"""
from argparse import ArgumentParser, RawTextHelpFormatter
import os
import shutil
import subprocess
import sys

import casacore.tables as pt
import numpy as np


def concat_ms(msfiles, output_file, data_colname='DATA', concat_property="frequency", overwrite=False):
    """
    Concatenate a number of Measurement Set files into one

    Parameters
    ----------
    msfiles : list of str
        List of paths to input Measurement Sets
    output_file : str
        Path to output Measurement Set
    concat_property : str, optional
        Property over which to concatenate: time or frequency. Note that,
        when concatenating over time, the metadata of the output MS file
        may not be correct. However, it will work correctly with WSClean
    overwrite : bool, optional
        If True and output_file points to an existing file, the file is
        overwritten

    Returns
    -------
    int : 0 if successfull; non-zero otherwise

    Raises
    ------
    TypeError
        If input Measurement Sets are not provided as a list of strings.
    ValueError
        If no input Measurement Sets are provided or concat_property is
        invalid.
    RuntimeError
        If one or more of the input Measurement Sets contain invalid data,
        or if DP3 encounters an error while concatenating.
    """
    # Check pre-conditions
    if not isinstance(msfiles, list) or not all(
        isinstance(item, str) for item in msfiles
    ):
        raise TypeError("Input Measurement Sets must provided as a list of strings")
    if len(msfiles) == 0:
        raise ValueError("At least one input Measurement Set must be provided")
    if concat_property.lower() not in ("frequency", "time"):
        raise ValueError("concat_property must be one of 'time' or 'frequency'.")
    if os.path.exists(output_file):
        for msfile in msfiles:
            if os.path.samefile(msfile, output_file):
                raise ValueError("Input Measurement Set '{0}' and output Measurement Set '{1}' "
                                 "are the same file".format(msfile, output_file))
        if overwrite:
            shutil.rmtree(output_file)
        else:
            raise FileExistsError("The output Measurement Set exists and overwrite=False")

    # Construct the command to run depending on what's needed. It will be executed
    # later
    if len(msfiles) > 1:
        if concat_property.lower() == "frequency":
            cmd = concat_freq_command(msfiles, data_colname, output_file)
        elif concat_property.lower() == "time":
            cmd = concat_time_command(msfiles, output_file)
    else:
        # Single input file -- just copy to output
        cmd = [
            "cp",
            "-r",
            "-L",
            "--no-preserve=mode",
            msfiles[0],
            output_file
        ]

    # Run the command
    try:
        return subprocess.run(cmd, check=True).returncode
    except subprocess.CalledProcessError as err:
        print(err, file=sys.stderr)
        return err.returncode


def concat_freq_command(msfiles, data_colname, output_file, make_dummies=True):
    """
    Construct command to concatenate files in frequency using DP3

    Parameters
    ----------
    msfiles : list of str
        List of MS filenames to be concatenated
    output_file : str
        Filename of output concatenated MS
    make_dummies: bool
        Insert dummy MSes when frequency gaps are detected.

    Returns
    -------
    cmd : list of str
        Command to be executed by subprocess.run()
    """
    # Order the files by frequency, filling any gaps with dummy files
    first = True
    freqs = []
    chfreqs_tmp = []
    for ms in msfiles:
        # Get the frequency info
        with pt.table(ms + "::SPECTRAL_WINDOW", ack=False) as sw:
            freq = sw.col("CHAN_FREQ")[0][0]
            chfreqs = sw.col("CHAN_FREQ")[0]
            if first:
                file_bandwidth = sw.col("TOTAL_BANDWIDTH")[0]
                chwidth = sw.col("CHAN_WIDTH")[0][0]
                first = False
            else:
                assert file_bandwidth == sw.col("TOTAL_BANDWIDTH")[0]
                assert chwidth == sw.col("CHAN_WIDTH")[0][0]
                # We used to also check here the number of channels using
                # assert nchans == sw.col("CHAN_WIDTH")[0].shape[0], but
                # the last measurement set from LINC can have fewer channels.
            chfreqs_tmp.extend(chfreqs)
        freqs.append(freq)

    freqlist = np.array(freqs)
    chfreqlist = sorted(np.array(chfreqs_tmp))
    mslist = np.array(msfiles)
    sorted_ind = np.argsort(freqlist)
    freqlist = freqlist[sorted_ind]
    mslist = mslist[sorted_ind]

    # Check for gaps in frequency coverage by looking for deviating channel widths.
    # Originally by Jurjen de Jong.
    # Adapted from https://github.com/jurjen93/lofar_vlbi_helpers/blob/main/extra_scripts/check_missing_freqs_in_ms.py
    chan_diff = np.abs(np.diff(chfreqlist, n=2))
    if np.sum(chan_diff) != 0:
        # Here we obtain the indices at which dummies should be inserted.
        # The division maps this back from individual channels to MSes.
        dummy_idx = (np.ndarray.flatten(np.argwhere(chan_diff > 0))/len(chan_diff)*len(mslist)).round(0).astype(int)
        # Obtain a unique list of indices and the indices of their first occurence in the above original array.
        dummy_idx_u, idx_idx_u = np.unique(dummy_idx, return_index=True)
        # Express channel difference in data chunks (e.g. 2 MHz if from LINC).
        # This has the same length as dummy_idx and will tell us how many dummies to insert.
        # For example, if there is a 8 MHz gap between 2 MHz chunks we need 3 dummy entries,
        # otherwise the output will not being regular.
        dummy_multiplier = (chan_diff / file_bandwidth).astype(int)
        # We only need dummies where the difference is non-zero.
        # This will yield as many entries as in dummy_idx
        dummy_multiplier = dummy_multiplier[dummy_multiplier > 0]
        if dummy_multiplier.size != 0:
            # Select entries corresponding to the gaps
            dummy_multiplier = dummy_multiplier[idx_idx_u]
            # dummy_multiplier now contains the number of dummies that need to be inserted.
            # This list needs to be flat for NumPy's insert later on.
            dummies = [['dummy.ms'] * x for x in dummy_multiplier]
            dummies_flat = [i for d in dummies for i in d]
            # Generate the indices at which each dummy needs to be inserted.
            final_idx = [[dummy_idx_u[i]] * len(dummies[i]) for i in range(len(dummies))]
            final_idx_flat = [i for f in final_idx for i in f]
            # Finally insert them all at once.
            mslist = np.insert(mslist, final_idx_flat, dummies_flat)

    # Construct DP3 command
    cmd = [
        "DP3",
        "msin=[{}]".format(",".join(mslist)),
        "msin.datacolumn={}".format(data_colname),
        "msout={}".format(output_file),
        "steps=[]",
        "msin.orderms=False",
        "msin.missingdata=True",
        "msout.writefullresflag=False",
        "msout.storagemanager=Dysco",
    ]
    return cmd


def concat_time_command(msfiles, output_file):
    """
    Construct command to concatenate files in time using TAQL

    Parameters
    ----------
    msfiles : list of str
        List of MS filenames to be concatenated
    output_file : str
        Filename of output concatenated MS

    Returns
    -------
    cmd : list of str
        Command to be executed by subprocess.run()
    """
    cmd = [
        "taql",
        "select",
        "from",
        "[{}]".format(",".join(msfiles)),
        "giving",
        "{}".format(output_file),
        "AS",
        "PLAIN"
    ]
    return cmd


def main():
    """
    Concatentates two or more input Measurement Sets into one output Measurement Set.

    Returns
    -------
    int : 0 if successfull; non-zero otherwise.
    """
    descriptiontext = "Concatenate Measurement Sets.\n"
    parser = ArgumentParser(
        description=descriptiontext, formatter_class=RawTextHelpFormatter
    )
    parser.add_argument("msin", nargs="+", help="List of input Measurement Sets")
    parser.add_argument("--data_colname", help="Data column to be concatenated", type=str, default='DATA')
    parser.add_argument("--msout", help="Output Measurement Set", type=str, default='concat.ms')
    parser.add_argument('--concat_property', help='Property over which to concatenate: time or frequency',
                        type=str, default='frequency')
    parser.add_argument('--overwrite', help='Overwrite existing output file', type=bool, default=False)

    args = parser.parse_args()
    return concat_ms(args.msin, args.msout, args.data_colname, concat_property=args.concat_property, overwrite=args.overwrite)


if __name__ == "__main__":
    sys.exit(main())
