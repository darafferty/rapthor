#!/usr/bin/env python3
"""
Script to concatenate the MS files from Linc for input to Rapthor
"""
import argparse
import sys
import subprocess
import casacore.tables as pt
import numpy as np


def concat_ms(msfiles, output_file):
    """
    Concatenate a number of Measurement Set files into one

    Parameters
    ----------
    msfiles : list of str
        List of paths to input Measurement Sets
    output_file : str
        Path to output Measurement Set

    Returns
    -------
    int : 0 if successfull; non-zero otherwise
    
    Raises
    ------
    ValueError
        If less than two input Measurement Sets are provided
    RuntimeError
        If one or more of the input Measurement Sets contain invalid data,
        or if DP3 encounters an error while concatenating.
    """
    if len(msfiles) < 2:
        raise ValueError("At least two input Measurement Set must be provided")
    # Order the files by frequency, filling any gaps with dummy files
    first = True
    nchans = 0
    freqs = []
    for ms in msfiles:
        # Get the frequency info
        sw = pt.table(ms + "::SPECTRAL_WINDOW", ack=False)
        freq = sw.col("REF_FREQUENCY")[0]
        if first:
            file_bandwidth = sw.col("TOTAL_BANDWIDTH")[0]
            nchans = sw.col("CHAN_WIDTH")[0].shape[0]
            chwidth = sw.col("CHAN_WIDTH")[0][0]
            first = False
        else:
            assert file_bandwidth == sw.col("TOTAL_BANDWIDTH")[0]
            assert nchans == sw.col("CHAN_WIDTH")[0].shape[0]
            assert chwidth == sw.col("CHAN_WIDTH")[0][0]
        sw.close()
        freqs.append(freq)
    freqlist = np.array(freqs)
    mslist = np.array(msfiles)
    sorted_ind = np.argsort(freqlist)
    freqlist = freqlist[sorted_ind]
    mslist = mslist[sorted_ind]
    freq_width = np.min(freqlist[1:] - freqlist[:-1])
    dp3_mslist = []
    dp3_freqlist = np.arange(
        np.min(freqlist), np.max(freqlist) + freq_width, freq_width
    )
    j = -1
    for freq, ms in zip(freqlist, mslist):
        while j < len(dp3_freqlist) - 1:
            j += 1
            if np.isclose(freq, dp3_freqlist[j]):
                # Frequency of MS file matches output frequency
                # Add the MS file to the list; break to move to the next MS file
                dp3_mslist.append(ms)
                break
            else:
                # Gap in frequency detected
                # Add a dummy MS to the list; stay on the current MS file
                dp3_mslist.append("dummy.ms")

    # Call DP3
    cmd = [
        "DP3",
        "msin=[{}]".format(",".join(dp3_mslist)),
        "msout={}".format(output_file),
        "steps=[]",
        "msout.overwrite=True",
        "msin.orderms=False",
        "msin.missingdata=True",
        "msout.writefullresflag=False",
        "msout.storagemanager=Dysco",
    ]
    try:
        return subprocess.run(cmd, check=True).returncode
    except subprocess.CalledProcessError as err:
        print(err, file=sys.stderr)
        return err.returncode


def main():
    """
    Concatentates two or more input Measurement Sets into one output Measurement Set.

    Returns
    -------
    int : 0 if successfull; non-zero otherwise.
    """
    descriptiontext = "Concatenate Measurement Sets.\n"
    parser = argparse.ArgumentParser(
        description=descriptiontext, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("msin", nargs="+", help="List of input Measurement Sets")
    parser.add_argument("msout", help="Output Measurement Set")

    args = parser.parse_args()
    return concat_ms(args.msin, args.msout)


if __name__ == "__main__":
    sys.exit(main())
