#!/usr/bin/env python3
"""
Script to add sector model data
"""
import os
import shutil
import subprocess
from argparse import ArgumentParser, RawTextHelpFormatter

import casacore.tables as pt
import numpy as np
from rapthor.lib import miscellaneous as misc


def get_nchunks(msin, nsectors, fraction=1.0, compressed=False):
    """
    Determines number of chunks for available memory of node

    Parameters
    ----------
    msin : str
        Input MS file name
    nsectors : int
        Number of imaging sectors
    fraction : float
        Fraction of MS file to be read
    compressed: bool
        True if data are compressed (by Dysco)

    Returns
    -------
    nchunks : int
        Number of chunks
    """
    scale_factor = 4.0
    if compressed:
        scale_factor *= 5.0
    tot_m, used_m, free_m = list(map(int, os.popen('free -tm').readlines()[-1].split()[1:]))
    msin_m = float(subprocess.check_output(['du', '-smL', msin]).split()[0]) * fraction
    tot_required_m = msin_m * nsectors * scale_factor * 2.0
    nchunks = max(1, int(np.ceil(tot_required_m / tot_m)))
    return nchunks


def main(msin, msmod_list, msin_column='DATA', model_column='DATA',
         out_column='MODEL_DATA', use_compression=False,
         starttime=None, quiet=True, infix=''):
    """
    Add sector model data

    Parameters
    ----------
    msin : str
        Name of MS file to which model data will be added
    msmod_list: list
        List of model data MS filenames
    msin_column : str, optional
        Name of input column
    model_column : str, optional
        Name of input model column
    out_column : str, optional
        Name of output column for summed model data
    use_compression : bool, optional
        If True, use Dysco compression on DATA column
    starttime : str, optional
        Start time in JD seconds
    quiet : bool
        If True, suppress (most) output
    infix : str, optional
        Infix string used in filenames
    """
    use_compression = misc.string2bool(use_compression)
    model_list = misc.string2list(msmod_list)

    # Get the model data filenames, filtering any that do not have the right start time
    if starttime is not None:
        # Filter the list of models to include only ones for the given times
        nrows_list = []
        for msmod in model_list[:]:
            tin = pt.table(msmod, readonly=True, ack=False)
            starttime_chunk = np.min(tin.getcol('TIME'))
            if not misc.approx_equal(starttime_chunk, misc.convert_mvt2mjd(starttime), tol=1.0):
                # Remove files with start times that are not within 1 sec of the
                # specified starttime
                i = model_list.index(msmod)
                model_list.pop(i)
            else:
                nrows_list.append(tin.nrows())
                starttime_exact = misc.convert_mjd2mvt(starttime_chunk)  # store exact value for use later
            tin.close()
        if len(set(nrows_list)) > 1:
            raise RuntimeError('Model data files have differing number of rows...')
    # In case the user did not concatenate LINC output and fed multiple frequency bands, find the correct frequency band.
    chan_freqs = pt.table(msin+"/SPECTRAL_WINDOW").getcol("CHAN_FREQ")
    for model_ms in model_list[:]:
        chan_freqs_model = pt.table(model_ms+"/SPECTRAL_WINDOW").getcol("CHAN_FREQ")
        if not np.allclose(chan_freqs_model, chan_freqs):
            i = model_list.index(model_ms)
            model_list.pop(i)
    nsectors = len(model_list)
    if nsectors == 0:
        raise ValueError('No model data found.')
    print('add_sector_models: Found {} model data files'.format(nsectors))

    # Define the template MS file. This file is copied to one or more files
    # to be filled with new data
    ms_template = model_list[0]

    # If starttime is given, figure out startrow and nrows for input MS file
    tin = pt.table(msin, readonly=True, ack=False)
    tarray = tin.getcol("TIME")
    nbl = np.where(tarray == tarray[0])[0].size
    tarray = None
    if starttime is not None:
        tapprox = misc.convert_mvt2mjd(starttime_exact) - 100.0
        approx_indx = np.where(tin.getcol('TIME') > tapprox)[0][0]
        for tind, t in enumerate(tin.getcol('TIME')[approx_indx:]):
            if misc.convert_mjd2mvt(t) == starttime_exact:
                startrow_in = tind + approx_indx
                break
        nrows_in = nrows_list[0]
    else:
        startrow_in = 0
        nrows_in = tin.nrows()

    # Define chunks based on available memory, making sure each
    # chunk gives a full timeslot (needed for reweighting)
    fraction = float(nrows_in) / float(tin.nrows())
    nchunks = get_nchunks(msin, nsectors, fraction)
    nrows_per_chunk = int(nrows_in / nchunks)
    while nrows_per_chunk % nbl > 0.0:
        nrows_per_chunk -= 1
        if nrows_per_chunk < nbl:
            nrows_per_chunk = nbl
            break
    nchunks = int(np.ceil(nrows_in / nrows_per_chunk))
    startrows_tmod = [0]
    nrows = [nrows_per_chunk]
    for i in range(1, nchunks):
        if i == nchunks-1:
            nrow = nrows_in - (nchunks - 1) * nrows_per_chunk
        else:
            nrow = nrows_per_chunk
        nrows.append(nrow)
        startrows_tmod.append(startrows_tmod[i-1] + nrows[i-1])
    print('add_sector_models: Using {} chunk(s)'.format(nchunks))

    # Open output table and add output column if needed
    msout = os.path.basename(model_list[0]).removesuffix('_modeldata') + '_di.ms'
    if os.path.exists(msout):
        # File may exist from a previous iteration; delete it if so
        shutil.rmtree(msout, ignore_errors=True)
    subprocess.check_call(['cp', '-r', '-L', '--no-preserve=mode', ms_template, msout])
    tout = pt.table(msout, readonly=False, ack=False)
    if out_column not in tout.colnames():
        desc = tout.getcoldesc('DATA')
        desc['name'] = out_column
        tout.addcols(desc)

    # Copy the DATA column from the input MS file to the output one
    data = tin.getcol('DATA', startrow=startrow_in, nrow=nrows_in)
    tout.putcol('DATA', data, startrow=0, nrow=nrows_in)
    tout.flush()

    # Process the data chunk by chunk
    for c, (startrow_tmod, nrow) in enumerate(zip(startrows_tmod, nrows)):
        # For each chunk, load data
        datamod_list = []
        for i, msmodel in enumerate(model_list):
            tmod = pt.table(msmodel, readonly=True, ack=False)
            datamod_list.append(tmod.getcol(model_column, startrow=startrow_tmod, nrow=nrow))
            tmod.close()

        # Sum model data for this chunk over all sectors
        datamod_all = None
        for i in range(nsectors):
            if datamod_all is None:
                datamod_all = datamod_list[i].copy()
            else:
                datamod_all += datamod_list[i]
        tout.putcol(out_column, datamod_all, startrow=startrow_tmod, nrow=nrow)
        tout.flush()
    tout.close()
    tin.close()


if __name__ == '__main__':
    descriptiontext = "Add sector model data.\n"

    parser = ArgumentParser(description=descriptiontext, formatter_class=RawTextHelpFormatter)
    parser.add_argument('msin', help='Filename of input MS data file')
    parser.add_argument('msmod', help='Filename of input MS model data file')
    parser.add_argument('--msin_column', help='Name of msin column', type=str, default='DATA')
    parser.add_argument('--model_column', help='Name of input model data column', type=str, default='DATA')
    parser.add_argument('--out_column', help='Name of output model data column', type=str, default='MODEL_DATA')
    parser.add_argument('--use_compression', help='Use compression', type=str, default='False')
    parser.add_argument('--starttime', help='Start time in MVT', type=str, default=None)
    parser.add_argument('--quiet', help='Quiet', type=str, default='True')
    parser.add_argument('--infix', help='Infix for output files', type=str, default='')
    args = parser.parse_args()

    main(args.msin, args.msmod, msin_column=args.msin_column,
         model_column=args.model_column, out_column=args.out_column,
         use_compression=args.use_compression,
         starttime=args.starttime, quiet=args.quiet, infix=args.infix)
