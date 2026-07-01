"""Measurement Set helpers for adding sector model data."""

import logging
from typing import Optional

import casacore.tables as pt

from rapthor.execution.outputs import output_path
from rapthor.execution.predict.measurement_sets import (
    copy_measurement_set,
    input_rows_for_models,
    modeldata_output_stem,
    plan_row_chunks,
    predict_chunk_count,
    read_model_data,
    select_models_for_frequency,
    select_models_for_starttime,
    sum_model_data,
)
from rapthor.lib import miscellaneous as misc

log = logging.getLogger("rapthor:predict:sector_model_addition")


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
    return predict_chunk_count(
        msin,
        nsectors,
        fraction=fraction,
        compressed=compressed,
    )


def add_sector_models(
    msin,
    msmod_list,
    msin_column="DATA",
    model_column="DATA",
    out_column="MODEL_DATA",
    use_compression=False,
    starttime=None,
    quiet=True,
    infix="",
    output_dir: Optional[str] = None,
):
    """
    Add sector model data.

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
    output_dir : str, optional
        Directory for generated Measurement Set outputs. If omitted, outputs
        are written relative to the current working directory for CLI parity.
    """
    use_compression = misc.string2bool(use_compression)
    model_list = misc.string2list(msmod_list)

    # Get the model data filenames, filtering any that do not have the right start time.
    model_selection = select_models_for_starttime(model_list, starttime)
    model_list = select_models_for_frequency(
        msin,
        model_selection.paths,
        spectral_window_separator="::",
    )
    nsectors = len(model_list)
    if nsectors == 0:
        raise ValueError("No model data found.")
    log.info("Found %s model data files", nsectors)

    # Define the template MS file. This file is copied to one or more files
    # to be filled with new data
    ms_template = model_list[0]

    # If starttime is given, figure out startrow and nrows for input MS file
    tin = pt.table(msin, readonly=True, ack=False)
    input_rows = input_rows_for_models(
        tin,
        starttime=starttime,
        starttime_exact=model_selection.starttime_exact,
        model_nrows=model_selection.nrows,
    )

    # Define chunks based on available memory, making sure each
    # chunk gives a full timeslot (needed for reweighting)
    fraction = float(input_rows.nrows) / float(tin.nrows())
    nchunks = get_nchunks(msin, nsectors, fraction)
    chunks = plan_row_chunks(
        nrows=input_rows.nrows,
        nchunks=nchunks,
        baseline_rows=input_rows.baseline_rows,
    )
    log.info("Using %s chunk(s)", len(chunks))

    # Open output table and add output column if needed
    msout = output_path(
        output_dir,
        modeldata_output_stem(model_list[0]) + "_di.ms",
    )
    copy_measurement_set(ms_template, msout)
    tout = pt.table(msout, readonly=False, ack=False)
    if out_column not in tout.colnames():
        desc = tout.getcoldesc("DATA")
        desc["name"] = out_column
        tout.addcols(desc)

    # Copy the DATA column from the input MS file to the output one
    data = tin.getcol("DATA", startrow=input_rows.startrow, nrow=input_rows.nrows)
    tout.putcol("DATA", data, startrow=0, nrow=input_rows.nrows)
    tout.flush()

    # Process the data chunk by chunk
    for chunk in chunks:
        # For each chunk, load data
        datamod_list = read_model_data(
            model_list,
            model_column,
            startrow=chunk.model_startrow,
            nrows=chunk.nrows,
        )

        # Sum model data for this chunk over all sectors
        datamod_all = sum_model_data(datamod_list)
        tout.putcol(out_column, datamod_all, startrow=chunk.model_startrow, nrow=chunk.nrows)
        tout.flush()
    tout.close()
    tin.close()
