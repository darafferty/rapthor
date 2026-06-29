"""Measurement Set concatenation helpers."""

import os
import shutil
import subprocess
import sys

import casacore.tables as pt
import numpy as np


def concat_ms(
    msfiles: list[str],
    output_file: str,
    data_colname: str = "DATA",
    concat_property: str = "frequency",
    overwrite: bool = False,
) -> int:
    """
    Concatenate one or more Measurement Sets into ``output_file``.

    Frequency concatenation is delegated to DP3. Time concatenation is delegated
    to TAQL. A single input Measurement Set is copied to the output path.
    """
    _validate_concat_inputs(msfiles, output_file, concat_property, overwrite)

    if len(msfiles) > 1:
        if concat_property.lower() == "frequency":
            command = concat_freq_command(msfiles, data_colname, output_file)
        else:
            command = concat_time_command(msfiles, output_file)
    else:
        command = _copy_measurement_set_command(msfiles[0], output_file)

    try:
        return subprocess.run(command, check=True).returncode
    except subprocess.CalledProcessError as err:
        print(err, file=sys.stderr)
        return err.returncode


def concat_freq_command(
    msfiles: list[str],
    data_colname: str,
    output_file: str,
    make_dummies: bool = True,
) -> list[str]:
    """
    Build the DP3 command used to concatenate Measurement Sets in frequency.

    When ``make_dummies`` is true, dummy Measurement Set placeholders are
    inserted into the DP3 input list when frequency gaps are detected.
    """
    mslist = _frequency_ordered_ms_list(msfiles)
    if make_dummies:
        mslist = _insert_dummy_ms_for_frequency_gaps(msfiles, mslist)

    return [
        "DP3",
        f"msin=[{','.join(mslist)}]",
        f"msin.datacolumn={data_colname}",
        f"msout={output_file}",
        "steps=[]",
        "msin.orderms=False",
        "msin.missingdata=True",
        "msout.writefullresflag=False",
        "msout.storagemanager=Dysco",
    ]


def concat_time_command(msfiles: list[str], output_file: str) -> list[str]:
    """Build the TAQL command used to concatenate Measurement Sets in time."""
    return [
        "taql",
        "select",
        "from",
        f"[{','.join(msfiles)}]",
        "giving",
        str(output_file),
        "AS",
        "PLAIN",
    ]


def _validate_concat_inputs(
    msfiles: list[str],
    output_file: str,
    concat_property: str,
    overwrite: bool,
) -> None:
    """Validate concat inputs before external commands are run."""
    if not isinstance(msfiles, list) or not all(isinstance(item, str) for item in msfiles):
        raise TypeError("Input Measurement Sets must provided as a list of strings")
    if len(msfiles) == 0:
        raise ValueError("At least one input Measurement Set must be provided")
    if concat_property.lower() not in ("frequency", "time"):
        raise ValueError("concat_property must be one of 'time' or 'frequency'.")
    if not os.path.exists(output_file):
        return

    for msfile in msfiles:
        if os.path.samefile(msfile, output_file):
            raise ValueError(
                f"Input Measurement Set {msfile!r} and output Measurement Set {output_file!r} "
                "are the same file"
            )
    if overwrite:
        shutil.rmtree(output_file)
        return
    raise FileExistsError("The output Measurement Set exists and overwrite=False")


def _copy_measurement_set_command(input_file: str, output_file: str) -> list[str]:
    """Build the copy command used for a single Measurement Set input."""
    return ["cp", "-r", "-L", "--no-preserve=mode", input_file, output_file]


def _frequency_ordered_ms_list(msfiles: list[str]) -> np.ndarray:
    """Return Measurement Sets ordered by their first channel frequency."""
    frequencies = []
    for msfile in msfiles:
        with pt.table(f"{msfile}::SPECTRAL_WINDOW", ack=False) as spectral_window:
            frequencies.append(spectral_window.col("CHAN_FREQ")[0][0])

    mslist = np.array(msfiles)
    return mslist[np.argsort(np.array(frequencies))]


def _insert_dummy_ms_for_frequency_gaps(msfiles: list[str], mslist: np.ndarray) -> np.ndarray:
    """Insert ``dummy.ms`` placeholders where frequency gaps are detected."""
    file_bandwidth = None
    channel_width = None
    channel_frequencies = []
    for msfile in msfiles:
        with pt.table(f"{msfile}::SPECTRAL_WINDOW", ack=False) as spectral_window:
            if file_bandwidth is None:
                file_bandwidth = spectral_window.col("TOTAL_BANDWIDTH")[0]
                channel_width = spectral_window.col("CHAN_WIDTH")[0][0]
            else:
                assert file_bandwidth == spectral_window.col("TOTAL_BANDWIDTH")[0]
                assert channel_width == spectral_window.col("CHAN_WIDTH")[0][0]
            channel_frequencies.extend(spectral_window.col("CHAN_FREQ")[0])

    channel_diffs = np.abs(np.diff(sorted(np.array(channel_frequencies)), n=2))
    if np.sum(channel_diffs) == 0:
        return mslist

    dummy_indices = (
        (np.ndarray.flatten(np.argwhere(channel_diffs > 0)) / len(channel_diffs) * len(mslist))
        .round(0)
        .astype(int)
    )
    unique_dummy_indices, first_index_indices = np.unique(dummy_indices, return_index=True)
    dummy_multiplier = (channel_diffs / file_bandwidth).astype(int)
    dummy_multiplier = dummy_multiplier[dummy_multiplier > 0]
    if dummy_multiplier.size == 0:
        return mslist

    dummy_multiplier = dummy_multiplier[first_index_indices]
    dummies = [["dummy.ms"] * count for count in dummy_multiplier]
    flat_dummies = [dummy for group in dummies for dummy in group]
    final_indices = [
        [unique_dummy_indices[index]] * len(dummies[index]) for index in range(len(dummies))
    ]
    flat_final_indices = [index for group in final_indices for index in group]
    return np.insert(mslist, flat_final_indices, flat_dummies)
