"""Measurement Set concatenation helpers."""

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Union

import casacore.tables as pt
import numpy as np

log = logging.getLogger("rapthor:concatenate:measurement_sets")

_PathInput = Union[str, os.PathLike]


def linc_measurement_sets(input_path: _PathInput) -> list[str]:
    """Return LINC Measurement Sets in ``input_path`` in a deterministic order."""
    input_dir = Path(input_path)
    return sorted(str(path) for pattern in ("*.ms", "*.MS") for path in input_dir.glob(pattern))


def concat_linc_measurement_sets(
    input_path: _PathInput,
    output_file: _PathInput,
    overwrite: bool = False,
) -> int:
    """
    Concatenate LINC Measurement Sets from a directory for input to Rapthor.

    Both lower-case ``*.ms`` and upper-case ``*.MS`` directory names are
    accepted to match historical LINC output naming.
    """
    return concat_ms(
        linc_measurement_sets(input_path),
        str(output_file),
        overwrite=overwrite,
    )


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
    command = select_concatenation_command(
        msfiles,
        output_file,
        data_colname=data_colname,
        concat_property=concat_property,
        overwrite=overwrite,
    )

    try:
        return subprocess.run(command, check=True).returncode
    except subprocess.CalledProcessError as err:
        log.error("Measurement Set concatenation command failed: %s", err)
        return err.returncode


def select_concatenation_command(
    msfiles: list[str],
    output_file: str,
    data_colname: str = "DATA",
    concat_property: str = "frequency",
    overwrite: bool = False,
) -> list[str]:
    """
    Validate inputs and choose the external command for Measurement Set concatenation.

    Frequency concatenation uses DP3, time concatenation uses TAQL, and a single
    input Measurement Set is copied to the output path.
    """
    _validate_concat_inputs(msfiles, output_file, concat_property, overwrite)

    if len(msfiles) == 1:
        return copy_measurement_set_command(msfiles[0], output_file)
    if concat_property.lower() == "frequency":
        return concat_freq_command(msfiles, data_colname, output_file)
    return concat_time_command(msfiles, output_file)


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
    ordered_msfiles = _time_ordered_ms_list(msfiles)
    ms_list = ",".join(_taql_string(msfile) for msfile in ordered_msfiles)
    return [
        "taql",
        "select",
        "from",
        f"[{ms_list}]",
        "giving",
        _taql_string(output_file),
        "AS",
        "PLAIN",
    ]


def copy_measurement_set_command(input_file: str, output_file: str) -> list[str]:
    """Build the copy command used for a single Measurement Set input."""
    return ["cp", "-r", "-L", "--no-preserve=mode", input_file, output_file]


def _taql_string(value: object) -> str:
    """Return a quoted TaQL string literal for table and output paths."""
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'


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


def _frequency_ordered_ms_list(msfiles: list[str]) -> np.ndarray:
    """Return Measurement Sets ordered by their first channel frequency."""
    frequencies = []
    for msfile in msfiles:
        with pt.table(f"{msfile}::SPECTRAL_WINDOW", ack=False) as spectral_window:
            frequencies.append(spectral_window.col("CHAN_FREQ")[0][0])

    mslist = np.array(msfiles)
    return mslist[np.argsort(np.array(frequencies))]


def _time_ordered_ms_list(msfiles: list[str]) -> list[str]:
    """Return Measurement Sets ordered by their first TIME value."""
    start_times = []
    for msfile in msfiles:
        with pt.table(msfile, ack=False) as table:
            start_times.append(float(np.min(table.getcol("TIME"))))
    return [msfile for _, _, msfile in sorted(zip(start_times, range(len(msfiles)), msfiles))]


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
