"""Shared Measurement Set helpers for predict post-processing."""

import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional

import casacore.tables as pt
import numpy as np

from rapthor.lib import miscellaneous as misc


@dataclass(frozen=True)
class ModelSelection:
    """Model Measurement Sets selected for one input observation/time slice."""

    paths: list[str]
    nrows: list[int]
    starttime_exact: Optional[str]


@dataclass(frozen=True)
class InputRows:
    """Rows from the input Measurement Set matched to selected model rows."""

    startrow: int
    nrows: int
    baseline_rows: int


@dataclass(frozen=True)
class RowChunk:
    """Matching row ranges in the input and model Measurement Sets."""

    input_startrow: int
    model_startrow: int
    nrows: int


def predict_chunk_count(
    msin: str,
    nsectors: int,
    *,
    fraction: float = 1.0,
    scale_factor: float = 4.0,
    compressed: bool = False,
) -> int:
    """Return the number of chunks needed for memory-bounded model operations."""
    if compressed:
        scale_factor *= 5.0
    total_memory_mb = int(os.popen("free -tm").readlines()[-1].split()[1])
    ms_size_mb = float(subprocess.check_output(["du", "-smL", msin]).split()[0]) * fraction
    required_mb = ms_size_mb * nsectors * scale_factor * 2.0
    return max(1, int(np.ceil(required_mb / total_memory_mb)))


def select_models_for_starttime(model_paths: list[str], starttime: Optional[str]) -> ModelSelection:
    """Filter model Measurement Sets to the ones matching ``starttime``."""
    if starttime is None:
        return ModelSelection(list(model_paths), [], None)

    selected_paths = []
    nrows = []
    starttime_exact = None
    for model_path in model_paths:
        table = pt.table(model_path, readonly=True, ack=False)
        try:
            starttime_chunk = np.min(table.getcol("TIME"))
            if misc.approx_equal(starttime_chunk, misc.convert_mvt2mjd(starttime), tol=1.0):
                selected_paths.append(model_path)
                nrows.append(table.nrows())
                starttime_exact = misc.convert_mjd2mvt(starttime_chunk)
        finally:
            table.close()

    if len(set(nrows)) > 1:
        raise RuntimeError("Model data files have differing number of rows...")
    return ModelSelection(selected_paths, nrows, starttime_exact)


def select_models_for_frequency(
    msin: str,
    model_paths: list[str],
    *,
    spectral_window_separator: str,
) -> list[str]:
    """Keep only model Measurement Sets with the same channel frequencies as ``msin``."""
    chan_freqs = _read_channel_frequencies(msin, spectral_window_separator)
    return [
        model_path
        for model_path in model_paths
        if np.allclose(
            _read_channel_frequencies(model_path, spectral_window_separator),
            chan_freqs,
        )
    ]


def input_rows_for_models(
    input_table,
    *,
    starttime: Optional[str],
    starttime_exact: Optional[str],
    model_nrows: list[int],
) -> InputRows:
    """Find the input MS row range corresponding to the selected model rows."""
    times = input_table.getcol("TIME")
    baseline_rows = np.where(times == times[0])[0].size
    if starttime is None:
        return InputRows(startrow=0, nrows=input_table.nrows(), baseline_rows=baseline_rows)

    if starttime_exact is None or not model_nrows:
        raise ValueError("No model data found.")

    approx_start_time = misc.convert_mvt2mjd(starttime_exact) - 100.0
    approx_index = np.where(input_table.getcol("TIME") > approx_start_time)[0][0]
    for offset, time in enumerate(input_table.getcol("TIME")[approx_index:]):
        if misc.convert_mjd2mvt(time) == starttime_exact:
            return InputRows(
                startrow=offset + approx_index,
                nrows=model_nrows[0],
                baseline_rows=baseline_rows,
            )

    raise ValueError(f"Input Measurement Set does not contain start time {starttime_exact}")


def plan_row_chunks(
    *,
    nrows: int,
    nchunks: int,
    input_startrow: int = 0,
    baseline_rows: Optional[int] = None,
) -> list[RowChunk]:
    """Build row chunks, optionally aligning them to complete timeslots."""
    nrows_per_chunk = int(nrows / nchunks)
    if baseline_rows is not None:
        while nrows_per_chunk % baseline_rows > 0.0:
            nrows_per_chunk -= 1
            if nrows_per_chunk < baseline_rows:
                nrows_per_chunk = baseline_rows
                break
        nchunks = int(np.ceil(nrows / nrows_per_chunk))

    chunks = []
    next_input_startrow = input_startrow
    next_model_startrow = 0
    for index in range(nchunks):
        if index == nchunks - 1:
            chunk_nrows = nrows - (nchunks - 1) * nrows_per_chunk
        else:
            chunk_nrows = nrows_per_chunk
        chunks.append(
            RowChunk(
                input_startrow=next_input_startrow,
                model_startrow=next_model_startrow,
                nrows=chunk_nrows,
            )
        )
        next_input_startrow += chunk_nrows
        next_model_startrow += chunk_nrows
    return chunks


def copy_measurement_set(source: str, destination: str) -> None:
    """Copy a Measurement Set directory with writable default permissions."""
    if os.path.exists(destination):
        shutil.rmtree(destination, ignore_errors=True)
    subprocess.check_call(["cp", "-r", "-L", "--no-preserve=mode", source, destination])


def modeldata_output_stem(model_path: str) -> str:
    """Return the model output name without the ``_modeldata`` suffix."""
    return os.path.basename(model_path).removesuffix("_modeldata")


def read_model_data(
    model_paths: list[str],
    column: str,
    *,
    startrow: int,
    nrows: int,
) -> list[np.ndarray]:
    """Read one data chunk from each model Measurement Set."""
    data = []
    for model_path in model_paths:
        table = pt.table(model_path, readonly=True, ack=False)
        try:
            data.append(table.getcol(column, startrow=startrow, nrow=nrows))
        finally:
            table.close()
    return data


def sum_model_data(model_data: list[np.ndarray]) -> Optional[np.ndarray]:
    """Sum model-data arrays while preserving the first array's dtype."""
    combined = None
    for data in model_data:
        if combined is None:
            combined = data.copy()
        else:
            combined += data
    return combined


def _read_channel_frequencies(ms_path: str, separator: str) -> np.ndarray:
    spectral_window = pt.table(f"{ms_path}{separator}SPECTRAL_WINDOW", ack=False)
    try:
        return spectral_window.getcol("CHAN_FREQ")
    finally:
        spectral_window.close()
