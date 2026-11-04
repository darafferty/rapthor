"""Integration tests for runs on observations from several epochs."""

import shutil
import subprocess
from pathlib import Path

import casacore.tables as pt
import numpy as np
import pytest
from astropy.io import fits

from .utils import find_command_records, get_working_dir_from_parset, update_parset_path

# Later epochs start well after the test MS has ended
EPOCH_SHIFT_SECONDS = 86400.0


def _shift_ms_in_time(ms_path):
    with pt.table(str(ms_path), readonly=False, ack=False) as table:
        for column in ("TIME", "TIME_CENTROID"):
            table.putcol(column, table.getcol(column) + EPOCH_SHIFT_SECONDS)
    with pt.table(f"{ms_path}::OBSERVATION", readonly=False, ack=False) as table:
        table.putcol("TIME_RANGE", table.getcol("TIME_RANGE") + EPOCH_SHIFT_SECONDS)


def _copy_ms_as_later_epoch(source_ms, target_ms):
    shutil.copytree(source_ms, target_ms)
    _shift_ms_in_time(target_ms)
    return target_ms


def _copy_ms_as_later_epoch_without_station(source_ms, target_ms):
    """Copy an MS to a later epoch, without its last remote station."""
    with pt.table(f"{source_ms}::ANTENNA", ack=False) as table:
        names = table.getcol("NAME")
    station = next(name for name in reversed(names) if name.startswith("RS"))
    subprocess.run(
        [
            "DP3",
            f"msin={source_ms}",
            "steps=[filter]",
            f"filter.baseline=!{station}",
            "filter.remove=True",
            f"msout={target_ms}",
        ],
        check=True,
    )
    _shift_ms_in_time(target_ms)
    return target_ms


def _run_two_epochs(generated_parset_path, strategy_path, later_epoch_ms):
    """Run a reduction on an input MS and a later epoch, and return its working directory."""
    parset_path = update_parset_path(
        generated_parset_path,
        {
            "allow_internet_access": "False",
            "strategy": str(strategy_path),
            "input_ms": later_epoch_ms,
        },
    )
    result = subprocess.run(["rapthor", str(parset_path)], capture_output=True, text=True)
    output = f"{result.stdout}\n{result.stderr}"
    assert result.returncode == 0, f"Rapthor failed with output:\n{output}"
    assert "Operation calibrate_1 completed" in output
    assert "Operation image_1 completed" in output
    assert "Rapthor has finished :)" in output
    return Path(get_working_dir_from_parset(parset_path))


def _assert_image_has_data(working_dir):
    images = sorted((working_dir / "images" / "image_1").glob("*-MFS*-image-pb.fits*"))
    assert images
    with fits.open(images[0], memmap=False) as hdul:
        image = next(hdu.data for hdu in hdul if hdu.data is not None)
    assert np.isfinite(image).any()


def _wsclean_measurement_sets(working_dir):
    records = find_command_records(working_dir, operation="image_1", executable="wsclean")
    assert records
    return [
        argument
        for argument in records[-1].command
        if argument.endswith(".ms") and not argument.startswith("-")
    ]


SKY_MODELS = (
    "tests/resources/integration_template.parset",
    "tests/resources/integration_true_sky.txt",
    "tests/resources/integration_apparent_sky.txt",
)


@pytest.mark.integration
@pytest.mark.parametrize("generated_parset_path", [SKY_MODELS], indirect=True)
def test_rapthor_run_concatenates_epochs_with_the_same_stations(
    generated_parset_path, single_loop_strategy_path, test_ms, tmp_path
):
    later_epoch = _copy_ms_as_later_epoch(test_ms, tmp_path / "later_epoch.ms")

    working_dir = _run_two_epochs(
        generated_parset_path, single_loop_strategy_path, f"[{test_ms}, {later_epoch}]"
    )

    measurement_sets = _wsclean_measurement_sets(working_dir)
    assert len(measurement_sets) == 1
    assert measurement_sets[0].endswith("_concat.ms")
    _assert_image_has_data(working_dir)


@pytest.mark.integration
@pytest.mark.parametrize("generated_parset_path", [SKY_MODELS], indirect=True)
def test_rapthor_run_images_epochs_with_different_stations(
    generated_parset_path, single_loop_strategy_path, test_ms, tmp_path
):
    later_epoch = _copy_ms_as_later_epoch_without_station(test_ms, tmp_path / "later_epoch.ms")

    working_dir = _run_two_epochs(
        generated_parset_path, single_loop_strategy_path, f"[{test_ms}, {later_epoch}]"
    )

    measurement_sets = _wsclean_measurement_sets(working_dir)
    assert len(measurement_sets) == 2
    assert not any(path.endswith("_concat.ms") for path in measurement_sets)
    assert not find_command_records(working_dir, operation="image_1", executable="taql")
    _assert_image_has_data(working_dir)


@pytest.mark.integration
@pytest.mark.parametrize(
    "generated_parset_path_normalisation",
    [
        (
            "tests/resources/integration_template.parset",
            "tests/resources/normalization_true_sky.txt",
            "tests/resources/normalization_apparent_sky.txt",
        )
    ],
    indirect=True,
)
def test_rapthor_run_normalizes_epochs_with_different_stations(
    generated_parset_path_normalisation,
    single_loop_do_normalize_strategy_path,
    ms_for_normalisation,
    normalization_reference_inputs,
    tmp_path,
):
    later_epoch = _copy_ms_as_later_epoch_without_station(
        ms_for_normalisation, tmp_path / "later_epoch.ms"
    )
    parset_path = update_parset_path(
        generated_parset_path_normalisation,
        {
            **normalization_reference_inputs,
            "allow_internet_access": "False",
            "strategy": str(single_loop_do_normalize_strategy_path),
            "input_ms": f"[{ms_for_normalisation}, {later_epoch}]",
        },
    )

    result = subprocess.run(["rapthor", str(parset_path)], capture_output=True, text=True)

    output = f"{result.stdout}\n{result.stderr}"
    assert result.returncode == 0, f"Rapthor failed with output:\n{output}"
    assert "Operation normalize_1 completed" in output
    assert "Operation image_1 completed" in output
    working_dir = Path(get_working_dir_from_parset(parset_path))
    records = find_command_records(working_dir, operation="normalize_1", executable="wsclean")
    assert sum(argument.endswith(".ms") for argument in records[-1].command) == 2
    log_text = output + (working_dir / "logs/rapthor.log").read_text()
    assert "Using reference sky models provided as input for normalization" in log_text
