"""Integration tests for runs on observations from several epochs."""

import subprocess
from pathlib import Path

import casacore.tables as pt
import numpy as np
import pytest
from astropy.io import fits

from tests.conftest import copy_ms_to_later_epoch, copy_ms_to_next_band

from .utils import (
    find_command_records,
    get_working_dir_from_parset,
    run_rapthor,
    update_parset_path,
)

# Each run starts its own Prefect server and Dask cluster in a subprocess
pytestmark = [pytest.mark.integration, pytest.mark.prefect, pytest.mark.serial]

# Well below the peak of about 3 Jy/beam in images of the test MS
MIN_IMAGE_PEAK_JY = 0.1

SKY_MODELS = (
    "tests/resources/integration_template.parset",
    "tests/resources/integration_true_sky.txt",
    "tests/resources/integration_apparent_sky.txt",
)


def _copy_ms_without_station(source_ms, target_ms):
    """Copy an MS without its last remote station."""
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
    return target_ms


def _parset_for(parset_path, strategy_path, input_ms, **options):
    return update_parset_path(
        parset_path,
        {
            "allow_internet_access": "False",
            "strategy": str(strategy_path),
            "input_ms": f"[{', '.join(str(path) for path in input_ms)}]",
            **options,
        },
    )


def _assert_image_has_data(working_dir):
    images = sorted((working_dir / "images" / "image_1").glob("*-MFS*-image-pb.fits*"))
    assert images
    with fits.open(images[0], memmap=False) as hdul:
        image = next(hdu.data for hdu in hdul if hdu.data is not None)
    assert np.nanmax(np.abs(image)) > MIN_IMAGE_PEAK_JY


def _wsclean_measurement_sets(working_dir, operation="image_1"):
    records = find_command_records(working_dir, operation=operation, executable="wsclean")
    assert records
    return [argument for argument in records[-1].command if argument.endswith(".ms")]


@pytest.mark.parametrize("generated_parset_path", [SKY_MODELS], indirect=True)
def test_rapthor_run_concatenates_epochs_with_the_same_stations(
    generated_parset_path, single_loop_strategy_path, test_ms, tmp_path
):
    later_epoch = copy_ms_to_later_epoch(test_ms, tmp_path / "later_epoch.ms")
    parset_path = _parset_for(
        generated_parset_path, single_loop_strategy_path, [test_ms, later_epoch]
    )

    run_rapthor(parset_path)

    working_dir = Path(get_working_dir_from_parset(parset_path))
    measurement_sets = _wsclean_measurement_sets(working_dir)
    assert len(measurement_sets) == 1
    assert measurement_sets[0].endswith("_concat.ms")
    _assert_image_has_data(working_dir)


@pytest.mark.parametrize("generated_parset_path", [SKY_MODELS], indirect=True)
def test_rapthor_run_images_epochs_with_different_stations(
    generated_parset_path, single_loop_strategy_path, test_ms, tmp_path
):
    later_epoch = copy_ms_to_later_epoch(
        _copy_ms_without_station(test_ms, tmp_path / "without_station.ms"),
        tmp_path / "later_epoch.ms",
    )
    parset_path = _parset_for(
        generated_parset_path, single_loop_strategy_path, [test_ms, later_epoch]
    )

    run_rapthor(parset_path)

    working_dir = Path(get_working_dir_from_parset(parset_path))
    measurement_sets = _wsclean_measurement_sets(working_dir)
    assert len(measurement_sets) == 2
    assert not any(path.endswith("_concat.ms") for path in measurement_sets)
    assert not find_command_records(working_dir, operation="image_1", executable="taql")
    _assert_image_has_data(working_dir)


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
    later_epoch = copy_ms_to_later_epoch(
        _copy_ms_without_station(ms_for_normalisation, tmp_path / "without_station.ms"),
        tmp_path / "later_epoch.ms",
    )
    parset_path = _parset_for(
        generated_parset_path_normalisation,
        single_loop_do_normalize_strategy_path,
        [ms_for_normalisation, later_epoch],
        **normalization_reference_inputs,
    )

    output = run_rapthor(parset_path)

    assert "Operation normalize_1 completed" in output
    working_dir = Path(get_working_dir_from_parset(parset_path))
    assert len(_wsclean_measurement_sets(working_dir, operation="normalize_1")) == 2
    log_text = output + (working_dir / "logs/rapthor.log").read_text()
    assert "Using reference sky models provided as input for normalization" in log_text


@pytest.mark.parametrize("generated_parset_path", [SKY_MODELS], indirect=True)
def test_rapthor_refuses_bands_of_an_epoch_with_different_stations(
    generated_parset_path, single_loop_strategy_path, test_ms, tmp_path
):
    other_band = copy_ms_to_next_band(
        _copy_ms_without_station(test_ms, tmp_path / "without_station.ms"),
        tmp_path / "other_band.ms",
    )
    parset_path = _parset_for(
        generated_parset_path, single_loop_strategy_path, [test_ms, other_band]
    )

    result = subprocess.run(
        ["rapthor", str(parset_path)], capture_output=True, text=True, check=False
    )

    output = f"{result.stdout}\n{result.stderr}"
    assert result.returncode != 0
    assert "of the same epoch" in output
    assert not find_command_records(
        get_working_dir_from_parset(parset_path), operation="concatenate_1", executable="DP3"
    )
