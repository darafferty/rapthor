"""Integration tests for the Rapthor pipeline when using the do_normalize step."""

import subprocess
from pathlib import Path

import lsmtool
import numpy as np
import pytest
from losoto.h5parm import h5parm

from rapthor.execution.image.catalog_cli import main as fetch_skymodels

from .utils import get_working_dir_from_parset, update_parset_path


class SkyModelServerUnavailable(Exception):
    """A sky model could not be downloaded because the server is down."""


@pytest.mark.xfail(
    strict=True,
    raises=SkyModelServerUnavailable,
    reason="vizier.u-strasbg.fr is not available to download sky models",
)
@pytest.mark.internet
@pytest.mark.integration
@pytest.mark.parametrize("normalization_skymodel_paths", [None], indirect=True)
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
def test_rapthor_run_single_loop_with_do_normalize(
    generated_parset_path_normalisation, single_loop_do_normalize_strategy_path
):
    """Test a single self-calibration loop end to end."""

    updated_parset_path = update_parset_path(
        generated_parset_path_normalisation,
        {
            "allow_internet_access": "True",
            "strategy": str(single_loop_do_normalize_strategy_path),
        },
    )
    command = ["rapthor", str(updated_parset_path)]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    output = f"{result.stdout}\n{result.stderr}"
    # Rapthor runs in another process, so its exception does not reach this test
    if result.returncode != 0 and "vizier.u-strasbg.fr" in output:
        raise SkyModelServerUnavailable(output)
    assert result.returncode == 0, f"Rapthor failed with output:\n{output}"
    assert "Operation calibrate_1 completed" in output
    assert "Operation predict_1 completed" in output
    assert "Operation normalize_1 completed" in output
    assert "Operation image_1 completed" in output
    assert "Operation mosaic_1 completed" in output
    assert "Rapthor has finished :)" in output


@pytest.mark.integration
@pytest.mark.parametrize(
    "generated_parset_path",
    [
        (
            "tests/resources/integration_template.parset",
            "tests/resources/normalization_true_sky.txt",
            "tests/resources/normalization_apparent_sky.txt",
        )
    ],
    indirect=True,
)
def test_rapthor_run_single_loop_with_do_normalize_no_internet_raises_error(
    generated_parset_path, single_loop_do_normalize_strategy_path
):
    """Test that rapthor raises an error when do_normalize is used without internet access."""

    updated_parset_path = update_parset_path(
        generated_parset_path,
        {
            "allow_internet_access": "False",
            "strategy": str(single_loop_do_normalize_strategy_path),
        },
    )

    command = ["rapthor", str(updated_parset_path)]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    output = f"{result.stdout}\n{result.stderr}"
    assert result.returncode != 0, (
        f"Rapthor should have failed but succeeded with output:\n{output}"
    )
    assert (
        "The strategy includes do_normalize in the first cycle, which requires internet access "
        in output
    )


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
def test_rapthor_run_single_loop_with_do_normalize_no_internet_provided_sky_models_ok(
    generated_parset_path_normalisation,
    single_loop_do_normalize_strategy_path,
    normalization_reference_inputs,
):
    """Test that rapthor runs successfully when do_normalize is used without internet access but sky models are provided."""

    updated_parset_path = update_parset_path(
        generated_parset_path_normalisation,
        {
            "allow_internet_access": "False",
            "strategy": str(single_loop_do_normalize_strategy_path),
            **normalization_reference_inputs,
        },
    )

    command = ["rapthor", str(updated_parset_path)]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    output = f"{result.stdout}\n{result.stderr}"
    assert result.returncode == 0, f"Rapthor failed with output:\n{output}"
    assert "Operation calibrate_1 completed" in output
    assert "Operation predict_1 completed" in output
    assert "Operation normalize_1 completed" in output
    assert "Operation image_1 completed" in output
    assert "Operation mosaic_1 completed" in output
    assert "Rapthor has finished :)" in output

    working_dir = Path(get_working_dir_from_parset(updated_parset_path))
    # Normalization is a Python task; its messages are in the main/worker logs,
    # rather than a normalize_flux_scale command-line log.
    log_text = output + (working_dir / "logs/rapthor.log").read_text()
    assert "Using reference sky models provided as input for normalization" in log_text
    assert "normalization_reference_120mhz.txt" in log_text
    assert "normalization_reference_160mhz.txt" in log_text
    assert "Downloading vlssr catalog for this field" not in log_text
    assert "Downloading wenss catalog for this field" not in log_text
    assert "Flux density scale normalization will be skipped" not in log_text

    normalization_file = working_dir / "solutions/normalize_1/sector_1_normalize.h5parm"
    with h5parm(str(normalization_file), readonly=True) as h5:
        amplitudes = h5.getSolset("sol000").getSoltab("amplitude000").getValues(retAxesVals=False)
    assert np.all(np.isfinite(amplitudes))
    assert not np.allclose(amplitudes, 1.0)


@pytest.mark.integration
@pytest.mark.parametrize("normalization_skymodel_paths", [None], indirect=True)
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
def test_rapthor_run_single_loop_with_do_normalize_no_matching_sources_skips_normalization(
    generated_parset_path_normalisation,
    single_loop_do_normalize_strategy_path,
):
    """Non-matching reference models must produce unity corrections and finish."""
    resource_dir = Path(__file__).parents[1] / "resources"
    updated_parset_path = update_parset_path(
        generated_parset_path_normalisation,
        {
            "allow_internet_access": "False",
            "strategy": str(single_loop_do_normalize_strategy_path),
            "photometry_skymodel": "",
            "astrometry_skymodel": "",
            "normalization_skymodels": (
                f"[{resource_dir / 'test_apparent_sky.txt'}, {resource_dir / 'test_true_sky.txt'}]"
            ),
            "normalization_reference_frequencies": "[120000000.0, 160000000.0]",
        },
    )
    result = subprocess.run(
        ["rapthor", str(updated_parset_path)], capture_output=True, text=True, check=False
    )
    output = f"{result.stdout}\n{result.stderr}"
    assert result.returncode == 0, f"Rapthor failed with output:\n{output}"
    assert "Operation normalize_1 completed" in output
    assert "Rapthor has finished :)" in output

    working_dir = Path(get_working_dir_from_parset(updated_parset_path))
    log_text = output + (working_dir / "logs/rapthor.log").read_text()
    assert "Using reference sky models provided as input for normalization" in log_text
    assert "Too few sources with successful SED fits" in log_text
    assert "Flux density scale normalization will be skipped" in log_text
    assert "Downloading vlssr catalog for this field" not in log_text
    assert "Downloading wenss catalog for this field" not in log_text
    normalization_file = working_dir / "solutions/normalize_1/sector_1_normalize.h5parm"
    with h5parm(str(normalization_file), readonly=True) as h5:
        amplitudes = h5.getSolset("sol000").getSoltab("amplitude000").getValues(retAxesVals=False)
    assert amplitudes == pytest.approx(1.0)


def _write_survey_model(_cone_params, survey, skymodel_path):
    """Stand in for a survey server, writing the test sky model at the frequency of the survey."""
    frequency = {"VLSSR": 74e6, "WENSS": 327e6}.get(survey, 150e6)
    model = lsmtool.load(str(Path(__file__).parents[1] / "resources/normalization_true_sky.txt"))
    spectral_indices = np.asarray(model.getColValues("SpectralIndex"))[:, 0]
    model.setColValues(
        "I",
        model.getColValues("I")
        * (frequency / model.getColValues("ReferenceFrequency")) ** spectral_indices,
    )
    model.setColValues("ReferenceFrequency", np.full(len(model), frequency))
    model.write(str(skymodel_path), clobber=True)


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
def test_rapthor_run_without_internet_on_sky_models_from_fetch_skymodels(
    generated_parset_path_normalisation,
    single_loop_do_normalize_strategy_path,
    ms_for_normalisation,
    tmp_path,
    monkeypatch,
    capsys,
):
    """The options printed by fetch_skymodels let normalization and diagnostics run offline."""
    monkeypatch.setattr(
        "lsmtool.download_skymodel.download_skymodel_from_survey", _write_survey_model
    )
    assert fetch_skymodels([str(ms_for_normalisation), str(tmp_path / "catalogs")]) == 0
    options = dict(
        line.split(" = ", 1) for line in capsys.readouterr().out.splitlines() if " = " in line
    )

    updated_parset_path = update_parset_path(
        generated_parset_path_normalisation,
        {**options, "strategy": str(single_loop_do_normalize_strategy_path)},
    )
    result = subprocess.run(
        ["rapthor", str(updated_parset_path)],
        capture_output=True,
        text=True,
        check=False,
    )

    output = f"{result.stdout}\n{result.stderr}"
    assert result.returncode == 0, f"Rapthor failed with output:\n{output}"
    assert "Operation normalize_1 completed" in output
    assert "Operation image_1 completed" in output
    log_text = (
        output
        + (Path(get_working_dir_from_parset(updated_parset_path)) / "logs/rapthor.log").read_text()
    )
    assert "Using reference sky models provided as input for normalization" in log_text
    assert "check will be skipped" not in log_text
    assert "Downloading vlssr catalog for this field" not in log_text
