"""Module for pytest fixtures."""

import shlex
import shutil
import subprocess
from pathlib import Path

import casacore.tables as pt
import numpy as np
import pytest

from ..conftest import REPO_ROOT_DIR, RESOURCE_DIR, generate_parset_path

TEST_INTEGRATION_TRUE_SKYMODEL = (RESOURCE_DIR / "integration_true_sky.txt").as_posix()
TEST_INTEGRATION_APPARENT_SKYMODEL = (RESOURCE_DIR / "integration_apparent_sky.txt").as_posix()


COMMON_STRATEGY_SETTINGS = {
    "channel_width_hz": 195312.5,
    # Set slow-gain and fulljones solves to False except when required
    "do_slowgain_solve": False,
    "do_fulljones_solve": False,
    # Don't remove bright outliers or in-field sources -- image full field
    "peel_outliers": False,
    "peel_bright_sources": False,
    # Fast phase (ionosphere) and slow gain (beam) time intervals (s)
    "fast_timestep_sec": 32.0,
    "medium_timestep_sec": 120.0,
    "slow_timestep_sec": 600.0,
    # Turn off flux-scale bootstrapping
    "do_normalize": False,
    # PyBDSF settings
    "auto_mask": 5.0,
    "auto_mask_nmiter": 2,
    "threshisl": 3.0,
    "threshpix": 5.0,
    # Constrain max nr of imaging major cycles
    "max_nmiter": 12,
    # Disable regrouping of sky model
    "regroup_model": True,
    # Max distance allowed between selected DDE calibrators
    "max_distance": None,  # no distance constraint
    # Don't check for self-cal convergence
    "do_check": False,
    "target_flux": 0.3,
    "max_directions": 4,
}


def make_strategy_step(**overrides):
    """Helper to create a strategy step with settings and overrides."""
    return {**COMMON_STRATEGY_SETTINGS, **overrides}


def _write_normalization_skymodel(output_path):
    """Write the apparent sky model used for normalization tests."""
    source_model_path = Path("tests/resources/integration_apparent_sky.txt")
    output_path.write_text(source_model_path.read_text(encoding="utf-8"), encoding="utf-8")


def _set_synthetic_uvw_geometry(ms_path):
    """Replace UVW with a denser antenna-consistent synthetic geometry."""
    ref_wavelength_m = 299792458.0 / 134373474.12109375
    with pt.table(str(ms_path), readonly=False, ack=False) as table:
        uvw = table.getcol("UVW")
        ant1 = table.getcol("ANTENNA1")
        ant2 = table.getcol("ANTENNA2")
        times = table.getcol("TIME")

        unique_times = np.unique(times)
        antennas = np.unique(np.concatenate([ant1, ant2]))
        base_radius_lambda = np.linspace(0.0, 2500.0, len(antennas))
        antenna_index = {antenna: index for index, antenna in enumerate(antennas)}
        time_index = {time_value: index for index, time_value in enumerate(unique_times)}

        positions = {}
        for time_value in unique_times:
            t_index = time_index[time_value]
            for antenna in antennas:
                a_index = antenna_index[antenna]
                theta = (2.0 * np.pi * a_index / len(antennas)) + 0.35 * t_index
                radius_lambda = base_radius_lambda[a_index]
                positions[(time_value, antenna)] = np.array(
                    [
                        np.cos(theta) * radius_lambda * ref_wavelength_m,
                        np.sin(theta) * radius_lambda * ref_wavelength_m,
                        0.0,
                    ]
                )

        for row_index in range(len(uvw)):
            first_position = positions[(times[row_index], ant1[row_index])]
            second_position = positions[(times[row_index], ant2[row_index])]
            uvw[row_index] = second_position - first_position

        table.putcol("UVW", uvw)


@pytest.fixture
def single_loop_strategy_path(tmp_path):
    """Fixture to generate a strategy file for a single self-calibration loop."""
    strategy_steps = [make_strategy_step(do_calibrate=True, do_image=True)]
    strategy_content = f"strategy_steps = {strategy_steps}"
    strategy_path = tmp_path / "single_loop_strategy.py"
    strategy_path.write_text(strategy_content)
    return strategy_path


@pytest.fixture(
    params=[True, False],
    ids=["peel_bright_sources_enabled", "peel_bright_sources_disabled"],
)
def single_loop_strategy_path_peel_bright_sources(request, tmp_path):
    """
    Fixture to generate a strategy file for a single self-calibration loop
    with bright sources peeling enabled or disabled.

    Returns a tuple of (strategy_path, peel_bright_sources) so the test can
    branch its assertions accordingly.
    """
    peel = request.param
    strategy_steps = [
        make_strategy_step(do_calibrate=True, do_image=True, peel_bright_sources=peel)
    ]
    strategy_content = f"strategy_steps = {strategy_steps}"
    strategy_path = tmp_path / "single_loop_strategy.py"
    strategy_path.write_text(strategy_content)
    return strategy_path, peel


@pytest.fixture
def single_loop_strategy_path_calibrate_di(tmp_path):
    """Fixture to generate a strategy file for a single self-calibration loop with DI calibration."""
    strategy_steps = [make_strategy_step(do_calibrate=True, do_image=True, do_fulljones_solve=True)]
    strategy_content = f"strategy_steps = {strategy_steps}"
    strategy_path = tmp_path / "single_loop_strategy_calibrate_di.py"
    strategy_path.write_text(strategy_content)
    return strategy_path


@pytest.fixture
def single_loop_do_normalize_strategy_path(tmp_path):
    """Strategy file for a single self-calibration loop with do_normalize."""
    strategy_steps = [make_strategy_step(do_calibrate=True, do_image=True, do_normalize=True)]
    strategy_content = f"strategy_steps = {strategy_steps}"
    strategy_path = tmp_path / "single_loop_do_normalize_strategy.py"
    strategy_path.write_text(strategy_content)
    return strategy_path


@pytest.fixture
def no_matching_normalization_inputs(single_loop_do_normalize_strategy_path):
    """Return parset updates that make do_normalize use non-matching reference models."""
    resource_dir = Path(__file__).parents[1] / "resources"
    apparent_skymodel = resource_dir / "test_apparent_sky.txt"
    true_skymodel = resource_dir / "test_true_sky.txt"
    return {
        "allow_internet_access": "False",
        "strategy": str(single_loop_do_normalize_strategy_path),
        "photometry_skymodel": "",
        "astrometry_skymodel": "",
        "normalization_skymodels": f"[{apparent_skymodel}, {true_skymodel}]",
        "normalization_reference_frequencies": "[150000000.0, 150000000.0]",
    }


@pytest.fixture
def single_loop_strategy_path_fast_medium_slow(tmp_path):
    """Fixture to generate a strategy file for a single self-calibration loop with fast, medium, and slow gains."""
    strategy_steps = [make_strategy_step(do_calibrate=True, do_image=True, do_slowgain_solve=True)]
    strategy_content = f"strategy_steps = {strategy_steps}"
    strategy_path = tmp_path / "single_loop_strategy_fast_medium_slow.py"
    strategy_path.write_text(strategy_content)
    return strategy_path


@pytest.fixture
def ms_for_normalisation(tmp_path, test_ms):
    """Provide a synthetic MS with denser UV coverage for normalization tests."""
    ms_path = tmp_path / "test_ms_for_normalization.ms"
    shutil.copytree(test_ms, ms_path)
    _set_synthetic_uvw_geometry(ms_path)
    with pt.table(str(ms_path), readonly=False, ack=False) as table:
        data = table.getcol("DATA")
        data[...] = 0.0j
        table.putcol("DATA", data)

    skymodel_path = tmp_path / "integration_apparent_sky_normalization.txt"
    _write_normalization_skymodel(skymodel_path)

    predicted_ms = tmp_path / "test_ms_for_normalization_predicted.ms"

    dp3_command = (
        f"DP3 msin={ms_path} steps=[predict] "
        f"predict.usebeammodel=True "
        f"predict.beam_interval=120 "
        f"predict.beammode=array_factor "
        f"predict.sourcedb={skymodel_path} "
        f"msout={predicted_ms}"
    )

    subprocess.run(shlex.split(dp3_command), check=True)
    rng = np.random.default_rng(0)
    with pt.table(str(predicted_ms), readonly=False, ack=False) as table:
        data = table.getcol("DATA")
        noise = (
            rng.normal(scale=0.05, size=data.shape) + 1j * rng.normal(scale=0.05, size=data.shape)
        ).astype(data.dtype)
        table.putcol("DATA", data + noise)
    shutil.rmtree(ms_path)
    predicted_ms.rename(ms_path)
    return ms_path


@pytest.fixture
def generated_parset_path(request, tmp_path, test_ms):
    """
    Fixture to generate a complete parset from a template and return the path.

    This fixture is used to read in and update a template parset file. It is
    parametrised using the pytest request fixture and expects a tuple
    containing three paths to the following files:

    1. Template parset (e.g. in tests/resources/parsets/)
    2. True sky model (e.g. in tests/resources/)
    3. Apparent sky model (e.g. in tests/resources/)

    This fixture can be used to test rapthor runs end to end on a small input
    measurement set with different strategies and sky models.
    For further details see `generate_parset` function.
    """
    parset_path, input_skymodel_path, apparent_skymodel_path = request.param
    parset_path = REPO_ROOT_DIR / parset_path
    output_parset_path = tmp_path / "generated.parset"

    generate_parset_path(
        parset_path,
        output_parset_path,
        test_ms,
        input_skymodel_path,
        apparent_skymodel_path,
        normalization_skymodel_paths=None,
    )

    return output_parset_path


@pytest.fixture(
    params=[
        None,
        [
            TEST_INTEGRATION_APPARENT_SKYMODEL,
            TEST_INTEGRATION_TRUE_SKYMODEL,
        ],
    ],
    ids=["downloaded_surveys", "reference_skymodels"],
)
def normalization_skymodel_paths(request):
    """Return optional normalization sky model paths for integration tests."""
    return request.param


@pytest.fixture
def generated_parset_path_normalisation(
    request, tmp_path, ms_for_normalisation, normalization_skymodel_paths
):
    """
    Fixture to generate a complete parset from a template and return the path.

    This fixture is used to read in and update a template parset file. It is
    parametrised using the pytest request fixture and expects a tuple
    containing three paths to the following files:

    1. Template parset (e.g. in tests/resources/parsets/)
    2. True sky model (e.g. in tests/resources/)
    3. Apparent sky model (e.g. in tests/resources/)

    This fixture can be used to test rapthor runs end to end on a small input
    measurement set with different strategies and sky models.
    For further details see `generate_parset` function.
    """
    parset_path, input_skymodel_path, apparent_skymodel_path = request.param
    parset_path = REPO_ROOT_DIR / parset_path
    output_parset_path = tmp_path / "generated.parset"

    generate_parset_path(
        parset_path,
        output_parset_path,
        ms_for_normalisation,
        input_skymodel_path,
        apparent_skymodel_path,
        normalization_skymodel_paths=normalization_skymodel_paths,
    )
    return output_parset_path
