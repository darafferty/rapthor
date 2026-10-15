"""Pytest setup for execution-layer tests."""

import os
import shutil
from contextlib import nullcontext
from unittest.mock import patch

import casacore.tables as pt
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from prefect.testing.utilities import prefect_test_harness

from rapthor.execution.config import ExecutionConfig
from rapthor.execution.image.flux_normalization import create_normalization_h5parm
from rapthor.execution.predict.flow import run_predict_model_data
from rapthor.lib import miscellaneous as misc
from tests.execution.payload_factories import representative_predict_payload

# Shared inputs for tests that run real DP3 on the small test Measurement Set.
requires_dp3 = pytest.mark.skipif(shutil.which("DP3") is None, reason="DP3 is not available")
TEST_PATCH_NAME = "Patch_0"
TEST_DIRECTION_NAME = f"[{TEST_PATCH_NAME}]"
# The normalization h5parm stores sqrt(normalization) as an amplitude, so a
# model corrupted with it scales by the normalization itself.
TEST_NORMALIZATION = 4.0


def pytest_sessionstart(session):  # pragma: no cover
    _ = session
    os.environ.setdefault("PREFECT_API_URL", "")
    os.environ.setdefault("PREFECT_LOGGING_RICH_CONSOLE", "false")
    os.environ.setdefault("PREFECT_LOGGING_LEVEL", "INFO")
    os.environ.setdefault("PREFECT_SERVER_EPHEMERAL_STARTUP_TIMEOUT_SECONDS", "180")


def run_flow_for_test(
    flow_fn,
    payload,
    execution_config=None,
    shell_operation_cls=None,
):
    """Run a production Prefect flow with optional fake shell loading for tests."""
    shell_loader = (
        patch("rapthor.execution.shell._load_shell_operation_cls", return_value=shell_operation_cls)
        if shell_operation_cls is not None
        else nullcontext()
    )
    config = execution_config or ExecutionConfig(task_runner="sync")
    with shell_loader, prefect_test_harness(server_startup_timeout=None):
        return flow_fn(payload, execution_config=config)


@pytest.fixture(scope="session")
def test_ms_info(test_ms):
    """Return the test MS quantities needed to build sky models and h5parms."""
    with pt.table(test_ms, ack=False) as table:
        times = np.unique(table.getcol("TIME"))
        interval = table.getcol("INTERVAL")[0]
        antenna1 = table.getcol("ANTENNA1")
        antenna2 = table.getcol("ANTENNA2")
    with pt.table(f"{test_ms}/SPECTRAL_WINDOW", ack=False) as spw_table:
        frequencies = spw_table.getcol("CHAN_FREQ")[0]
    with pt.table(f"{test_ms}/FIELD", ack=False) as field_table:
        ra, dec = np.squeeze(field_table.getcol("PHASE_DIR"))
    with pt.table(f"{test_ms}/ANTENNA", ack=False) as antenna_table:
        antennas = list(antenna_table.getcol("NAME"))
    return {
        # Start just before the first sample, as Observation does.
        "starttime": misc.convert_mjd2mvt(times[0] - interval / 100),
        "times": times,
        "frequencies": frequencies,
        "ra": ra,
        "dec": dec,
        "antennas": antennas,
        "antenna1": antenna1,
        "antenna2": antenna2,
    }


@pytest.fixture(scope="session")
def point_source_skymodel(test_ms_info, tmp_path_factory):
    """Write a one-patch sky model with a single 1 Jy point source at the phase centre."""
    coord = SkyCoord(test_ms_info["ra"], test_ms_info["dec"], unit="rad")
    ra = coord.ra.to_string(unit="hourangle", sep=":", precision=4)
    dec = coord.dec.to_string(unit="deg", sep=".", precision=4, alwayssign=True)
    path = tmp_path_factory.mktemp("skymodel") / "point_source.skymodel"
    path.write_text(
        "FORMAT = Name, Type, Patch, Ra, Dec, I, SpectralIndex='[]', LogarithmicSI, "
        "ReferenceFrequency='134000000.0', MajorAxis, MinorAxis, Orientation\n"
        f" , , {TEST_PATCH_NAME}, {ra}, {dec}\n"
        f"source_0, POINT, {TEST_PATCH_NAME}, {ra}, {dec}, 1.0, [], false, 134000000.0, , , \n"
    )
    return path.as_posix()


@pytest.fixture(scope="session")
def normalization_h5parm(test_ms, test_ms_info, tmp_path_factory):
    """Write a normalization h5parm of TEST_NORMALIZATION in the production layout."""
    path = tmp_path_factory.mktemp("normalize") / "normalize.h5"
    create_normalization_h5parm(
        antenna_file=f"{test_ms}/ANTENNA",
        field_file=f"{test_ms}/FIELD",
        h5parm_file=path.as_posix(),
        frequencies=test_ms_info["frequencies"],
        normalizations=np.full(len(test_ms_info["frequencies"]), TEST_NORMALIZATION),
    )
    return path.as_posix()


@pytest.fixture(scope="session")
def predicted_model_ms(test_ms, test_ms_info, point_source_skymodel, tmp_path_factory):
    """Return an MS whose DATA column holds the predicted point-source model.

    The prediction runs through the production predict task, so the model
    carries the same beam treatment as a pipeline prediction.
    """
    workdir = tmp_path_factory.mktemp("model_ms")
    task = representative_predict_payload(workdir)["predict_tasks"][0]
    task.update(
        {
            "msin": test_ms,
            "msout": "model.ms",
            "msout_path": (workdir / "model.ms").as_posix(),
            "starttime": test_ms_info["starttime"],
            "ntimes": 0,
            "sourcedb": point_source_skymodel,
            "directions": [TEST_DIRECTION_NAME],
            "numthreads": 2,
        }
    )
    return run_predict_model_data(task, workdir.as_posix())["path"]
