"""Real DP3 checks for the calibration solve command.

These tests run the production calibration chunk task (``run_calibrate_chunk``)
with DP3 on the small test Measurement Set, with the data set equal to the
predicted sky model. They verify where the flux-scale normalization reaches the
solve model, not the scientific quality of the solutions.
"""

import numpy as np
import pytest
from losoto.h5parm import h5parm

from rapthor.execution.calibrate.solves import run_calibrate_chunk
from rapthor.execution.predict.flow import run_predict_model_data
from tests.execution.conftest import TEST_DIRECTION_NAME, TEST_NORMALIZATION, requires_dp3
from tests.execution.payload_factories import representative_predict_payload

pytestmark = [pytest.mark.integration, requires_dp3]

# Diagonal gains solved against unscaled data come out at 1 / sqrt(normalization)
# when the model has been scaled by the normalization.
EXPECTED_NORMALIZED_AMPLITUDE = 1.0 / np.sqrt(TEST_NORMALIZATION)


@pytest.fixture(scope="module")
def model_ms(test_ms, test_ms_info, point_source_skymodel, tmp_path_factory):
    """Return an MS whose DATA column holds the predicted sky model."""
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


def _solve_slot(slot, solve_type, mode, h5parm_path, **overrides):
    options = {
        "slot": slot,
        "solve_type": solve_type,
        "h5parm": h5parm_path.name,
        "h5parm_path": h5parm_path.as_posix(),
        "solint": 0,
        "mode": mode,
        "nchan": 0,
        "solutions_per_direction": [1],
        "smoothness_dd_factors": [1.0],
        "smoothnessconstraint": 0.0,
        "smoothnessreffrequency": None,
        "smoothnessrefdistance": None,
        "antennaconstraint": "[]",
        "keepmodel": None,
        "reusemodel": None,
        "datause": "full",
    }
    options.update(overrides)
    return options


def _solution_values(h5parm_path, soltab_name):
    """Return the unflagged values of a soltab in the sol000 solset."""
    with h5parm(h5parm_path) as solutions:
        soltab = solutions.getSolset("sol000").getSoltab(soltab_name)
        values = soltab.getValues()[0]
        weights = soltab.getValues(weight=True)[0]
    unflagged_values = values[weights > 0]
    assert unflagged_values.size > 0, f"No unflagged values in {h5parm_path}: {soltab_name}"
    return unflagged_values


@pytest.fixture(scope="module")
def calibrate(model_ms, test_ms_info, point_source_skymodel, tmp_path_factory):
    """Return a function that runs a fast-phase plus slow-gain chunk with DP3.

    The function takes a name and the normalization h5parm (or None) and returns
    the paths of the fast-phase and slow-gain h5parms written by DP3.
    """

    def run(name, normalize_h5parm):
        workdir = tmp_path_factory.mktemp(name)
        fast = workdir / "fast_phase_0.h5parm"
        slow = workdir / "slow_gain_0.h5parm"
        payload = {
            "mode": "dd",
            "calibration_kind": "dd_phase_slow",
            "pipeline_working_dir": workdir.as_posix(),
            "data_colname": "DATA",
            "modeldatacolumn": None,
            "dp3_steps": "[solve1,solve2]",
            "image_based_predict": False,
            "wsclean_predict": False,
            "max_threads": 2,
            "maxiter": 100,
            "llssolver": "qr",
            "propagatesolutions": False,
            "solveralgorithm": "directionsolve",
            "solverlbfgs_dof": 200.0,
            "solverlbfgs_iter": 4,
            "solverlbfgs_minibatches": 1,
            "stepsize": 0.2,
            "stepsigma": 0.0,
            "tolerance": 1e-6,
            "uvlambdamin": 0.0,
            "correctfreqsmearing": False,
            "correcttimesmearing": False,
            "sourcedb": point_source_skymodel,
            "directions": [TEST_DIRECTION_NAME],
            "onebeamperpatch": True,
            "parallelbaselines": False,
            "sagecalpredict": False,
            "applycal_steps": None,
            "applycal_h5parm": None,
            "fulljones_h5parm": None,
            "normalize_h5parm": normalize_h5parm,
            "calibration_applycal_steps": None if normalize_h5parm is None else "[normalization]",
        }
        chunk = {
            "msin": model_ms,
            "starttime": test_ms_info["starttime"],
            "ntimes": 0,
            "solve_slots": [
                _solve_slot(1, "fast_phase", "scalarphase", fast, keepmodel="True"),
                _solve_slot(2, "slow_gains", "diagonal", slow, reusemodel="[solve1.*]"),
            ],
        }
        records = run_calibrate_chunk(payload, chunk)
        return records["solve1"]["path"], records["solve2"]["path"]

    return run


def test_slow_gain_amplitudes_are_unity_without_normalization(calibrate):
    fast, slow = calibrate("plain", None)

    np.testing.assert_allclose(_solution_values(fast, "phase000"), 0.0, atol=0.02)
    np.testing.assert_allclose(_solution_values(slow, "amplitude000"), 1.0, rtol=0.02)


def test_slow_gain_amplitudes_absorb_the_inverse_normalization(calibrate, normalization_h5parm):
    """The normalized model reaches the slow-gain solve through the kept model."""
    fast, slow = calibrate("normalized", normalization_h5parm)

    # A per-frequency scalar on the model leaves the phase-only solve unchanged.
    np.testing.assert_allclose(_solution_values(fast, "phase000"), 0.0, atol=0.02)
    np.testing.assert_allclose(
        _solution_values(slow, "amplitude000"), EXPECTED_NORMALIZED_AMPLITUDE, rtol=0.02
    )
