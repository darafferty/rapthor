"""Real DP3 checks for the calibration solve command.

These tests run the production calibration chunk task (``run_calibrate_chunk``)
with DP3 on the small test Measurement Set, with the data set equal to the
predicted sky model. They verify where the flux-scale normalization reaches the
solve model and that the solve leaves its input MS untouched, not the
scientific quality of the solutions.
"""

import shutil

import casacore.tables as pt
import numpy as np
import pytest
from losoto.h5parm import h5parm

from rapthor.execution.calibrate.solves import run_calibrate_chunk
from rapthor.operations.calibrate.plan import build_calibration_dp3_steps
from tests.conftest import write_scalar_h5parm
from tests.execution.conftest import TEST_DIRECTION_NAME, TEST_NORMALIZATION, requires_dp3

pytestmark = [pytest.mark.integration, requires_dp3]

# Diagonal gains solved against unscaled data come out at 1 / sqrt(normalization)
# when the model has been scaled by the normalization.
EXPECTED_NORMALIZED_AMPLITUDE = 1.0 / np.sqrt(TEST_NORMALIZATION)
# A DI amplitude pre-applied to the data through the stand-alone applycal step
# divides the data by its square, so the solved gains come out at its inverse.
TEST_DI_AMPLITUDE = 2.0
EXPECTED_PREAPPLIED_AMPLITUDE = 1.0 / TEST_DI_AMPLITUDE


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
def calibrate(predicted_model_ms, test_ms_info, point_source_skymodel, tmp_path_factory):
    """Return a function that runs a fast-phase plus slow-gain chunk with DP3.

    The function takes a name and the normalization h5parm (or None) and returns
    the paths of the fast-phase and slow-gain h5parms written by DP3. With a
    ``di_h5parm`` its amplitudes are pre-applied to the data through the
    stand-alone applycal step, as the pipeline does for DI solutions. The step
    chain comes from the production plan builder.
    """

    def run(name, normalize_h5parm, *, msin=None, di_h5parm=None):
        workdir = tmp_path_factory.mktemp(name)
        fast = workdir / "fast_phase_0.h5parm"
        slow = workdir / "slow_gain_0.h5parm"
        dp3_steps = build_calibration_dp3_steps(
            0,
            0,
            all_channels_regular=True,
            use_image_based_predict=False,
            solve_steps=["solve1", "solve2"],
            preapply_solutions=di_h5parm is not None,
        )
        payload = {
            "mode": "dd",
            "calibration_kind": "dd_phase_slow",
            "pipeline_working_dir": workdir.as_posix(),
            "data_colname": "DATA",
            "modeldatacolumn": None,
            "dp3_steps": f"[{','.join(dp3_steps)}]",
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
            "applycal_steps": None if di_h5parm is None else "[slowgain]",
            "applycal_h5parm": di_h5parm,
            "fulljones_h5parm": None,
            "normalize_h5parm": normalize_h5parm,
            "calibration_applycal_steps": None if normalize_h5parm is None else "[normalization]",
        }
        chunk = {
            "msin": predicted_model_ms if msin is None else msin,
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
    np.testing.assert_allclose(_solution_values(slow, "amplitude000"), 1.0, rtol=1e-4)


def test_slow_gain_amplitudes_absorb_the_inverse_normalization(calibrate, normalization_h5parm):
    """The normalized model reaches the slow-gain solve through the kept model."""
    fast, slow = calibrate("normalized", normalization_h5parm)

    # A per-frequency scalar on the model leaves the phase-only solve unchanged.
    np.testing.assert_allclose(_solution_values(fast, "phase000"), 0.0, atol=0.02)
    np.testing.assert_allclose(
        _solution_values(slow, "amplitude000"), EXPECTED_NORMALIZED_AMPLITUDE, rtol=1e-4
    )


def test_preapply_chain_leaves_the_input_ms_unchanged(
    calibrate, predicted_model_ms, test_ms_info, tmp_path
):
    """The stand-alone applycal step must not write corrected data back to msin.

    DP3 appends an in-place MS update when msout is empty and a step provides
    data. The calibration input can be an original input MS, so the chain must
    only read it.
    """
    antennas = test_ms_info["antennas"]
    di_h5parm = write_scalar_h5parm(
        tmp_path / "di_solutions.h5",
        antennas,
        with_amplitudes=True,
        amplitudes=TEST_DI_AMPLITUDE,
        directions=[TEST_DIRECTION_NAME],
        times=test_ms_info["times"],
        frequencies=test_ms_info["frequencies"],
    )
    msin = tmp_path / "input.ms"
    shutil.copytree(predicted_model_ms, msin)
    with pt.table(msin.as_posix(), ack=False) as table:
        data_before = table.getcol("DATA")

    _, slow = calibrate("preapply", None, msin=msin.as_posix(), di_h5parm=di_h5parm)

    with pt.table(msin.as_posix(), ack=False) as table:
        np.testing.assert_array_equal(table.getcol("DATA"), data_before)
    # The solve itself did see the pre-applied normalization.
    np.testing.assert_allclose(
        _solution_values(slow, "amplitude000"), EXPECTED_PREAPPLIED_AMPLITUDE, rtol=1e-4
    )
