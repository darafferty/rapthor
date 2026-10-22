"""Real DP3 checks for the prepare-imaging-data command.

These tests run the production preparation task (``prepare_visibility_ms``) on
the predicted point-source model and verify that the selected applycal steps
apply the combined DD solution product exactly once.
"""

from pathlib import Path

import casacore.tables as pt
import numpy as np
import pytest

from rapthor.execution.config import ExecutionConfig
from rapthor.execution.image.preparation import prepare_visibility_ms
from rapthor.operations.image.plan import build_image_applycal_steps
from tests.conftest import write_scalar_h5parm
from tests.execution.conftest import TEST_DIRECTION_NAME, TEST_PATCH_NAME, requires_dp3

pytestmark = [pytest.mark.integration, requires_dp3]

PHASE_STEP_RAD = 0.1
AMPLITUDE = 2.0
# The prepared MS is written with Dysco compression, which perturbs the
# visibilities at the few-per-mille level.
TOLERANCE = 1e-2


@pytest.fixture(scope="module")
def phasecenter(test_ms_info):
    return f"[{np.degrees(test_ms_info['ra'])}deg, {np.degrees(test_ms_info['dec'])}deg]"


def _dd_product(path, test_ms_info, *, with_amplitudes, unit=False):
    """Write a combined DD product with phase (i * 0.1 rad) and amplitude 2 per antenna i."""
    antennas = test_ms_info["antennas"]
    shape = (len(test_ms_info["times"]), len(test_ms_info["frequencies"]), len(antennas), 1)
    phases = (
        np.zeros(shape) if unit else np.arange(len(antennas))[None, None, :, None] * PHASE_STEP_RAD
    )
    amplitudes = np.ones(shape) if unit else np.full(shape, AMPLITUDE)
    return write_scalar_h5parm(
        path,
        antennas,
        with_amplitudes=with_amplitudes,
        phases=np.broadcast_to(phases, shape),
        amplitudes=amplitudes,
        directions=[TEST_DIRECTION_NAME],
        times=test_ms_info["times"],
        frequencies=test_ms_info["frequencies"],
    )


@pytest.fixture
def prepare(predicted_model_ms, test_ms_info, phasecenter, tmp_path):
    """Return a function that prepares the model MS for imaging with DP3.

    It takes a name, the solution h5parm and the DP3 applycal step list, runs
    the production task, and returns DATA, ANTENNA1 and ANTENNA2 of the result
    with flagged visibilities set to NaN. The fixture is function scoped
    because the production task reuses an existing output directory.
    """
    workdir = tmp_path / "prepared"
    workdir.mkdir()

    def run(name, h5parm, applycal_steps):
        msout = f"{name}.ms"
        sector = {
            "data_colname": "DATA",
            "phasecenter": phasecenter,
            "max_threads": 2,
            "prepare_data_steps": "[applybeam,shift,applycal]",
            "prepare_data_h5parm": h5parm,
            "prepare_data_applycal_steps": applycal_steps,
            "central_patch_name": TEST_PATCH_NAME,
        }
        task = {
            "msin": predicted_model_ms,
            "msout": msout,
            "msout_path": (workdir / msout).as_posix(),
            "starttime": test_ms_info["starttime"],
            "ntimes": 0,
            "freqstep": 1,
            "timestep": 1,
            "minchannels": 1,
        }
        record = prepare_visibility_ms(
            sector, task, workdir.as_posix(), ExecutionConfig(task_runner="sync")
        )
        with pt.table(record["path"], ack=False) as table:
            data = np.where(table.getcol("FLAG"), np.nan, table.getcol("DATA"))
            return data, table.getcol("ANTENNA1"), table.getcol("ANTENNA2")

    return run


def _baseline_ratio(corrected, reference):
    data, antenna1, antenna2 = corrected
    ref_data = reference[0]
    valid = np.isfinite(data) & np.isfinite(ref_data) & (np.abs(ref_data) > 1e-6)
    ratio = np.full(data.shape, np.nan, dtype=complex)
    ratio[valid] = data[valid] / ref_data[valid]
    # One value per row: the ratio is the same for all channels and correlations.
    row_ratio = np.nanmean(ratio.reshape(len(antenna1), -1), axis=1)
    return row_ratio, antenna1, antenna2


def _assert_phase_applications(corrected, reference, *, applications):
    """Check how many times the per-antenna phases were applied; return the valid ratios.

    An inverted correction of per-antenna phases phi_i shifts a baseline (p, q)
    by -(phi_p - phi_q) per application.
    """
    row_ratio, antenna1, antenna2 = _baseline_ratio(corrected, reference)
    valid = np.isfinite(row_ratio)
    # Guard against a vacuous pass: fully flagged output would leave nothing to
    # compare, and autocorrelations carry no phase difference.
    assert np.any(valid & (antenna1 != antenna2)), "no unflagged cross-antenna rows to compare"
    expected_phase = -applications * (antenna1 - antenna2) * PHASE_STEP_RAD
    phase_error = np.angle(row_ratio[valid] * np.exp(-1j * expected_phase[valid]))
    np.testing.assert_allclose(phase_error, 0.0, atol=TOLERANCE)
    return row_ratio[valid]


@pytest.mark.parametrize(
    "strategy, with_amplitudes, dd_apply_amplitudes, expected_steps",
    [
        ({"dd": ["fast_phase", "medium_phase"]}, False, False, "[fastphase]"),
        ({"dd": ["fast_phase", "medium_phase", "slow_gains"]}, True, True, "[fastphase,slowgain]"),
    ],
)
def test_prepare_imaging_applies_the_combined_dd_product_once(
    prepare, test_ms_info, tmp_path, strategy, with_amplitudes, dd_apply_amplitudes, expected_steps
):
    """dde_method = single applies the combined phases (and amplitudes) exactly once."""
    unit = _dd_product(
        tmp_path / "unit.h5", test_ms_info, with_amplitudes=with_amplitudes, unit=True
    )
    product = _dd_product(tmp_path / "product.h5", test_ms_info, with_amplitudes=with_amplitudes)
    steps, selected = build_image_applycal_steps(
        strategy,
        dd_h5parm=product,
        di_h5parm=None,
        has_fulljones_h5parm=False,
        use_facets=False,
        apply_amplitudes=dd_apply_amplitudes,
        apply_normalizations=False,
        apply_none=False,
        dd_apply_amplitudes=dd_apply_amplitudes,
    )
    applycal_steps = f"[{','.join(steps)}]"
    assert applycal_steps == expected_steps
    assert selected == product

    reference = prepare("reference", unit, applycal_steps)
    corrected = prepare("corrected", product, applycal_steps)

    row_ratio = _assert_phase_applications(corrected, reference, applications=1)
    expected_amplitude = 1.0 / AMPLITUDE**2 if with_amplitudes else 1.0
    np.testing.assert_allclose(np.abs(row_ratio), expected_amplitude, rtol=TOLERANCE)


def test_duplicate_phase_step_applies_the_phases_twice(prepare, test_ms_info, tmp_path):
    """Lock the DP3 behaviour that made the former [fastphase,mediumphase] list wrong."""
    unit = _dd_product(tmp_path / "unit.h5", test_ms_info, with_amplitudes=False, unit=True)
    product = _dd_product(tmp_path / "product.h5", test_ms_info, with_amplitudes=False)

    reference = prepare("reference", unit, "[fastphase,mediumphase]")
    corrected = prepare("corrected", product, "[fastphase,mediumphase]")

    _assert_phase_applications(corrected, reference, applications=2)


def _measurement_set_state(path):
    return sorted(
        (str(item.relative_to(path)), item.stat().st_size, item.stat().st_mtime_ns)
        for item in Path(path).rglob("*")
        if item.is_file()
    )


def test_prepare_restart_remakes_a_stopped_ms_and_then_reuses_it(
    test_ms, test_ms_info, phasecenter, tmp_path
):
    """A restart replaces an MS left by a stopped attempt, then reuses the complete one."""
    workdir = tmp_path / "image_1"
    workdir.mkdir()
    sector = {
        "data_colname": "DATA",
        "phasecenter": phasecenter,
        "max_threads": 2,
        "prepare_data_steps": "[shift]",
    }
    task = {
        "msin": test_ms,
        "msout": "prep.ms",
        "msout_path": (workdir / "prep.ms").as_posix(),
        "starttime": test_ms_info["starttime"],
        "ntimes": 0,
        "freqstep": 1,
        "timestep": 1,
        "minchannels": 1,
    }
    input_state = _measurement_set_state(test_ms)
    # A stopped attempt left an incomplete MS at the final path and a partial one.
    (workdir / "prep.ms").mkdir()
    (workdir / "prep.ms" / "table.dat").write_text("incomplete")
    (workdir / ".prep.ms.partial").mkdir()

    config = ExecutionConfig(task_runner="sync")
    record = prepare_visibility_ms(sector, task, workdir.as_posix(), config)
    with pt.table(record["path"], ack=False) as table:
        assert table.nrows() == len(test_ms_info["antenna1"])
    product_state = _measurement_set_state(record["path"])

    assert prepare_visibility_ms(sector, task, workdir.as_posix(), config) == record
    assert _measurement_set_state(record["path"]) == product_state
    assert not (workdir / ".prep.ms.partial").exists()
    assert _measurement_set_state(test_ms) == input_state
