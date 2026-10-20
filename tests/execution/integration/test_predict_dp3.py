"""Integration tests for the model data prediction task.

Tests for the pipeline's predict task (``run_predict_model_data``), which
build the DP3 command, run DP3 and validate the output. Checks ensure
that the requested corrections are actually applied to the visibilities written
to the output measurement set.
"""

import casacore.tables as pt
import numpy as np
import pytest
from losoto.h5parm import h5parm

from rapthor.execution.predict.commands import build_predict_model_data_command
from rapthor.execution.predict.flow import run_predict_model_data
from rapthor.execution.shell import ShellCommandError
from tests.execution.conftest import TEST_DIRECTION_NAME, TEST_NORMALIZATION, requires_dp3
from tests.execution.payload_factories import representative_predict_payload

pytestmark = [pytest.mark.integration, requires_dp3]


class _SagecalNormalizationNotApplied(RuntimeError):
    """DP3's SAGECal predictor ignored the separate normalization H5parm."""


class _SagecalApplycalStepLimit(RuntimeError):
    """DP3's SAGECal predictor rejected more than two applycal steps."""


class _SagecalRequiresAmplitudeFirst(RuntimeError):
    """DP3's SAGECal predictor read zero phases as zero amplitudes."""


@pytest.fixture(scope="module")
def unity_dd_h5parm(test_ms_info, tmp_path_factory):
    """Write a direction-dependent h5parm with zero phases and unit amplitudes."""
    path = tmp_path_factory.mktemp("solutions") / "solutions.h5"
    gains = np.ones((len(test_ms_info["antennas"]), 2), dtype=complex)
    return _write_dd_h5parm(path, test_ms_info, gains)


@pytest.fixture(scope="module")
def diagonal_dd_solutions(test_ms_info, tmp_path_factory):
    """Give each antenna and polarization a distinct amplitude and phase."""
    path = tmp_path_factory.mktemp("diagonal_solutions") / "solutions.h5"
    nant = len(test_ms_info["antennas"])
    amplitudes = np.linspace(1.1, 1.9, nant)[:, None] * [1.0, 1.2]
    phases = np.linspace(-0.4, 0.6, nant)[:, None] + [0.0, 0.3]
    gains = amplitudes * np.exp(1j * phases)
    return _write_dd_h5parm(path, test_ms_info, gains), gains


def _write_dd_h5parm(path, test_ms_info, gains):
    """Write diagonal gains that are constant in time and frequency."""
    axes = {
        "time": test_ms_info["times"],
        "freq": test_ms_info["frequencies"],
        "ant": test_ms_info["antennas"],
        "dir": [TEST_DIRECTION_NAME],
        "pol": ["XX", "YY"],
    }
    shape = tuple(len(values) for values in axes.values())
    with h5parm(path.as_posix(), readonly=False) as output:
        solset = output.makeSolset("sol000")
        solset.obj._f_get_child("source").append(
            [(TEST_DIRECTION_NAME, (test_ms_info["ra"], test_ms_info["dec"]))]
        )
        solset.obj._f_get_child("antenna").append(
            [(name, np.zeros(3)) for name in test_ms_info["antennas"]]
        )
        for soltype, values in [("phase", np.angle(gains)), ("amplitude", np.abs(gains))]:
            solset.makeSoltab(
                soltype,
                f"{soltype}000",
                axesNames=list(axes),
                axesVals=list(axes.values()),
                vals=np.broadcast_to(values[None, None, :, None, :], shape).copy(),
                weights=np.ones(shape),
            )
    return path.as_posix()


@pytest.fixture(scope="module")
def predict(test_ms, test_ms_info, point_source_skymodel, tmp_path_factory):
    """Return a function that runs one production predict task with DP3.

    The function takes the output MS name and any optional task fields, runs
    ``run_predict_model_data`` as the Prefect task does, and returns the
    written visibilities with flagged values set to NaN.
    """
    workdir = tmp_path_factory.mktemp("predictions")
    with pt.table(test_ms, ack=False) as table:
        unflagged = ~table.getcol("FLAG")
    assert unflagged.any(), "The input MS has no unflagged visibilities"

    def run(name, **task_overrides):
        msout = f"{name}.ms"
        task = representative_predict_payload(workdir)["predict_tasks"][0]
        task.update(
            {
                "msin": test_ms,
                "msout": msout,
                "msout_path": (workdir / msout).as_posix(),
                "starttime": test_ms_info["starttime"],
                "ntimes": 0,
                "sourcedb": point_source_skymodel,
                "directions": [TEST_DIRECTION_NAME],
                "numthreads": 2,
                **task_overrides,
            }
        )
        record = run_predict_model_data(task, workdir.as_posix())
        with pt.table(record["path"], ack=False) as table:
            data = table.getcol("DATA")
            flags = table.getcol("FLAG")
        assert data.shape == flags.shape == unflagged.shape
        # These complete, unflagged solutions should preserve every input sample.
        # Check against the input flags so two equally broken predictions cannot
        # hide missing samples from the ratio assertions.
        assert not flags[unflagged].any(), "Prediction flagged previously unflagged samples"
        assert np.isfinite(data[unflagged]).all(), "Prediction contains non-finite samples"
        return np.where(unflagged, data, np.nan)

    return run


@pytest.fixture(scope="module")
def plain_prediction(predict):
    """Predict once without any corrections for use as the reference."""
    return predict("plain")


@pytest.fixture(scope="module")
def sagecal_prediction(predict, unity_dd_h5parm):
    """Use SAGECal itself as the reference to exclude differences between predictors."""
    return predict(
        "sagecal_phase",
        sagecalpredict=True,
        h5parm=unity_dd_h5parm,
        applycal_steps="[fastphase]",
    )


def _visibility_ratio(numerator, denominator):
    assert numerator.shape == denominator.shape
    unflagged = np.isfinite(denominator)
    assert np.isfinite(numerator[unflagged]).all(), "Prediction lost reference samples"
    # Zero reference visibilities cannot constrain a multiplicative correction.
    valid = unflagged & (np.abs(denominator) > 1e-6)
    assert valid.any(), "The reference prediction has no usable visibilities"
    return numerator[valid] / denominator[valid]


def _assert_ratio(numerator, denominator, expected):
    np.testing.assert_allclose(
        _visibility_ratio(numerator, denominator), expected, rtol=1e-4, atol=0
    )


def test_predict_normalization_scales_written_visibilities(
    predict, plain_prediction, normalization_h5parm
):
    """A DI prediction with a normalization h5parm scales the model by the normalization."""
    normalized = predict(
        "di_normalized",
        applycal_steps="[normalization]",
        normalize_h5parm=normalization_h5parm,
    )

    _assert_ratio(normalized, plain_prediction, TEST_NORMALIZATION)


def test_h5parmpredict_normalization_scales_written_visibilities(
    predict, plain_prediction, normalization_h5parm, unity_dd_h5parm
):
    """A DD prediction applies the normalization alongside the DD solutions."""
    corrupted = predict(
        "dd_corrupted",
        h5parm=unity_dd_h5parm,
        applycal_steps="[fastphase,slowgain]",
    )
    normalized = predict(
        "dd_normalized",
        h5parm=unity_dd_h5parm,
        applycal_steps="[fastphase,slowgain,normalization]",
        normalize_h5parm=normalization_h5parm,
    )

    # Unit DD solutions must leave the prediction unchanged; adding the
    # normalization must scale it.
    _assert_ratio(corrupted, plain_prediction, 1.0)
    _assert_ratio(normalized, plain_prediction, TEST_NORMALIZATION)


@pytest.mark.parametrize(
    "applycal_steps",
    [
        pytest.param("[slowgain,fastphase]", id="amplitude-phase"),
        pytest.param("[fastphase,slowgain]", id="phase-amplitude"),
    ],
)
def test_sagecalpredict_unity_solutions_preserve_visibilities(
    predict, sagecal_prediction, unity_dd_h5parm, applycal_steps
):
    """Named unity corrections should preserve the model in either order."""
    corrupted = predict(
        f"sagecal_unity_{applycal_steps[1:-1].replace(',', '_')}",
        sagecalpredict=True,
        h5parm=unity_dd_h5parm,
        applycal_steps=applycal_steps,
    )

    _assert_ratio(corrupted, sagecal_prediction, 1.0)


def test_sagecalpredict_applies_diagonal_amplitude_and_phase(
    predict, sagecal_prediction, diagonal_dd_solutions, test_ms_info
):
    """The production step order must apply both parts of each complex gain."""
    solutions, gains = diagonal_dd_solutions
    corrupted = predict(
        "sagecal_diagonal",
        sagecalpredict=True,
        h5parm=solutions,
        applycal_steps="[fastphase,slowgain]",
    )

    # V_pq = G_p V G_q^H, in the test MS's XX, XY, YX, YY correlation order.
    baseline_gains = (
        gains[test_ms_info["antenna1"], :, None] * gains[test_ms_info["antenna2"], None, :].conj()
    )
    expected = sagecal_prediction * baseline_gains.reshape(-1, 1, 4)
    _assert_ratio(corrupted, expected, 1.0)


@pytest.mark.xfail(
    strict=True,
    raises=_SagecalRequiresAmplitudeFirst,
    reason="DP3 SAGECalPredict reads the first of two soltabs as amplitude regardless of type.",
)
def test_dp3_sagecalpredict_accepts_phase_first_soltabs(
    predict, sagecal_prediction, unity_dd_h5parm, monkeypatch
):
    """Flag an upstream fix so Rapthor's ordering workaround can be revisited."""

    def phase_first_command(*args, **kwargs):
        command = build_predict_model_data_command(*args, **kwargs)
        index = next(
            i for i, token in enumerate(command) if token.startswith("predict.applycal.steps=")
        )
        # Bypass only Rapthor's ordering workaround; run real DP3 and retain
        # the fixture's checks for missing, flagged or non-finite samples.
        command[index] = "predict.applycal.steps=[fastphase,slowgain]"
        return command

    monkeypatch.setattr(
        "rapthor.execution.predict.flow.build_predict_model_data_command", phase_first_command
    )
    corrupted = predict(
        "dp3_phase_first",
        sagecalpredict=True,
        h5parm=unity_dd_h5parm,
        applycal_steps="[fastphase,slowgain]",
    )

    ratio = _visibility_ratio(corrupted, sagecal_prediction)
    if np.all(ratio == 0):
        raise _SagecalRequiresAmplitudeFirst("DP3 read the zero phases as zero amplitudes")
    # Correct output becomes a strict XPASS, failing CI when DP3 fixes this.
    np.testing.assert_allclose(ratio, 1.0, rtol=1e-4, atol=0)


@pytest.mark.xfail(
    strict=True,
    raises=_SagecalApplycalStepLimit,
    reason="DP3 SAGECalPredict rejects more than two applycal steps.",
)
def test_sagecalpredict_normalization_scales_written_visibilities(
    predict, sagecal_prediction, normalization_h5parm, unity_dd_h5parm, caplog
):
    """SAGECal should apply normalization alongside fast-phase and slow-gain solutions."""
    caplog.set_level("INFO", logger="rapthor:shell")
    try:
        normalized = predict(
            "sagecal_normalized",
            sagecalpredict=True,
            h5parm=unity_dd_h5parm,
            applycal_steps="[fastphase,slowgain,normalization]",
            normalize_h5parm=normalization_h5parm,
        )
    except ShellCommandError as error:
        # Only the known DP3 limitation should be treated as an expected failure.
        message = "applycal.steps should specify only one or two steps"
        if message in caplog.text:
            raise _SagecalApplycalStepLimit(message) from error
        raise

    _assert_ratio(normalized, sagecal_prediction, TEST_NORMALIZATION)


@pytest.mark.xfail(
    strict=True,
    raises=_SagecalNormalizationNotApplied,
    reason="DP3 SAGECalPredict ignores the separate normalization H5parm in a two-step prediction.",
)
def test_sagecalpredict_normalization_uses_separate_h5parm(
    predict, sagecal_prediction, normalization_h5parm, unity_dd_h5parm
):
    """The normalization must come from its own file, not the main solutions file."""
    # Use the amplitude-first order required by DP3's SAGECal reader to isolate
    # file selection from the two-table ordering limitation. The main H5parm contains
    # unit amplitudes, while the separate normalization should scale by four.
    normalized = predict(
        "sagecal_phase_normalized",
        sagecalpredict=True,
        h5parm=unity_dd_h5parm,
        applycal_steps="[normalization,fastphase]",
        normalize_h5parm=normalization_h5parm,
    )

    ratio = _visibility_ratio(normalized, sagecal_prediction)

    # Only a finite prediction at the original flux scale is the known failure.
    # Zero output, missing samples and other incorrect ratios must fail normally.
    if np.allclose(ratio, 1.0, rtol=1e-4, atol=0):
        raise _SagecalNormalizationNotApplied("SAGECal ignored the normalization H5parm")
    np.testing.assert_allclose(ratio, TEST_NORMALIZATION, rtol=1e-4, atol=0)
