"""Real DP3 checks for the prediction command.

These tests run the production predict task (``run_predict_model_data``, which
builds the command, runs DP3 through the shell runner and validates the output)
on the small test Measurement Set, and verify that the requested corrections
reach the written visibilities. They cover the buffer handling of the command,
not the scientific equivalence of the beam or the predictors.
"""

import casacore.tables as pt
import numpy as np
import pytest
from losoto.h5parm import h5parm

from rapthor.execution.predict.flow import run_predict_model_data
from tests.execution.conftest import TEST_DIRECTION_NAME, TEST_NORMALIZATION, requires_dp3
from tests.execution.payload_factories import representative_predict_payload

pytestmark = [pytest.mark.integration, requires_dp3]


@pytest.fixture(scope="module")
def unity_dd_h5parm(test_ms_info, tmp_path_factory):
    """Write a direction-dependent h5parm with zero phases and unit amplitudes."""
    path = tmp_path_factory.mktemp("solutions") / "solutions.h5"
    axes = {
        "time": test_ms_info["times"],
        "freq": test_ms_info["frequencies"],
        "ant": test_ms_info["antennas"],
        "dir": [TEST_DIRECTION_NAME],
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
        for soltype, value in [("phase", 0.0), ("amplitude", 1.0)]:
            solset.makeSoltab(
                soltype,
                f"{soltype}000",
                axesNames=list(axes),
                axesVals=list(axes.values()),
                vals=np.full(shape, value),
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
            return np.where(table.getcol("FLAG"), np.nan, table.getcol("DATA"))

    return run


@pytest.fixture(scope="module")
def plain_prediction(predict):
    """Predict once without any corrections for use as the reference."""
    return predict("plain")


def _assert_ratio(numerator, denominator, expected):
    valid = np.isfinite(numerator) & np.isfinite(denominator) & (np.abs(denominator) > 1e-6)
    assert valid.any()
    np.testing.assert_allclose(numerator[valid] / denominator[valid], expected, rtol=1e-4)


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
