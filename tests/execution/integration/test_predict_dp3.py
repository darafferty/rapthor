"""Integration tests for the model data prediction task.

Tests for the pipeline's predict task (``run_predict_model_data``), which
builds the DP3 command, runs DP3 and validates the output. Checks ensure
that the requested corrections are actually applied to the visibilities written
to the output measurement set.
"""

import shutil

import casacore.tables as pt
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from losoto.h5parm import h5parm

from rapthor.execution.image.flux_normalization import create_normalization_h5parm
from rapthor.execution.predict.flow import run_predict_model_data
from rapthor.lib import miscellaneous as misc
from tests.execution.payload_factories import representative_predict_payload

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(shutil.which("DP3") is None, reason="DP3 is not available"),
]

PATCH_NAME = "Patch_0"
DIRECTION_NAME = f"[{PATCH_NAME}]"
# Normalization factor for the normalize_h5parm fixture
NORMALIZATION = 4.0


@pytest.fixture(scope="module")
def ms_info(test_ms):
    """Return the MS quantities needed to build sky models and h5parms."""
    with pt.table(test_ms, ack=False) as table:
        times = np.unique(table.getcol("TIME"))
        interval = table.getcol("INTERVAL")[0]
    with pt.table(f"{test_ms}/SPECTRAL_WINDOW", ack=False) as spw_table:
        frequencies = spw_table.getcol("CHAN_FREQ")[0]
    with pt.table(f"{test_ms}/FIELD", ack=False) as table:
        ra, dec = np.squeeze(table.getcol("PHASE_DIR"))
    with pt.table(f"{test_ms}/ANTENNA", ack=False) as table:
        antennas = list(table.getcol("NAME"))
    return {
        # Start just before the first sample, as Observation does.
        "starttime": misc.convert_mjd2mvt(times[0] - interval / 100),
        "times": times,
        "frequencies": frequencies,
        "ra": ra,
        "dec": dec,
        "antennas": antennas,
    }


@pytest.fixture(scope="module")
def skymodel(ms_info, tmp_path_factory):
    """Write a one-patch sky model with a single point source at the phase centre."""
    coord = SkyCoord(ms_info["ra"], ms_info["dec"], unit="rad")
    ra = coord.ra.to_string(unit="hourangle", sep=":", precision=4)
    dec = coord.dec.to_string(unit="deg", sep=".", precision=4, alwayssign=True)
    path = tmp_path_factory.mktemp("skymodel") / "sector_1.skymodel"
    path.write_text(
        "FORMAT = Name, Type, Patch, Ra, Dec, I, SpectralIndex='[]', LogarithmicSI, "
        "ReferenceFrequency='134000000.0', MajorAxis, MinorAxis, Orientation\n"
        f" , , {PATCH_NAME}, {ra}, {dec}\n"
        f"source_0, POINT, {PATCH_NAME}, {ra}, {dec}, 1.0, [], false, 134000000.0, , , \n"
    )
    return path.as_posix()


@pytest.fixture(scope="module")
def normalize_h5parm(test_ms, ms_info, tmp_path_factory):
    """Write a normalization h5parm in the production layout (one amplitude per frequency)."""
    path = tmp_path_factory.mktemp("normalize") / "normalize.h5"
    create_normalization_h5parm(
        antenna_file=f"{test_ms}/ANTENNA",
        field_file=f"{test_ms}/FIELD",
        h5parm_file=path.as_posix(),
        frequencies=ms_info["frequencies"],
        normalizations=np.full(len(ms_info["frequencies"]), NORMALIZATION),
    )
    return path.as_posix()


@pytest.fixture(scope="module")
def unity_dd_h5parm(ms_info, tmp_path_factory):
    """Write a direction-dependent h5parm with zero phases and unit amplitudes."""
    path = tmp_path_factory.mktemp("solutions") / "solutions.h5"
    axes = {
        "time": ms_info["times"],
        "freq": ms_info["frequencies"],
        "ant": ms_info["antennas"],
        "dir": [DIRECTION_NAME],
    }
    shape = tuple(len(values) for values in axes.values())
    with h5parm(path.as_posix(), readonly=False) as output:
        solset = output.makeSolset("sol000")
        solset.obj._f_get_child("source").append(
            [(DIRECTION_NAME, (ms_info["ra"], ms_info["dec"]))]
        )
        solset.obj._f_get_child("antenna").append(
            [(name, np.zeros(3)) for name in ms_info["antennas"]]
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
def predict(test_ms, ms_info, skymodel, tmp_path_factory):
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
                "starttime": ms_info["starttime"],
                "ntimes": 0,
                "sourcedb": skymodel,
                "directions": [DIRECTION_NAME],
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
    predict, plain_prediction, normalize_h5parm
):
    """A DI prediction with a normalization h5parm scales the model by the normalization."""
    normalized = predict(
        "di_normalized",
        applycal_steps="[normalization]",
        normalize_h5parm=normalize_h5parm,
    )

    _assert_ratio(normalized, plain_prediction, NORMALIZATION)


def test_h5parmpredict_normalization_scales_written_visibilities(
    predict, plain_prediction, normalize_h5parm, unity_dd_h5parm
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
        normalize_h5parm=normalize_h5parm,
    )

    # Unit DD solutions must leave the prediction unchanged; adding the
    # normalization must scale it.
    _assert_ratio(corrupted, plain_prediction, 1.0)
    _assert_ratio(normalized, plain_prediction, NORMALIZATION)
