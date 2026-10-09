"""Tests for gain-processing execution helpers."""

import numpy as np
import pytest

import rapthor.execution.calibrate.gain_processing as gain_processing
from rapthor.execution.calibrate.gain_processing import (
    flag_amps,
    get_angular_distance,
    get_median_amp,
    get_smooth_box_size,
    normalize_direction,
    process_gain_solutions,
    smooth_solutions,
    transfer_flags,
)


class FakeSolset:
    def __init__(self, soltabs=None, sources=None):
        self.soltabs = soltabs or {}
        self.sources = sources or {}

    def getSoltab(self, name):
        return self.soltabs[name]

    def getSou(self):
        return self.sources


class FakeSoltab:
    def __init__(
        self,
        vals,
        weights=None,
        ants=None,
        dirs=None,
        times=None,
        freqs=None,
        solset=None,
    ):
        self.val = np.array(vals, dtype=float)
        self.weight = (
            np.ones_like(self.val, dtype=float)
            if weights is None
            else np.array(weights, dtype=float)
        )
        self.ant = list(ants or ["CS001"])
        self.dir = list(dirs or ["[Patch_0]"])
        self.time = np.array(times if times is not None else range(self.val.shape[0]), dtype=float)
        self.freq = np.array(freqs if freqs is not None else range(self.val.shape[1]), dtype=float)
        self._solset = solset or FakeSolset()

    def getSolset(self):
        return self._solset

    def setValues(self, values, weight=False):
        if weight:
            self.weight = np.array(values, dtype=float)
        else:
            self.val = np.array(values, dtype=float)


def test_get_angular_distance():
    ra_dec1 = (0, 0)
    ra_dec2 = (45, 45)
    dist = get_angular_distance(ra_dec1, ra_dec2)
    assert np.isclose(dist, 60), f"Expected angular distance 60 degrees, got {dist}"


def test_normalize_direction_sets_station_medians_to_unity():
    vals = np.ones((2, 1, 2, 1, 2), dtype=float)
    vals[:, :, 0, :, :] = 2.0
    vals[:, :, 1, :, :] = 4.0
    soltab = FakeSoltab(vals, ants=["CS001", "RS001"])

    normalize_direction(soltab)

    assert np.allclose(soltab.val, 1.0)


def test_normalize_direction_requires_phase_center_when_scaling():
    vals = np.ones((2, 1, 1, 2, 2), dtype=float)
    soltab = FakeSoltab(vals, dirs=["[Patch_0]", "[Patch_1]"])

    with pytest.raises(ValueError, match="phase_center"):
        normalize_direction(soltab, scale_delta_with_dist=True)


def test_smooth_solutions_smooths_core_station_only(monkeypatch):
    vals = np.ones((3, 3, 2, 1, 1), dtype=float)
    vals[1, 1, 0, 0, 0] = 10.0
    vals[1, 1, 1, 0, 0] = 10.0
    soltab = FakeSoltab(vals, ants=["CS001", "RS001"])

    monkeypatch.setattr(gain_processing, "get_smooth_box_size", lambda *args, **kwargs: 3)

    smooth_solutions(soltab)

    assert soltab.val[1, 1, 0, 0, 0] == pytest.approx(1.0)
    assert soltab.val[1, 1, 1, 0, 0] == pytest.approx(10.0)


def test_get_smooth_box_size_uses_minimum_for_low_noise_solutions():
    vals = np.ones((4, 3, 2, 1, 2), dtype=float)
    soltab = FakeSoltab(vals, ants=["CS001", "RS001"])

    assert get_smooth_box_size(soltab, direction=0, min_box_size=3) == 3
    assert get_smooth_box_size(soltab, direction=0, ant_list=["MISSING"]) is None


def test_get_median_amp():
    amps = np.array([[[1.0, 4.0], [9.0, 16.0]]])
    weights = np.ones_like(amps)

    assert get_median_amp(amps, weights) == pytest.approx(5.5)


def test_flag_amps_flags_values_outside_thresholds():
    vals = np.array([[[[[0.4, 1.0]]]], [[[[6.0, np.nan]]]]])
    weights = np.array([[[[[1.0, 1.0]]]], [[[[1.0, 0.0]]]]])
    soltab = FakeSoltab(vals, weights=weights)

    flag_amps(soltab, lowampval=0.5, highampval=5.0)

    assert np.isnan(soltab.val[0, 0, 0, 0, 0])
    assert soltab.weight[0, 0, 0, 0, 0] == 0.0
    assert soltab.val[0, 0, 0, 0, 1] == pytest.approx(1.0)
    assert soltab.weight[0, 0, 0, 0, 1] == 1.0
    assert np.isnan(soltab.val[1, 0, 0, 0, 0])
    assert soltab.weight[1, 0, 0, 0, 0] == 0.0
    assert np.isnan(soltab.val[1, 0, 0, 0, 1])
    assert soltab.weight[1, 0, 0, 0, 1] == 0.0


def test_flag_amps_rejects_invalid_threshold_factor():
    soltab = FakeSoltab(np.ones((1, 1, 1, 1, 2), dtype=float))

    with pytest.raises(SystemExit, match="threshold_factor"):
        flag_amps(soltab, threshold_factor=1.0)


def test_transfer_flags_copies_nan_and_zero_weight_flags():
    source_vals = np.ones((2, 1, 1, 1, 2), dtype=float)
    source_weights = np.ones_like(source_vals)
    source_vals[0, 0, 0, 0, 0] = np.nan
    source_weights[1, 0, 0, 0, 1] = 0.0
    source = FakeSoltab(source_vals, weights=source_weights)
    target = FakeSoltab(np.full_like(source_vals, 2.0), weights=np.ones_like(source_vals))

    transfer_flags(source, target)

    assert np.isnan(target.val[0, 0, 0, 0, 0])
    assert target.weight[0, 0, 0, 0, 0] == 0.0
    assert np.isnan(target.val[1, 0, 0, 0, 1])
    assert target.weight[1, 0, 0, 0, 1] == 0.0
    assert target.val[0, 0, 0, 0, 1] == pytest.approx(2.0)
    assert target.weight[0, 0, 0, 0, 1] == 1.0


def test_process_gain_solutions_dispatches_requested_processing_steps(monkeypatch):
    ampsoltab = FakeSoltab(np.ones((1, 1, 1, 1, 2), dtype=float))
    phasesoltab = FakeSoltab(np.ones((1, 1, 1, 1, 2), dtype=float))
    solset = FakeSolset({"amplitude000": ampsoltab, "phase000": phasesoltab})
    calls = []

    class FakeH5parm:
        def __init__(self, filename, readonly=False):
            calls.append(("open", filename, readonly))

        def getSolset(self, name):
            calls.append(("getSolset", name))
            return solset

        def close(self):
            calls.append(("close",))

    monkeypatch.setattr(gain_processing, "h5parm", FakeH5parm)
    monkeypatch.setattr(
        gain_processing.misc,
        "get_reference_station",
        lambda soltab, max_ind: calls.append(("reference", soltab, max_ind)) or 1,
    )
    monkeypatch.setattr(
        gain_processing,
        "flag_amps",
        lambda soltab, lowampval=None, highampval=None: calls.append(
            ("flag", soltab, lowampval, highampval)
        ),
    )
    monkeypatch.setattr(
        gain_processing,
        "transfer_flags",
        lambda soltab1, soltab2: calls.append(("transfer", soltab1, soltab2)),
    )
    monkeypatch.setattr(
        gain_processing,
        "smooth_solutions",
        lambda amps, phasesoltab=None, ref_id=0: calls.append(
            ("smooth", amps, phasesoltab, ref_id)
        ),
    )
    monkeypatch.setattr(
        gain_processing,
        "normalize_direction",
        lambda soltab, **kwargs: calls.append(("normalize", soltab, kwargs)),
    )

    process_gain_solutions(
        "solutions.h5",
        ref_id=None,
        smooth="True",
        normalize="True",
        flag="True",
        lowampval=0.5,
        highampval=2.0,
        max_station_delta=0.25,
        scale_delta_with_dist="True",
        phase_center=(12.0, -30.0),
    )

    assert calls == [
        ("open", "solutions.h5", False),
        ("getSolset", "sol000"),
        ("reference", phasesoltab, 10),
        ("flag", ampsoltab, 0.5, 2.0),
        ("transfer", ampsoltab, phasesoltab),
        ("smooth", ampsoltab, phasesoltab, 1),
        (
            "normalize",
            ampsoltab,
            {
                "max_station_delta": 0.25,
                "scale_delta_with_dist": True,
                "phase_center": (12.0, -30.0),
            },
        ),
        ("close",),
    ]
