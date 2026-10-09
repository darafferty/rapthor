"""Tests for the `rapthor.lib.miscellaneous` module."""

import multiprocessing
import os

import numpy as np
import pytest

import rapthor.lib.miscellaneous as miscellaneous
from rapthor.lib.miscellaneous import (
    angular_separation,
    approx_equal,
    calc_theoretical_noise,
    convert_mjd2mvt,
    convert_mvt2mjd,
    find_unflagged_fraction,
    get_flagged_solution_fraction,
    get_max_spectral_terms,
    get_reference_station,
    make_template_image,
    nproc,
    remove_soltabs,
    rename_skymodel_patches,
    string2bool,
    string2list,
)


def test_make_template_image_accepts_empty_output_name():
    make_template_image(
        image_name=None,
        reference_ra_deg=None,
        reference_dec_deg=None,
        ximsize=512,
        yimsize=512,
        cellsize_deg=0.000417,
        freqs=None,
        times=None,
        antennas=None,
        aterm_type="tec",
        fill_val=0,
    )


def test_string2bool_preserves_none():
    assert string2bool(None) is None


def test_string2list_preserves_none():
    assert string2list(None) is None


def test_approx_equal_accepts_relative_tolerance():
    assert approx_equal(1.23456789, 1.23457890, 1e-5, rel=1e-5)


@pytest.mark.filterwarnings("error")
def test_convert_mjd_mvt_round_trips_without_warning_for_modern_date():
    mjd_sec = 4567890123.125
    mvt_str = "18Aug2003/02:22:03.125"

    assert convert_mjd2mvt(mjd_sec) == mvt_str
    assert convert_mvt2mjd(mvt_str) == mjd_sec


@pytest.mark.filterwarnings("ignore:ERFA.*dubious year")
def test_convert_mjd_mvt_round_trips_mjd_zero():
    mjd_sec = 0
    mvt_str = "17Nov1858/00:00:00.000"

    assert convert_mjd2mvt(mjd_sec) == mvt_str
    assert convert_mvt2mjd(mvt_str) == mjd_sec


def test_angular_separation_returns_expected_quantity():
    assert angular_separation((0.0, 0.0), (45.0, 45.0)).value == pytest.approx(60.0)


def test_calc_theoretical_noise_returns_zero_for_empty_observation_list():
    assert calc_theoretical_noise([], w_factor=1.5) == (0.0, 0.0)


def test_get_reference_station_selects_least_flagged_antenna():
    class Soltab:
        ant = ["CS001", "CS002", "CS003"]

        def getValues(self, retAxesVals=False, weight=True):
            assert retAxesVals is False
            assert weight is True
            return np.array(
                [
                    [0.0, 1.0],
                    [1.0, 1.0],
                    [1.0, 0.0],
                ]
            )

        def getAxesNames(self):
            return ["ant", "time"]

    assert get_reference_station(Soltab(), max_ind=2) == 1


def test_remove_soltabs_deletes_named_solution_tables():
    class Soltab:
        def __init__(self):
            self.deleted = False

        def delete(self):
            self.deleted = True

    class Solset:
        def __init__(self):
            self.soltabs = {"phase": Soltab(), "amplitude": Soltab()}

        def getSoltab(self, name):
            return self.soltabs[name]

    solset = Solset()

    remove_soltabs(solset, "phase, amplitude")

    assert solset.soltabs["phase"].deleted is True
    assert solset.soltabs["amplitude"].deleted is True


def test_find_unflagged_fraction_runs_taql_for_time_range(monkeypatch):
    calls = []

    def fake_run(command, *, shell, capture_output, check):
        calls.append(command)
        assert shell is True
        assert capture_output is True
        assert check is True
        return miscellaneous.subprocess.CompletedProcess(command, 0, stdout=b"0.75")

    monkeypatch.setattr(miscellaneous.subprocess, "run", fake_run)

    assert find_unflagged_fraction("input.ms", 4567890123, 4567890134) == 0.75
    assert "input.ms" in calls[0]
    assert "TIME in [4567890123 =:= 4567890134]" in calls[0]


def test_get_flagged_solution_fraction_counts_nonfinite_and_zero_weight(monkeypatch):
    class Soltab:
        val = np.array([[1.0, np.nan], [3.0, 4.0]])
        weight = np.array([[1.0, 1.0], [0.0, 1.0]])

    class Solset:
        def getSoltabs(self):
            return [Soltab()]

    class H5Parm:
        def __init__(self, filename):
            self.filename = filename

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def getSolset(self, name):
            assert name == "sol000"
            return Solset()

    monkeypatch.setattr(miscellaneous, "h5parm", H5Parm)

    assert get_flagged_solution_fraction("solutions.h5") == 0.5


def test_rename_skymodel_patches_orders_sources_by_ra_within_dec_bin():
    class Coordinate:
        def __init__(self, value):
            self.value = value

    class SkyModel:
        hasPatches = True

        def __init__(self):
            self.patch_positions = {
                "old_low_ra": (Coordinate(10.0), Coordinate(1.0)),
                "old_high_ra": (Coordinate(20.0), Coordinate(2.0)),
            }
            self.patch_col = np.array(["old_low_ra", "old_high_ra"], dtype=object)
            self.updated_patch_positions = None

        def getPatchPositions(self):
            return self.patch_positions

        def getColValues(self, name):
            assert name == "Patch"
            return self.patch_col.copy()

        def getRowIndex(self, name):
            return {"old_low_ra": 0, "old_high_ra": 1}[name]

        def setColValues(self, name, values):
            assert name == "Patch"
            self.patch_col = values

        def setPatchPositions(self, patch_positions):
            self.updated_patch_positions = patch_positions

    skymodel = SkyModel()

    rename_skymodel_patches(skymodel, order_dec="low_to_high", order_ra="high_to_low")

    assert skymodel.patch_col.tolist() == ["Patch_2", "Patch_1"]
    assert set(skymodel.updated_patch_positions) == {"Patch_1", "Patch_2"}


def test_get_max_spectral_terms_raises_for_missing_skymodel_file():
    with pytest.raises(FileNotFoundError):
        get_max_spectral_terms("skymodel_file")


def test_nproc_uses_process_cpu_affinity(monkeypatch):
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0, 1, 2})

    assert nproc() == 3


def test_nproc_falls_back_to_multiprocessing_cpu_count(monkeypatch):
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(multiprocessing, "cpu_count", lambda: 4)

    assert nproc() == 4
