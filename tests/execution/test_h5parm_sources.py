"""Tests for h5parm source-coordinate execution helpers."""

import numpy as np
import pytest

import rapthor.execution.calibrate.h5parm_sources as h5parm_sources
from rapthor.execution.calibrate.h5parm_sources import adjust_h5parm_source_coordinates


class FakeAngle:
    def __init__(self, value):
        self.value = value


class FakeSkyModel:
    def __init__(self, positions):
        self.positions = positions

    def getPatchPositions(self):
        return self.positions


class FakeSourceTable:
    def __init__(self):
        self.removed = False
        self.remove_kwargs = None
        self.rows = None

    def _f_remove(self, recursive=False):
        self.removed = True
        self.remove_kwargs = {"recursive": recursive}

    def append(self, rows):
        self.rows = rows


class FakeSolsetObject:
    def __init__(self, source_table):
        self.source_table = source_table

    def _f_get_child(self, name):
        assert name == "source"
        return self.source_table


class FakeH5Handle:
    def __init__(self):
        self.get_node_calls = []
        self.create_table_calls = []

    def get_node(self, root, solset_name):
        self.get_node_calls.append((root, solset_name))
        return f"{root}/{solset_name}"

    def create_table(self, snode, name, descriptor, title="", expectedrows=None):
        self.create_table_calls.append(
            {
                "snode": snode,
                "name": name,
                "descriptor": descriptor,
                "title": title,
                "expectedrows": expectedrows,
            }
        )
        return object()


class FakeSoltab:
    def __init__(
        self,
        name="phase000",
        soltype="phase",
        axes_names=None,
        axes_vals=None,
        vals=None,
        weights=None,
        dirs=None,
    ):
        self.name = name
        self.soltype = soltype
        self.axes_names = list(axes_names or ["time"])
        self.axes_vals = {
            axis: np.array(values) for axis, values in zip(self.axes_names, axes_vals or [[0.0]])
        }
        self.vals = np.array(vals if vals is not None else [0.0], dtype=float)
        self.weights = (
            np.ones_like(self.vals, dtype=float)
            if weights is None
            else np.array(weights, dtype=float)
        )
        self.deleted = False
        if dirs is not None:
            self.dir = list(dirs)

    def getValues(self, weight=False):
        if weight:
            return self.weights.copy(), self.axes_vals
        return self.vals.copy(), self.axes_vals

    def getType(self):
        return self.soltype

    def getAxesNames(self):
        return list(self.axes_names)

    def getAxisValues(self, axis_name):
        return self.axes_vals[axis_name]

    def delete(self):
        self.deleted = True


class FakeSolset:
    def __init__(self, soltabs, source_table=None):
        self.soltabs = list(soltabs)
        self.source_table = source_table or FakeSourceTable()
        self.obj = FakeSolsetObject(self.source_table)
        self.created_soltabs = []

    def getSoltabs(self):
        return self.soltabs

    def makeSoltab(self, soltype, soltabName, axesNames, axesVals, vals, weights):
        created = {
            "soltype": soltype,
            "soltabName": soltabName,
            "axesNames": axesNames,
            "axesVals": axesVals,
            "vals": vals,
            "weights": weights,
        }
        self.created_soltabs.append(created)
        return created


class FakeH5Parm:
    def __init__(self, solset):
        self.solset = solset
        self.H = FakeH5Handle()
        self.open_calls = []

    def __call__(self, filename, readonly=False):
        self.open_calls.append((filename, readonly))
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def getSolset(self, solset_name):
        self.requested_solset_name = solset_name
        return self.solset


def patch_script(monkeypatch, source_positions, solset):
    fake_h5parm = FakeH5Parm(solset)
    monkeypatch.setattr(
        h5parm_sources.lsmtool,
        "load",
        lambda skymodel: FakeSkyModel(source_positions),
    )
    monkeypatch.setattr(h5parm_sources, "h5parm", fake_h5parm)
    monkeypatch.setattr(
        h5parm_sources,
        "normalize_ra_dec",
        lambda ra, dec: (ra, dec),
    )
    return fake_h5parm


def patch_position(ra_deg, dec_deg):
    return (FakeAngle(ra_deg), FakeAngle(dec_deg))


def test_main_rewrites_source_table_for_matching_directions(monkeypatch):
    source_table = FakeSourceTable()
    solset = FakeSolset(
        [
            FakeSoltab(
                dirs=["[Patch_A]", "[Patch_B]"],
                axes_names=["time", "dir"],
                axes_vals=[[0.0], ["[Patch_A]", "[Patch_B]"]],
                vals=[[1.0, 2.0]],
            )
        ],
        source_table=source_table,
    )
    source_positions = {
        "Patch_A": patch_position(10.0, -20.0),
        "Patch_B": patch_position(30.0, 40.0),
    }
    fake_h5parm = patch_script(monkeypatch, source_positions, solset)

    adjust_h5parm_source_coordinates("calibrators.sky", "solutions.h5", solset_name="sol001")

    assert fake_h5parm.open_calls == [("solutions.h5", False)]
    assert fake_h5parm.requested_solset_name == "sol001"
    assert source_table.removed is True
    assert source_table.remove_kwargs == {"recursive": True}
    assert fake_h5parm.H.get_node_calls == [("/", "sol001")]
    create_table_call = fake_h5parm.H.create_table_calls[0]
    assert create_table_call["name"] == "source"
    assert create_table_call["descriptor"].names == ("name", "dir")
    assert create_table_call["expectedrows"] == 25
    names, coords = zip(*source_table.rows)
    assert names == ("[Patch_A]", "[Patch_B]")
    assert np.allclose(
        coords,
        [
            [np.deg2rad(10.0), np.deg2rad(-20.0)],
            [np.deg2rad(30.0), np.deg2rad(40.0)],
        ],
    )


def test_main_rejects_direction_count_mismatch(monkeypatch):
    solset = FakeSolset([FakeSoltab(dirs=["[Patch_A]"])])
    source_positions = {
        "Patch_A": patch_position(10.0, -20.0),
        "Patch_B": patch_position(30.0, 40.0),
    }
    patch_script(monkeypatch, source_positions, solset)

    with pytest.raises(ValueError, match="must have the same length"):
        adjust_h5parm_source_coordinates("calibrators.sky", "solutions.h5")


def test_main_rejects_unknown_h5parm_direction(monkeypatch):
    solset = FakeSolset([FakeSoltab(dirs=["[Patch_A]", "[Missing]"])])
    source_positions = {
        "Patch_A": patch_position(10.0, -20.0),
        "Patch_B": patch_position(30.0, 40.0),
    }
    patch_script(monkeypatch, source_positions, solset)

    with pytest.raises(ValueError, match="not in the sky model"):
        adjust_h5parm_source_coordinates("calibrators.sky", "solutions.h5")


def test_main_duplicates_direction_independent_solutions(monkeypatch, capsys):
    soltab = FakeSoltab(
        axes_names=["time"],
        axes_vals=[[0.0, 10.0]],
        vals=[1.0, 2.0],
        weights=[0.5, 0.75],
        dirs=None,
    )
    solset = FakeSolset([soltab])
    source_positions = {
        "Patch_A": patch_position(10.0, -20.0),
        "Patch_B": patch_position(30.0, 40.0),
    }
    patch_script(monkeypatch, source_positions, solset)

    adjust_h5parm_source_coordinates("calibrators.sky", "solutions.h5")

    captured = capsys.readouterr()
    assert "duplicated for all directions" in captured.out
    assert soltab.deleted is True
    assert len(solset.created_soltabs) == 1
    created = solset.created_soltabs[0]
    assert created["soltype"] == "phase"
    assert created["soltabName"] == "phase000"
    assert created["axesNames"] == ["time", "dir"]
    assert created["axesVals"][1] == ["[Patch_A]", "[Patch_B]"]
    assert np.allclose(created["vals"], [[1.0, 1.0], [2.0, 2.0]])
    assert np.allclose(created["weights"], [[0.5, 0.5], [0.75, 0.75]])
