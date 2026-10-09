"""Tests for screen h5parm collection helpers."""

import h5py
import numpy as np
import pytest

from rapthor.execution.calibrate.screen_h5parms import collect_screen_h5parms


def _write_screen_h5parm(path, times, value_offset):
    with h5py.File(path, "w") as h5parm:
        solset = h5parm.create_group("sol000")
        solset.attrs["TITLE"] = "solution set"
        soltab = solset.create_group("phase000")
        soltab.attrs["TITLE"] = "phase"
        time = soltab.create_dataset("time", data=np.array(times, dtype=float))
        time.attrs["AXES"] = "time"
        values = np.arange(len(times), dtype=float).reshape(1, 1, len(times), 1) + value_offset
        val = soltab.create_dataset("val", data=values)
        val.attrs["AXES"] = "dir,ant,time,freq"
        weight = soltab.create_dataset("weight", data=np.ones_like(values) * value_offset)
        weight.attrs["AXES"] = "dir,ant,time,freq"


def _read_collected(path):
    with h5py.File(path, "r") as h5parm:
        soltab = h5parm["sol000/phase000"]
        return {
            "time": soltab["time"][:],
            "val": soltab["val"][:],
            "weight": soltab["weight"][:],
            "title": h5parm["sol000"].attrs["TITLE"],
            "version": h5parm["sol000"].attrs["h5parm_version"],
        }


def test_collect_screen_h5parms_concatenates_time_axis(tmp_path):
    first = tmp_path / "first.h5"
    second = tmp_path / "second.h5"
    output = tmp_path / "combined.h5"
    _write_screen_h5parm(first, [0.0, 10.0], 1.0)
    _write_screen_h5parm(second, [20.0], 5.0)

    collect_screen_h5parms([str(first), str(second)], str(output))

    collected = _read_collected(output)
    assert np.allclose(collected["time"], [0.0, 10.0, 20.0])
    assert collected["val"].shape == (1, 1, 3, 1)
    assert np.allclose(collected["val"][0, 0, :, 0], [1.0, 2.0, 5.0])
    assert np.allclose(collected["weight"][0, 0, :, 0], [1.0, 1.0, 5.0])
    assert collected["title"] == "solution set"
    assert collected["version"] == 1.0


def test_collect_screen_h5parms_rejects_existing_output_without_overwrite(tmp_path):
    first = tmp_path / "first.h5"
    output = tmp_path / "combined.h5"
    _write_screen_h5parm(first, [0.0], 1.0)
    output.write_text("existing")

    with pytest.raises(FileExistsError, match="overwrite=False"):
        collect_screen_h5parms([str(first)], str(output), overwrite=False)
