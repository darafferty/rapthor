"""Focused tests for combine_h5parms mode contracts."""

from pathlib import Path

import pytest

import rapthor.scripts.combine_h5parms as combine_module


class _FakeH5:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def getSolset(self, solset=None):
        return object()

    def makeSolset(self, solsetName=None, addTables=False):
        return object()


@pytest.fixture
def temp_h5_files(tmp_path):
    h51 = tmp_path / "in1.h5"
    h52 = tmp_path / "in2.h5"
    h51.touch()
    h52.touch()
    return h51, h52, tmp_path / "out.h5"


def test_main_rejects_unknown_mode(temp_h5_files):
    h51, h52, out = temp_h5_files
    with pytest.raises(ValueError, match="Mode unknown_mode unknown"):
        combine_module.main(str(h51), str(h52), str(out), "unknown_mode")


@pytest.mark.parametrize(
    "mode, called_func",
    [
        ("p1a2", "combine_phase1_amp2"),
        ("p1p2_scalar", "combine_phase1_phase2_scalar"),
        ("p1a1a2", "combine_phase1_amp1_amp2"),
        ("p1p2a2", "combine_phase1_phase2_amp2"),
        ("p1p2a2_diagonal", "combine_phase1_phase2_amp2_diagonal"),
        ("p1p2a2_scalar", "combine_phase1_phase2_amp2_scalar"),
        ("separate", "copy_solset"),
    ],
)
def test_main_dispatches_supported_modes(monkeypatch, temp_h5_files, mode, called_func):
    """main() should dispatch to the mode-specific combine function."""
    h51, h52, out = temp_h5_files

    # Avoid real file/h5 operations.
    monkeypatch.setattr(combine_module.shutil, "copy", lambda src, dst: str(src))
    monkeypatch.setattr(combine_module.os.path, "exists", lambda path: False)
    monkeypatch.setattr(combine_module, "h5parm", lambda *args, **kwargs: _FakeH5())

    called = {"name": None}

    def _record(name):
        def _inner(*args, **kwargs):
            called["name"] = name
            # For combine_* funcs, return the output solset (3rd positional arg).
            # For copy_solset, return the destination solset (2nd positional arg).
            if len(args) >= 3:
                return args[2]
            if len(args) >= 2:
                return args[1]
            return object()

        return _inner

    monkeypatch.setattr(combine_module, "combine_phase1_amp2", _record("combine_phase1_amp2"))
    monkeypatch.setattr(
        combine_module,
        "combine_phase1_phase2_scalar",
        _record("combine_phase1_phase2_scalar"),
    )
    monkeypatch.setattr(
        combine_module,
        "combine_phase1_amp1_amp2",
        _record("combine_phase1_amp1_amp2"),
    )
    monkeypatch.setattr(
        combine_module,
        "combine_phase1_phase2_amp2",
        _record("combine_phase1_phase2_amp2"),
    )
    monkeypatch.setattr(
        combine_module,
        "combine_phase1_phase2_amp2_diagonal",
        _record("combine_phase1_phase2_amp2_diagonal"),
    )
    monkeypatch.setattr(
        combine_module,
        "combine_phase1_phase2_amp2_scalar",
        _record("combine_phase1_phase2_amp2_scalar"),
    )
    monkeypatch.setattr(combine_module, "copy_solset", _record("copy_solset"))

    combine_module.main(str(h51), str(h52), str(out), mode)

    assert called["name"] == called_func
