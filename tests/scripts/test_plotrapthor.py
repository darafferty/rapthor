import importlib.util
from importlib.machinery import SourceFileLoader
from pathlib import Path

import pytest


def load_plotrapthor():
    script_path = Path(__file__).resolve().parents[2] / "bin" / "plotrapthor"
    loader = SourceFileLoader("plotrapthor_bin", str(script_path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


class FakeH5parm:
    def __init__(self, soltab):
        self.soltab = soltab

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def getSolset(self, name):
        return FakeSolset(self.soltab)


class FakeSolset:
    def __init__(self, soltab):
        self.soltab = soltab

    def getSoltab(self, name):
        return self.soltab


class FakeSoltab:
    axesNames = []
    ant = ["CS001"]
    time = [0.0, 1.0]
    freq = [150e6]
    dir = ["[Patch_A]", "[Patch_B]"]

    def __init__(self):
        self.selected_dir = None

    def setSelection(self, dir=None):
        self.selected_dir = dir


def test_plotrapthor_first_dir_selects_first_available_direction(monkeypatch):
    try:
        plotrapthor = load_plotrapthor()
    except ImportError as exc:
        pytest.skip(f"Skipping plotrapthor test due to ImportError: {exc}")

    soltab = FakeSoltab()
    plot_calls = []
    monkeypatch.setattr(plotrapthor, "h5parm", lambda filename: FakeH5parm(soltab))
    monkeypatch.setattr(
        plotrapthor.plot,
        "run",
        lambda *args, **kwargs: plot_calls.append((args, kwargs)),
    )

    plotrapthor.main("solutions.h5parm", "phase", first_dir=True)

    assert soltab.selected_dir == "[Patch_A]"
    assert plot_calls
