"""Tests for calibration solution plotting helpers."""

from rapthor.execution.calibrate.plotting import plot_solutions


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


class FakePlotRunner:
    def __init__(self):
        self.calls = []

    def run(self, *args, **kwargs):
        self.calls.append((args, kwargs))


def test_plot_solutions_first_dir_selects_first_available_direction():
    soltab = FakeSoltab()
    plot_runner = FakePlotRunner()

    plot_solutions(
        "solutions.h5parm",
        "phase",
        first_dir=True,
        h5parm_factory=lambda filename: FakeH5parm(soltab),
        plot_runner=plot_runner,
    )

    assert soltab.selected_dir == "[Patch_A]"
    assert plot_runner.calls
