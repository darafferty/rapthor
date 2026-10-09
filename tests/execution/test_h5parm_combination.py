"""Tests for h5parm combination execution helpers."""

import numpy as np
import pytest

from rapthor.execution.calibrate.h5parm_combination import (
    average_polarizations,
    combine_h5parms,
    combine_phase1_amp1_amp2,
    combine_phase1_amp2,
    combine_phase1_phase2_amp2,
    combine_phase1_phase2_amp2_diagonal,
    combine_phase1_phase2_amp2_scalar,
    combine_phase1_phase2_scalar,
    copy_solset,
    expand_array,
    interpolate_solutions,
)


class FakeSolsetObject:
    def __init__(self, solset):
        self.solset = solset

    def _f_copy_children(self, output_obj, recursive=False, overwrite=False):
        for name, soltab in self.solset.soltabs.items():
            output_obj.solset.soltabs[name] = soltab.clone(parent=output_obj.solset)


class FakeSolset:
    def __init__(self, soltabs=()):
        self.soltabs = {}
        self.obj = FakeSolsetObject(self)
        for soltab in soltabs:
            self.add(soltab)

    def add(self, soltab):
        soltab.parent = self
        self.soltabs[soltab.name] = soltab

    def getSoltabNames(self):
        return list(self.soltabs)

    def getSoltab(self, name):
        return self.soltabs[name]

    def makeSoltab(self, soltype, soltabName, axesNames, axesVals, vals, weights):
        soltab = FakeSoltab(soltabName, soltype, axesNames, axesVals, vals, weights)
        self.add(soltab)
        return soltab


class FakeSoltab:
    def __init__(
        self,
        name,
        soltype,
        axes_names,
        axes_vals,
        vals,
        weights=None,
        parent=None,
    ):
        self.name = name
        self.soltype = soltype
        self.axes_names = list(axes_names)
        self.axes_vals = {
            axis: np.array(values) for axis, values in zip(self.axes_names, axes_vals)
        }
        self.val = np.array(vals, dtype=float)
        self.weight = (
            np.ones_like(self.val, dtype=float)
            if weights is None
            else np.array(weights, dtype=float)
        )
        self.parent = parent
        for axis, values in self.axes_vals.items():
            setattr(self, axis, values)

    def clone(self, parent=None):
        axes_vals = [self.axes_vals[axis].copy() for axis in self.axes_names]
        return FakeSoltab(
            self.name,
            self.soltype,
            self.axes_names,
            axes_vals,
            self.val.copy(),
            self.weight.copy(),
            parent=parent,
        )

    def delete(self):
        if self.parent is not None:
            self.parent.soltabs.pop(self.name, None)

    def getAxesNames(self):
        return list(self.axes_names)

    def getAxisValues(self, axis):
        return self.axes_vals[axis]

    def getType(self):
        return self.soltype


def make_soltab(name, soltype, axes_names, axes_vals, vals, weights=None):
    return FakeSoltab(name, soltype, axes_names, axes_vals, vals, weights)


def assert_soltab_values(solset, name, expected_values, expected_weights=None):
    soltab = solset.getSoltab(name)
    assert np.allclose(soltab.val, expected_values)
    if expected_weights is not None:
        assert np.allclose(soltab.weight, expected_weights)


def test_expand_array():
    array = np.array([[1, 2], [3, 4]])
    new_shape = (2, 2, 2)
    new_axis_ind = 1
    expected_shape = (2, 2, 2)
    expanded_array = expand_array(array, new_shape, new_axis_ind)
    assert expanded_array.shape == expected_shape, (
        f"Expected shape {expected_shape}, got {expanded_array.shape}"
    )
    # Check if the values are correctly expanded
    expected_values = np.array([[[1, 2], [1, 2]], [[3, 4], [3, 4]]])
    assert np.array_equal(expanded_array, expected_values), (
        f"Expected values {expected_values}, got {expanded_array}"
    )


def test_expand_array_with_new_trailing_axis():
    array = np.array([[1, 2], [3, 4]])
    expanded_array = expand_array(array, (2, 2, 2), 2)
    expected_values = np.array([[[1, 1], [2, 2]], [[3, 3], [4, 4]]])
    assert np.array_equal(expanded_array, expected_values)


def test_expand_array_with_existing_singleton_axis():
    array = np.array([[[1, 2]], [[3, 4]]])
    expanded_array = expand_array(array, (2, 3, 2), 1)
    expected_values = np.array([[[1, 2], [1, 2], [1, 2]], [[3, 4], [3, 4], [3, 4]]])
    assert np.array_equal(expanded_array, expected_values)


def test_average_polarizations_averages_amplitudes_in_log_space():
    soltab = make_soltab(
        "amplitude000",
        "amplitude",
        ["time", "pol"],
        [[0.0], ["XX", "YY"]],
        [[2.0, 8.0]],
    )

    vals, weights = average_polarizations(soltab)

    assert vals == pytest.approx([4.0])
    assert weights == pytest.approx([1.0])


def test_average_polarizations_uses_circular_mean_for_phases():
    soltab = make_soltab(
        "phase000",
        "phase",
        ["time", "pol"],
        [[0.0], ["XX", "YY"]],
        [[0.0, np.pi / 2.0]],
    )

    vals, weights = average_polarizations(soltab)

    assert vals == pytest.approx([np.pi / 4.0])
    assert weights == pytest.approx([1.0])


def test_interpolate_solutions_expands_single_slow_solution():
    fast_soltab = make_soltab(
        "fast",
        "phase",
        ["time", "freq"],
        [[0.0, 10.0, 20.0], [100.0, 200.0]],
        np.zeros((3, 2)),
    )
    slow_soltab = make_soltab(
        "slow",
        "amplitude",
        ["time", "freq"],
        [[0.0], [150.0]],
        [[2.0]],
    )

    vals, weights = interpolate_solutions(fast_soltab, slow_soltab, [3, 2])

    assert np.allclose(vals, 2.0)
    assert np.allclose(weights, 1.0)


def test_interpolate_solutions_preserves_flags_as_nans():
    fast_soltab = make_soltab(
        "fast",
        "phase",
        ["time", "freq"],
        [[0.0, 10.0], [100.0]],
        np.zeros((2, 1)),
    )
    slow_soltab = make_soltab(
        "slow",
        "amplitude",
        ["time", "freq"],
        [[0.0], [100.0]],
        [[9.0]],
        weights=[[0.0]],
    )

    vals, weights = interpolate_solutions(fast_soltab, slow_soltab, [2, 1])

    assert np.isnan(vals).all()
    assert np.allclose(weights, 0.0)


def test_combine_phase1_amp2_keeps_phase1_and_amp2_only():
    ss1 = FakeSolset(
        [
            make_soltab("phase000", "phase", ["time"], [[0.0]], [1.0]),
            make_soltab("amplitude000", "amplitude", ["time"], [[0.0]], [2.0]),
        ]
    )
    ss2 = FakeSolset(
        [
            make_soltab("phase000", "phase", ["time"], [[0.0]], [3.0]),
            make_soltab("amplitude000", "amplitude", ["time"], [[0.0]], [4.0]),
        ]
    )
    output = FakeSolset()

    combine_phase1_amp2(ss1, ss2, output)

    assert set(output.getSoltabNames()) == {"phase000", "amplitude000"}
    assert_soltab_values(output, "phase000", [1.0])
    assert_soltab_values(output, "amplitude000", [4.0])
    assert "amplitude000" not in ss1.getSoltabNames()
    assert "phase000" not in ss2.getSoltabNames()


def test_combine_phase1_amp1_amp2_multiplies_amplitudes_and_expands_phases():
    ss1 = FakeSolset(
        [
            make_soltab("phase000", "phase", ["time", "freq"], [[0.0], [100.0]], [[0.2]]),
            make_soltab(
                "amplitude000",
                "amplitude",
                ["time", "freq"],
                [[0.0], [100.0]],
                [[2.0]],
            ),
        ]
    )
    ss2 = FakeSolset(
        [
            make_soltab(
                "phase000",
                "phase",
                ["time", "freq"],
                [[0.0, 10.0], [100.0]],
                [[0.0], [0.0]],
            ),
            make_soltab(
                "amplitude000",
                "amplitude",
                ["time", "freq"],
                [[0.0, 10.0], [100.0]],
                [[3.0], [4.0]],
            ),
        ]
    )
    output = FakeSolset()

    combine_phase1_amp1_amp2(ss1, ss2, output)

    assert_soltab_values(output, "amplitude000", [[6.0], [8.0]])
    assert_soltab_values(output, "phase000", [[0.2], [0.2]])


def test_combine_phase1_phase2_scalar_sums_interpolated_phases():
    ss1 = FakeSolset(
        [
            make_soltab(
                "phase000",
                "phase",
                ["time", "freq"],
                [[0.0, 10.0], [100.0]],
                [[1.0], [2.0]],
            )
        ]
    )
    ss2 = FakeSolset(
        [make_soltab("phase000", "phase", ["time", "freq"], [[0.0], [100.0]], [[0.5]])]
    )
    output = FakeSolset()

    combine_phase1_phase2_scalar(ss1, ss2, output)

    assert_soltab_values(output, "phase000", [[1.5], [2.5]])


def test_combine_phase1_phase2_amp2_sums_averaged_phases_and_copies_amplitudes():
    ss1 = FakeSolset(
        [
            make_soltab(
                "phase000",
                "phase",
                ["time", "freq"],
                [[0.0, 10.0], [100.0]],
                [[1.0], [2.0]],
            )
        ]
    )
    ss2 = FakeSolset(
        [
            make_soltab(
                "phase000",
                "phase",
                ["time", "freq", "pol"],
                [[0.0], [100.0], ["XX", "YY"]],
                [[[0.2, 0.6]]],
            ),
            make_soltab(
                "amplitude000",
                "amplitude",
                ["time", "freq", "pol"],
                [[0.0], [100.0], ["XX", "YY"]],
                [[[3.0, 5.0]]],
            ),
        ]
    )
    output = FakeSolset()

    combine_phase1_phase2_amp2(ss1, ss2, output)

    assert_soltab_values(output, "phase000", [[1.4], [2.4]])
    assert_soltab_values(output, "amplitude000", [[[3.0, 5.0]]])
    assert "phase000" not in ss2.getSoltabNames()


def test_combine_phase1_phase2_amp2_diagonal_preserves_polarizations():
    ss1 = FakeSolset(
        [
            make_soltab(
                "phase000",
                "phase",
                ["time", "freq"],
                [[0.0, 10.0], [100.0]],
                [[1.0], [2.0]],
            )
        ]
    )
    ss2 = FakeSolset(
        [
            make_soltab(
                "phase000",
                "phase",
                ["time", "freq", "pol"],
                [[0.0], [100.0], ["XX", "YY"]],
                [[[0.1, 0.2]]],
            ),
            make_soltab(
                "amplitude000",
                "amplitude",
                ["time", "freq", "pol"],
                [[0.0], [100.0], ["XX", "YY"]],
                [[[3.0, 5.0]]],
            ),
        ]
    )
    output = FakeSolset()

    combine_phase1_phase2_amp2_diagonal(ss1, ss2, output)

    expected = np.array([[[1.1, 1.2]], [[2.1, 2.2]]])
    assert_soltab_values(output, "phase000", expected)
    assert output.getSoltab("phase000").getAxesNames() == ["time", "freq", "pol"]
    assert_soltab_values(output, "amplitude000", [[[3.0, 5.0]]])


def test_combine_phase1_phase2_amp2_scalar_averages_phase_and_amplitude_polarizations():
    ss1 = FakeSolset(
        [
            make_soltab(
                "phase000",
                "phase",
                ["time", "freq"],
                [[0.0, 10.0], [100.0]],
                [[1.0], [2.0]],
            )
        ]
    )
    ss2 = FakeSolset(
        [
            make_soltab(
                "phase000",
                "phase",
                ["time", "freq", "pol"],
                [[0.0], [100.0], ["XX", "YY"]],
                [[[0.2, 0.6]]],
            ),
            make_soltab(
                "amplitude000",
                "amplitude",
                ["time", "freq", "pol"],
                [[0.0], [100.0], ["XX", "YY"]],
                [[[2.0, 8.0]]],
            ),
        ]
    )
    output = FakeSolset()

    combine_phase1_phase2_amp2_scalar(ss1, ss2, output)

    assert_soltab_values(output, "phase000", [[1.4], [2.4]])
    assert_soltab_values(output, "amplitude000", [[4.0]])
    assert output.getSoltab("amplitude000").getAxesNames() == ["time", "freq"]


def test_copy_solset_copies_all_soltabs():
    source = FakeSolset(
        [
            make_soltab("phase000", "phase", ["time"], [[0.0]], [1.0]),
            make_soltab("amplitude000", "amplitude", ["time"], [[0.0]], [2.0]),
        ]
    )
    output = FakeSolset()

    copy_solset(source, output)

    assert set(output.getSoltabNames()) == {"phase000", "amplitude000"}
    assert_soltab_values(output, "phase000", [1.0])
    assert_soltab_values(output, "amplitude000", [2.0])
    assert output.getSoltab("phase000") is not source.getSoltab("phase000")


def test_main_rejects_unknown_mode_before_opening_files():
    with pytest.raises(ValueError, match="Mode unknown"):
        combine_h5parms("one.h5", "two.h5", "out.h5", "unknown")
