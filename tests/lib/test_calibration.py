"""Tests for shared calibration strategy helpers."""

import pytest

from rapthor.lib.calibration import resolve_calibration_solves


@pytest.mark.parametrize(
    "mode, expected",
    [
        ("di", ["medium_phase", "fast_phase"]),
        ("dd", ["slow_gains"]),
    ],
)
def test_resolve_calibration_solves_preserves_explicit_order(mode, expected):
    strategy = {
        "di": ["medium_phase", "fast_phase"],
        "dd": ["slow_gains"],
    }

    solves, defaulted = resolve_calibration_solves(
        mode,
        calibration_strategy=strategy,
        do_slowgain_solve=False,
        do_fulljones_solve=False,
    )

    assert solves == expected
    assert defaulted is False


def test_resolve_calibration_solves_returns_no_solves_for_missing_explicit_mode():
    solves, defaulted = resolve_calibration_solves(
        "di",
        calibration_strategy={"dd": ["fast_phase"]},
        do_slowgain_solve=True,
        do_fulljones_solve=True,
    )

    assert solves == []
    assert defaulted is False


@pytest.mark.parametrize(
    "mode, slowgain, fulljones, expected",
    [
        ("dd", False, False, ["fast_phase", "medium_phase"]),
        ("dd", True, False, ["fast_phase", "medium_phase", "slow_gains"]),
        ("di", False, False, []),
        ("di", False, True, ["full_jones"]),
    ],
)
def test_resolve_calibration_solves_expands_legacy_flags(mode, slowgain, fulljones, expected):
    solves, defaulted = resolve_calibration_solves(
        mode,
        calibration_strategy=None,
        do_slowgain_solve=slowgain,
        do_fulljones_solve=fulljones,
    )

    assert solves == expected
    assert defaulted is True
