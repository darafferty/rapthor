"""Integration tests for legacy calibration strategy handling across cycles."""

import pytest

from rapthor.lib.strategy import set_selfcal_strategy
from rapthor.operations.calibrate import Calibrate


@pytest.mark.integration
def test_legacy_selfcal_strategy_stays_defaulted_across_cycles(field, monkeypatch):
    """Legacy selfcal steps must be defaulted independently for each cycle."""
    field.parset["generate_initial_skymodel"] = True
    field.parset["strategy"] = "selfcal"
    monkeypatch.setattr(field, "update_skymodels", lambda *args, **kwargs: None)
    monkeypatch.setattr(field, "remove_skymodels", lambda: None)
    field.outlier_sectors = []
    field.bright_source_sectors = []

    strategy_steps = set_selfcal_strategy(field)
    assert "calibration_strategy" not in strategy_steps[0]
    assert "calibration_strategy" not in strategy_steps[1]

    expected_default_strategy = {
        "dd": ["fast_phase", "medium_phase", "slow_gains"],
        "di": [],
    }
    expected_dd_solves = ["fast_phase", "medium_phase", "slow_gains", "medium_phase"]

    for cycle_number, step in enumerate(strategy_steps[:2], start=1):
        field.update(step, cycle_number, final=False)

        assert field._calibration_strategy_defaulted is True
        assert field.calibration_strategy == expected_default_strategy

        solve_plan = Calibrate("dd", field, cycle_number)._build_solve_plan()
        assert [solve.solve_type for solve in solve_plan] == expected_dd_solves
