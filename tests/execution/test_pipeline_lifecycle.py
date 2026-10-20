"""Tests for the scheduler-independent pipeline lifecycle helpers."""

from types import SimpleNamespace

import pytest

from rapthor.execution.pipeline.lifecycle import do_final_pass
from rapthor.lib.strategy import set_selfcal_strategy


def _field(parset, cycle_number):
    return SimpleNamespace(
        parset=parset,
        cycle_number=cycle_number,
        do_check=False,
        selfcal_state=SimpleNamespace(diverged=False, failed=False),
        make_quv_images=False,
    )


@pytest.mark.parametrize("generate_initial_skymodel", [True, False])
def test_do_final_pass_runs_the_narrower_final_cycle_of_the_selfcal_strategy(
    field, generate_initial_skymodel
):
    """With equal data fractions the final 4 MHz cycle still differs from selfcal."""
    field.parset["strategy"] = "selfcal"
    field.parset["generate_initial_skymodel"] = generate_initial_skymodel
    field.parset["selfcal_data_fraction"] = 1.0
    field.parset["final_data_fraction"] = 1.0
    strategy_steps = set_selfcal_strategy(field)
    selfcal_steps, final_step = strategy_steps[:-1], strategy_steps[-1]

    assert do_final_pass(_field(field.parset, len(selfcal_steps)), selfcal_steps, final_step)


def test_do_final_pass_skips_a_final_step_identical_to_the_last_selfcal_step():
    """A user strategy whose final step repeats the last selfcal step needs no final pass."""
    parset = {"selfcal_data_fraction": 1.0, "final_data_fraction": 1.0}
    last_step = {"do_calibrate": True, "channel_width_hz": 4e6}
    selfcal_steps = [{"do_calibrate": True, "channel_width_hz": 8e6}, last_step]

    assert not do_final_pass(_field(parset, 2), selfcal_steps, dict(last_step))


def test_do_final_pass_runs_for_a_different_final_data_fraction():
    parset = {"selfcal_data_fraction": 0.2, "final_data_fraction": 1.0}
    last_step = {"do_calibrate": True, "channel_width_hz": 4e6}

    assert do_final_pass(_field(parset, 1), [last_step], dict(last_step))
