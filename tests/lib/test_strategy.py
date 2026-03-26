"""
Test cases for the `rapthor.lib.strategy` module.
"""

from pathlib import Path

import pytest
from rapthor.lib.strategy import (
    set_image_strategy,
    set_selfcal_strategy,
    set_strategy,
    set_user_strategy,
    check_and_adjust_parameters,
)


@pytest.fixture
def strategy_path():
    """
    Fixture to provide the path to the default imaging strategy file.
    """
    return Path(__file__).parent.parent.parent / "examples" / "default_imaging_strategy.py"


@pytest.mark.parametrize("strategy", ["selfcal", "image", "custom_strategy", "unrecognised"])
def test_set_strategy(field, request, strategy):
    """Test the that different strategy inputs are handled correctly."""
    # User-defined path to a custom strategy
    if strategy == "custom_strategy":
        strategy = request.getfixturevalue(strategy)
    field.parset["strategy"] = str(strategy)

    if strategy == "unrecognised":
        # If the strategy is unrecognised, set_strategy should raise a ValueError
        with pytest.raises(ValueError):
            set_strategy(field)
    else:
        strategy_steps = set_strategy(field)
        assert isinstance(strategy_steps, list)
        assert all(isinstance(step, dict) for step in strategy_steps)
        assert len(strategy_steps) > 0
        assert "do_calibrate" in strategy_steps[0]
        assert "do_image" in strategy_steps[0]
        assert "do_normalize" in strategy_steps[0]


@pytest.mark.parametrize("generate_initial_skymodel", [True, False])
def test_set_selfcal_strategy(field, generate_initial_skymodel):
    field.parset["strategy"] = "selfcal"
    field.parset["generate_initial_skymodel"] = generate_initial_skymodel
    strategy_steps = set_selfcal_strategy(field)
    assert isinstance(strategy_steps, list)
    assert all(isinstance(step, dict) for step in strategy_steps)
    # regroup_model, do_calibrate, and do_image should be True for all steps
    assert all(step["regroup_model"] for step in strategy_steps)
    assert all(step["do_calibrate"] for step in strategy_steps)
    assert all(step["do_image"] for step in strategy_steps)
    if generate_initial_skymodel:
        assert len(strategy_steps) == 6 + 1  # 6 selfcal steps + 1 final step
    else:
        assert len(strategy_steps) == 8 + 1  # 8 selfcal steps + 1 final step
        # Phase-only solves for the first two cycles when no initial sky model is generated
        assert not strategy_steps[0]["do_slowgain_solve"]
        assert not strategy_steps[1]["do_slowgain_solve"]
        assert all(step["do_slowgain_solve"] for step in strategy_steps[2:])


def test_set_image_strategy(field):
    field.parset["strategy"] = "image"
    strategy_steps = set_image_strategy(field)
    assert isinstance(strategy_steps, list)
    assert all(isinstance(step, dict) for step in strategy_steps)
    assert len(strategy_steps) == 1
    assert not strategy_steps[0]["do_calibrate"]
    assert not strategy_steps[0]["do_normalize"]
    assert strategy_steps[0]["peel_outliers"]
    assert not strategy_steps[0]["peel_bright_sources"]
    assert not strategy_steps[0]["regroup_model"]


def test_set_user_strategy(field, custom_strategy):
    field.parset["strategy"] = str(custom_strategy)
    strategy_steps = set_user_strategy(field)
    assert isinstance(strategy_steps, list)
    assert all(isinstance(step, dict) for step in strategy_steps)
    assert len(strategy_steps) == 3
    assert all(step["do_calibrate"] for step in strategy_steps)
    assert all(step["do_slowgain_solve"] for step in strategy_steps)
    assert all(step["fast_timestep_sec"] == 32.0 for step in strategy_steps)
    assert all(step["medium_timestep_sec"] == 120.0 for step in strategy_steps)
    assert all(step["slow_timestep_sec"] == 600.0 for step in strategy_steps)
    assert all(step["fulljones_timestep_sec"] == 600.0 for step in strategy_steps)


def test_check_and_adjust_parameters_corrects_deprecated(field, custom_strategy, caplog):
    field.parset["strategy"] = str(custom_strategy)
    strategy_steps = set_user_strategy(field)

    # Add deprecated parameters to the strategy steps to test that they are correctly adjusted and raise warnings
    for step in strategy_steps:
        step["slow_timestep_joint_sec"] = 3.0
        step["slow_timestep_separate_sec"] = 0.5

    with caplog.at_level("WARNING"):
        strategy_steps_updated = check_and_adjust_parameters(field, strategy_steps)

    for i, step in enumerate(strategy_steps_updated):
        # slow_timestep_joint_sec should be removed and raise a warning
        assert "slow_timestep_joint_sec" not in step
        assert (
            f"Parameter 'slow_timestep_joint_sec' is defined in the strategy for cycle {i + 1} but is no longer used."
            in caplog.text
        )

        # slow_timestep_separate_sec should be replaced by slow_timestep_sec and raise a warning
        assert "slow_timestep_separate_sec" not in step
        assert "slow_timestep_sec" in step
        assert (
            f"Parameter 'slow_timestep_separate_sec' is defined in the strategy for cycle {i + 1} but is deprecated. Please use 'slow_timestep_sec' instead."
            in caplog.text
        )


@pytest.mark.parametrize(
    "missing_parameter", ["do_calibrate", "do_image", "do_normalize", "do_check"]
)
def test_check_and_adjust_parameters_raises_error_for_missing_primary_parameters(
    field, custom_strategy, missing_parameter
):
    field.parset["strategy"] = str(custom_strategy)
    strategy_steps = set_user_strategy(field)

    # Remove a required primary parameter from one of the steps
    del strategy_steps[0][missing_parameter]

    with pytest.raises(
        ValueError,
        match=f'Required parameter "{missing_parameter}" is not defined in the strategy for cycle 1.',
    ):
        check_and_adjust_parameters(field, strategy_steps)


@pytest.mark.parametrize(
    ("primary_parameter", "missing_parameter"),
    [
        ("do_calibrate", "do_slowgain_solve"),
        ("do_calibrate", "max_normalization_delta"),
        ("do_calibrate", "solve_min_uv_lambda"),
        ("do_calibrate", "fast_timestep_sec"),
        ("do_calibrate", "slow_timestep_sec"),
        ("do_calibrate", "fulljones_timestep_sec"),
        ("do_calibrate", "scale_normalization_delta"),
        ("do_check", "convergence_ratio"),
        ("do_check", "divergence_ratio"),
        ("do_check", "failure_ratio"),
    ],
)
def test_check_and_adjust_parameters_warns_for_missing_parameters_with_defaults(
    field, custom_strategy, primary_parameter, missing_parameter, caplog
):
    field.parset["strategy"] = str(custom_strategy)
    strategy_steps = set_user_strategy(field)

    # Set primary parameters to True
    strategy_steps[0][primary_parameter] = True
    if missing_parameter in strategy_steps[0]:
        # Remove the parameter to test that the default value is used and a warning is raised
        del strategy_steps[0][missing_parameter]

    with caplog.at_level("WARNING"):
        check_and_adjust_parameters(field, strategy_steps)

    assert (
        f"Parameter {missing_parameter!r} is not defined in the strategy "
        f"for cycle 1. Using the default value of {getattr(field, missing_parameter)!r}."
        in caplog.text
    )


@pytest.mark.parametrize(
    ("primary_parameter", "missing_parameter"),
    [
        ("do_calibrate", "do_fulljones_solve"),
        ("do_calibrate", "target_flux"),
        ("do_calibrate", "max_directions"),
        ("do_calibrate", "regroup_model"),
        ("do_image", "auto_mask"),
        ("do_image", "auto_mask_nmiter"),
        ("do_image", "channel_width_hz"),
        ("do_image", "threshisl"),
        ("do_image", "threshpix"),
        ("do_image", "max_nmiter"),
    ],
)
def test_check_and_adjust_parameters_warns_for_missing_parameters_without_defaults(
    field,
    custom_strategy,
    primary_parameter,
    missing_parameter,
):
    field.parset["strategy"] = str(custom_strategy)
    strategy_steps = set_user_strategy(field)

    strategy_steps[0][primary_parameter] = True
    if missing_parameter in strategy_steps[0]:
        # Remove the parameter to test that the default value is used and a warning is raised
        del strategy_steps[0][missing_parameter]

    with pytest.raises(
        ValueError,
        match=f'Required parameter "{missing_parameter}" is not defined in the strategy for cycle 1.',
    ):
        check_and_adjust_parameters(field, strategy_steps)
