from pathlib import Path

import pytest

from rapthor.lib.field import Field


@pytest.fixture
def field(parset_for_field_test):

    field = Field(parset_for_field_test)
    field.fast_timestep_sec = 32.0
    field.update_skymodels(1, True, target_flux=0.2)
    field.set_obs_parameters()
    field.define_imaging_sectors()
    field.define_outlier_sectors(1)
    yield field


def test_scan_observations(field):
    assert field.fwhm_ra_deg == 4.500843683229519


def test_regular_frequency_spacing(field):
    assert all(obs.channels_are_regular for obs in field.observations)


def test_imaging_sectors(field):
    assert field.sector_bounds_deg == "[258.558431;57.961675;259.103519;56.885818]"


def test_outlier_sectors(field):
    assert field.outlier_sectors == []


def test_chunk_observations(field):
    for obs in field.full_observations:
        obs.data_fraction = 0.8
    field.chunk_observations(600.0, prefer_high_el_periods=False)
    full_obs = field.full_observations[0]
    obs = field.imaging_sectors[0].observations[0]
    chunked_starttime = full_obs.starttime
    chunked_endtime = full_obs.endtime - full_obs.timepersample
    assert obs.starttime == chunked_starttime
    assert obs.endtime == chunked_endtime


def test_chunk_observations_high_el(field):
    for obs in field.full_observations:
        obs.data_fraction = 0.2
    field.chunk_observations(600.0, prefer_high_el_periods=True)
    full_obs = field.full_observations[0]
    obs = field.imaging_sectors[0].observations[0]
    chunked_starttime = full_obs.starttime + 2 * full_obs.timepersample
    chunked_endtime = full_obs.endtime - 3 * full_obs.timepersample
    assert obs.starttime == chunked_starttime
    assert obs.endtime == chunked_endtime


def test_get_obs_parameters(field):
    obsp = field.get_obs_parameters("starttime")
    assert obsp == ["29Mar2013/13:59:52.907"]


def test_define_imaging_sectors(field):
    field.define_imaging_sectors()
    assert field.sector_bounds_mid_deg == "[258.841667;57.410833]"


def test_define_outlier_sectors(field):
    field.define_outlier_sectors(1)
    assert field.outlier_sectors == []


def test_define_bright_source_sectors(field):
    field.define_bright_source_sectors(0)
    assert field.bright_source_sectors == []


def test_find_intersecting_sources(field):
    iss = field.find_intersecting_sources()
    assert iss[0].area == pytest.approx(18.37996802132365)


def test_check_selfcal_progress(field):
    assert field.check_selfcal_progress() == (False, False, False)


def test_plot_overview_patches(field):
    plot_filename = "field_overview_1.png"
    plot_path = Path(field.parset["dir_working"]) / "plots" / plot_filename

    assert plot_path.exists()

    plot_path.unlink()  # Remove existing plot to test creation
    field.plot_overview(plot_filename, show_calibration_patches=True)
    assert plot_path.exists()


def test_plot_overview_initial(field):
    plot_filename = "initial_field_overview.png"
    plot_path = Path(field.parset["dir_working"]) / "plots" / plot_filename

    assert plot_path.exists()
    plot_path.unlink()  # Remove existing plot to test creation

    field.plot_overview(plot_filename, show_initial_coverage=True)
    assert plot_path.exists()


def test_plot_overview_initial_near_pole(field):
    plot_filename = "initial_field_overview.png"
    plot_path = Path(field.parset["dir_working"]) / "plots" / plot_filename

    assert plot_path.exists()
    plot_path.unlink()  # Remove existing plot to test creation

    field.dec = 89.5
    field.plot_overview(plot_filename, show_initial_coverage=True)
    assert plot_path.exists()


@pytest.mark.parametrize(
    "do_slowgain_solve, do_fulljones_solve",
    [(True, True), (True, False), (False, True), (False, False)],
)
def test_set_calibration_strategy_legacy_default(field, do_slowgain_solve, do_fulljones_solve):
    """Test that the default calibration strategy is set correctly.

    This should capture the current behaviour of the pipeline using the
    legacy settings 'do_fulljones_solve' and 'do_slowgain_solve'in the strategy file.
    """
    step_dict = {
        "do_calibrate": True,
        "do_slowgain_solve": do_slowgain_solve,
        "do_fulljones_solve": do_fulljones_solve,
    }
    field.__dict__.update(step_dict)
    field.set_calibration_strategy()
    expected_strategy = {
        "dd": [
            "fast_phase",
            "medium_phase",
            *(["slow_gains"] if do_slowgain_solve else []),
        ],
        "di": [*(["full_jones"] if do_fulljones_solve else [])],
    }
    assert field.calibration_strategy == expected_strategy


@pytest.mark.parametrize(
    "do_slowgain_solve, do_fulljones_solve",
    [(True, True), (True, False), (False, True), (False, False)],
)
def test_set_calibration_strategy_user_provided(field, do_slowgain_solve, do_fulljones_solve):
    """Test that the calibration strategy is set correctly when provided.

    This captures the behaviour of the pipeline using the merged DD/DI classes.
    """
    user_provided_strategy = {
        "di": ["fast_phase", "medium_phase", "slow_gain", "full_jones"],
        "dd": ["fast_phase", "medium_phase", "slow_gain", "full_jones"],
    }
    step_dict = {
        "do_calibrate": True,
        "calibration_strategy": user_provided_strategy,
        # The following legacy settings should be ignored when a user-provided strategy is given
        "do_slowgain_solve": do_slowgain_solve,
        "do_fulljones_solve": do_fulljones_solve,
    }
    field.__dict__.update(step_dict)
    field.set_calibration_strategy()
    assert field.calibration_strategy == user_provided_strategy


@pytest.mark.parametrize(
    "strategy_items",
    [
        [
            ("di", ["fast_phase", "medium_phase"]),
            ("dd", ["fast_phase", "medium_phase"]),
        ],
        [
            ("dd", ["fast_phase", "medium_phase"]),
            ("di", ["fast_phase", "medium_phase"]),
        ],
    ],
)
def test_strategy_preserves_top_level_order(field, strategy_items):
    """Test that the order of the top-level keys in the calibration strategy is preserved when set."""
    user_provided_strategy = dict(strategy_items)
    field.__dict__.update(
        {
            "do_calibrate": True,
            "calibration_strategy": user_provided_strategy,
        }
    )
    field.set_calibration_strategy()

    assert list(field.calibration_strategy.items()) == strategy_items


@pytest.mark.parametrize("didd_order", [("di", "dd"), ("dd", "di")])
def test_set_calibration_strategy_preserves_order_of_di_vs_dd(field, didd_order):
    """Test that the calibration strategy preserves the order of DI vs DD keys."""
    user_provided_strategy = {
        didd_order[0]: ["fast_phase", "medium_phase", "slow_gain", "full_jones"],
        didd_order[1]: ["fast_phase", "medium_phase", "slow_gain", "full_jones"],
    }
    step_dict = {"do_calibrate": True, "calibration_strategy": user_provided_strategy}
    field.__dict__.update(step_dict)
    field.set_calibration_strategy()
    assert list(field.calibration_strategy.keys()) == list(user_provided_strategy.keys())
    assert field.calibration_strategy == user_provided_strategy


@pytest.mark.parametrize(
    "solve_order",
    [
        ("fast_phase", "medium_phase", "slow_gain", "full_jones"),
        ("full_jones", "slow_gain", "medium_phase", "fast_phase"),
    ],
)
def test_set_calibration_strategy_preserves_order_of_solves(field, solve_order):
    """Test that the calibration strategy preserves the order of DI vs DD keys."""
    user_provided_strategy = {
        "di": list(solve_order),
        "dd": list(solve_order),
    }
    step_dict = {"do_calibrate": True, "calibration_strategy": user_provided_strategy}
    field.__dict__.update(step_dict)
    field.set_calibration_strategy()
    assert list(field.calibration_strategy.keys()) == list(user_provided_strategy.keys())
    for key in user_provided_strategy.keys():
        assert list(field.calibration_strategy[key]) == list(user_provided_strategy[key])
    assert field.calibration_strategy == user_provided_strategy
