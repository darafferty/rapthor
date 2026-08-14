import logging
from pathlib import Path

import pytest
from lsmtool.testing import get_context

from rapthor.lib.field import Field
from rapthor.lib.strategy import set_image_strategy

logging.getLogger("matplotlib.font_manager").disabled = True

EXPECTED_ANTENNA_CONSTRAINTS = {
    "HBA": [
        "CS001HBA0",
        "CS002HBA0",
        "CS003HBA0",
        "CS004HBA0",
        "CS005HBA0",
        "CS006HBA0",
        "CS007HBA0",
        "CS011HBA0",
        "CS013HBA0",
        "CS017HBA0",
        "CS021HBA0",
        "CS024HBA0",
        "CS026HBA0",
        "CS028HBA0",
        "CS030HBA0",
        "CS031HBA0",
        "CS032HBA0",
        "CS101HBA0",
        "CS103HBA0",
        "CS201HBA0",
        "CS301HBA0",
        "CS302HBA0",
        "CS401HBA0",
        "CS501HBA0",
        "CS001HBA1",
        "CS002HBA1",
        "CS003HBA1",
        "CS004HBA1",
        "CS005HBA1",
        "CS006HBA1",
        "CS007HBA1",
        "CS011HBA1",
        "CS013HBA1",
        "CS017HBA1",
        "CS021HBA1",
        "CS024HBA1",
        "CS026HBA1",
        "CS028HBA1",
        "CS030HBA1",
        "CS031HBA1",
        "CS032HBA1",
        "CS101HBA1",
        "CS103HBA1",
        "CS201HBA1",
        "CS301HBA1",
        "CS302HBA1",
        "CS401HBA1",
        "CS501HBA1",
        "RS106HBA0",
        "RS205HBA0",
        "RS305HBA0",
        "RS306HBA0",
        "RS503HBA0",
        "RS106HBA1",
        "RS205HBA1",
        "RS305HBA1",
        "RS306HBA1",
        "RS503HBA1",
    ],
    "LBA": [
        "CS001LBA",
        "CS002LBA",
        "CS003LBA",
        "CS004LBA",
        "CS005LBA",
        "CS006LBA",
        "CS007LBA",
        "CS011LBA",
        "CS013LBA",
        "CS017LBA",
        "CS021LBA",
        "CS024LBA",
        "CS026LBA",
        "CS028LBA",
        "CS030LBA",
        "CS031LBA",
        "CS032LBA",
        "CS101LBA",
        "CS103LBA",
        "CS201LBA",
        "CS301LBA",
        "CS302LBA",
        "CS401LBA",
        "CS501LBA",
        "RS106LBA",
        "RS205LBA",
        "RS305LBA",
        "RS306LBA",
        "RS503LBA",
    ],
}


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


def test_update_image_strategy_without_calibration(field, monkeypatch):
    """Image-only strategy steps should update without calibration-only parameters."""
    monkeypatch.setattr(field, "update_skymodels", lambda *args, **kwargs: None)
    monkeypatch.setattr(field, "remove_skymodels", lambda: None)
    field.parset["regroup_input_skymodel"] = False
    field.outlier_sectors = []
    field.bright_source_sectors = []

    step_dict = set_image_strategy(field)[0]
    field.update(step_dict, index=1, final=True)

    assert field.do_calibrate is False
    assert field.do_image is True
    assert field.do_fulljones_solve is False


def test_update_later_image_only_cycle_reuses_previous_solution_layout(field, monkeypatch):
    """Later image-only cycles must keep the layout used by previous scalar solutions."""

    def fail_update_skymodels(*args, **kwargs):
        raise AssertionError("update_skymodels should not be called")

    def fail_remove_skymodels():
        raise AssertionError("remove_skymodels should not be called")

    monkeypatch.setattr(field, "update_skymodels", fail_update_skymodels)
    monkeypatch.setattr(field, "remove_skymodels", fail_remove_skymodels)
    field.h5parm_filename = "previous-solutions.h5"
    field.dd_h5parm_filename = "previous-solutions.h5"
    field.outlier_sectors = []
    field.bright_source_sectors = []

    step_dict = set_image_strategy(field)[0]
    step_dict["regroup_model"] = False
    field.update(step_dict, index=2, final=True)

    assert field.do_calibrate is False
    assert field.do_image is True


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


@pytest.mark.parametrize(
    "input_constraints, expected_result",
    [
        # nominal case. all stations are present in the constraints
        (
            [
                [
                    "CS001HBA0",
                    "CS002HBA0",
                    "CS002HBA1",
                    "CS004HBA1",
                ],
                [
                    "RS106HBA",
                    "RS208HBA",
                    "RS305HBA",
                    "RS307HBA",
                ],
            ],
            [
                [
                    "CS001HBA0",
                    "CS002HBA0",
                    "CS002HBA1",
                    "CS004HBA1",
                ],
                [
                    "RS106HBA",
                    "RS208HBA",
                    "RS305HBA",
                    "RS307HBA",
                ],
            ],
        ),
        # Constraints contain stations that are not present in the field. These
        # should be filtered out.
        (
            [
                [
                    "CS001HBA0",
                    "CS002HBA0",
                    "CS002HBA1",
                    "CS004HBA1",
                    "CS999HBA0",  # Not present in the field
                ],
                [
                    "RS106HBA",
                    "RS208HBA",
                    "RS305HBA",
                    "RS307HBA",
                    "RS999HBA",  # Not present in the field
                ],
            ],
            [
                [
                    "CS001HBA0",
                    "CS002HBA0",
                    "CS002HBA1",
                    "CS004HBA1",
                ],
                [
                    "RS106HBA",
                    "RS208HBA",
                    "RS305HBA",
                    "RS307HBA",
                ],
            ],
        ),
        # Single list of stations instead of a list of lists. Should be
        # wrapped in a list when resolved
        (
            [
                "CS001HBA0",
                "CS002HBA0",
                "CS002HBA1",
                "CS004HBA1",
            ],
            [
                [
                    "CS001HBA0",
                    "CS002HBA0",
                    "CS002HBA1",
                    "CS004HBA1",
                ]
            ],
        ),
        # Constraints with no stations present in the field. Should raise a
        # ValueError.
        (
            [
                [
                    "CS999HBA0",  # Not present in the field
                ],
            ],
            ValueError,
        ),
    ],
)
def test_resolve_antenna_constraints(field, input_constraints, expected_result):
    """Test that the antenna constraints are resolved correctly."""
    with get_context(expected_result):
        result = list(field._resolve_antenna_constraints(input_constraints))
        assert result == expected_result


@pytest.mark.parametrize(
    "antenna, antenna_constraints, expected_result",
    [
        pytest.param(
            "HBA",
            True,
            [["CS001HBA0", "CS002HBA0", "CS002HBA1", "CS004HBA1"]],
            id="LOFAR HBA constraints default",
        ),
        pytest.param("HBA", False, [], id="LOFAR HBA constraints False"),
        pytest.param("HBA", [], [], id="default LOFAR HBA constraints empty"),
        pytest.param(
            "LBA",
            True,
            ValueError,
            id="default LOFAR LBA constraints",
        ),
    ],
)
def test_antenna_constraints(field, antenna, antenna_constraints, expected_result):
    """Test that the antenna constraints are loaded c."""
    field.antenna = antenna
    field.antenna_constraints = antenna_constraints
    with get_context(expected_result):
        result = list(field.resolve_antenna_constraints())
        assert result == expected_result
