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


class TestAntennaConstraints:
    """Test the resolution of antenna constraints in the Field class."""

    @pytest.fixture
    def mock_hba_field_with_stations(self):
        """
        Create a mock Field object with specified stations matching those in the
        LOFAR HBA test data set.
        """
        field = Field.__new__(Field)  # Create an uninitialized Field instance
        field.log = logging.getLogger("rapthor:field")
        field.stations = [
            "CS001HBA0",
            "CS002HBA0",
            "CS002HBA1",
            "CS004HBA1",
            "RS106HBA",
            "RS208HBA",
            "RS305HBA",
            "RS307HBA",
        ]
        return field

    @pytest.mark.parametrize(
        "antenna, antenna_constraints, expected_result",
        [
            # Case: When `field.antenna_constraints` is True, load the HBA
            # constraints from file and match against the field stations. The
            # defaults constraints group the core stations together and leave
            # the remote stations unconstrained.
            pytest.param(
                "HBA",
                True,
                [["CS001HBA0", "CS002HBA0", "CS002HBA1", "CS004HBA1"]],
                id="LOFAR HBA constraints default",
            ),
            # Case: When `field.antenna_constraints` is False, do not load any
            # constraints.
            pytest.param("HBA", False, [], id="LOFAR HBA constraints False"),
            # Case: When `field.antenna_constraints` is an empty list, do not
            # load any constraints.
            pytest.param("HBA", [], [], id="LOFAR HBA constraints empty"),
            # When we change the `field.antenna` attribute to "LBA", using True
            # for antenna_constraints loads the default constraints for LBA,
            # which is not compatible with the HBA data in the test field. This
            # should raise a ValueError.
            pytest.param(
                "LBA",
                True,
                pytest.raises(
                    ValueError,
                    match="Could not match any field stations to the station "
                    "names given in antenna constraints",
                ),
                id="LOFAR LBA constraints",
            ),
        ],
    )
    def test_antenna_constraints(
        self, mock_hba_field_with_stations, antenna, antenna_constraints, expected_result
    ):
        """
        Test that the antenna constraints are loaded correctly from file for
        LOFAR HBA test dataset when required, and that an error is raised when
        we attempt to use antenna constraints for LBA on HBA data.
        """

        # Arrange
        field = mock_hba_field_with_stations
        field.antenna = antenna
        field.antenna_constraints = antenna_constraints

        with get_context(expected_result):
            # Act
            result = list(field.resolve_antenna_constraints())
            # Assert
            assert result == expected_result

    @pytest.mark.parametrize(
        "input_constraints, expected_result",
        [
            # Case. All field stations are given in the constraints
            pytest.param(
                # input_constraints
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
                # expected_result
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
                id="All field stations present in constraints",
            ),
            # Constraints contain stations that are not present in the field. These
            # should be filtered out.
            pytest.param(
                # input_constraints
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
                # expected_result
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
                id="Constraints contain stations not present in the field",
            ),
            # Single list of stations instead of a list of lists. Should be
            # wrapped in a list when resolved
            pytest.param(
                # input_constraints
                [
                    "CS001HBA0",
                    "CS002HBA0",
                    "CS002HBA1",
                    "CS004HBA1",
                ],
                # expected_result
                [
                    [
                        "CS001HBA0",
                        "CS002HBA0",
                        "CS002HBA1",
                        "CS004HBA1",
                    ]
                ],
                id="Single list of stations resolves to list of lists",
            ),
            # Constraints with no stations present in the field. Should raise a
            # ValueError.
            pytest.param(
                # input_constraints
                [
                    [
                        "CS999HBA0",  # Not present in the field
                    ],
                ],
                # expected_result
                ValueError,
                id="Constraints with no stations present in the field raises ValueError",
            ),
        ],
    )
    def test_resolve_antenna_constraints(
        self, mock_hba_field_with_stations, input_constraints, expected_result
    ):
        """Test that the antenna constraints are resolved correctly."""
        with get_context(expected_result):
            result = list(
                mock_hba_field_with_stations._resolve_antenna_constraints(input_constraints)
            )
            assert result == expected_result

    @pytest.fixture
    def mock_ska_field_with_stations(self):
        """
        Create a mock Field object with specified stations matching those
        expected from SKA AA2 data set.
        """
        field = Field.__new__(Field)  # Create an uninitialized Field instance
        field.log = logging.getLogger("rapthor:field")
        field.stations = [
            "s0000 (station315_E8-1)",
            "s0001 (station316_E8-2)",
            "s0002 (station317_E8-3)",
            "s0003 (station318_E8-4)",
            "s0004 (station321_E9-1)",
            "s0005 (station322_E9-2)",
            "s0006 (station323_E9-3)",
            "s0007 (station324_E9-4)",
            "s0008 (station345_S8-1)",
            "s0009 (station346_S8-2)",
            "s0010 (station347_S8-3)",
            "s0011 (station348_S8-4)",
            "s0012 (station349_S8-5)",
            "s0013 (station350_S8-6)",
            "s0014 (station352_S9-2)",
            "s0015 (station353_S9-3)",
            "s0016 (station354_S9-4)",
            "s0017 (station355_S9-5)",
            "s0018 (station375_N8-1)",
            "s0019 (station376_N8-2)",
            "s0020 (station377_N8-3)",
            "s0021 (station378_N8-4)",
            "s0022 (station381_N9-1)",
            "s0023 (station382_N9-2)",
            "s0024 (station383_N9-3)",
            "s0025 (station384_N9-4)",
            "s0026 (station387_E10-1)",
            "s0027 (station388_E10-2)",
            "s0028 (station389_E10-3)",
            "s0029 (station390_E10-4)",
            "s0030 (station405_E13-1)",
            "s0031 (station406_E13-2)",
            "s0032 (station407_E13-3)",
            "s0033 (station408_E13-4)",
            "s0034 (station429_S10-1)",
            "s0035 (station430_S10-2)",
            "s0036 (station431_S10-3)",
            "s0037 (station432_S10-4)",
            "s0038 (station433_S10-5)",
            "s0039 (station434_S10-6)",
            "s0040 (station447_S13-1)",
            "s0041 (station448_S13-2)",
            "s0042 (station449_S13-3)",
            "s0043 (station450_S13-4)",
            "s0044 (station460_S15-2)",
            "s0045 (station461_S15-3)",
            "s0046 (station463_S15-5)",
            "s0047 (station464_S15-6)",
            "s0048 (station465_S16-1)",
            "s0049 (station466_S16-2)",
            "s0050 (station467_S16-3)",
            "s0051 (station468_S16-4)",
            "s0052 (station471_N10-1)",
            "s0053 (station472_N10-2)",
            "s0054 (station473_N10-3)",
            "s0055 (station474_N10-4)",
            "s0056 (station489_N13-1)",
            "s0057 (station490_N13-2)",
            "s0058 (station491_N13-3)",
            "s0059 (station492_N13-4)",
            "s0060 (station501_N15-1)",
            "s0061 (station502_N15-2)",
            "s0062 (station503_N15-3)",
            "s0063 (station504_N15-4)",
            "s0064 (station507_N16-1)",
            "s0065 (station508_N16-2)",
            "s0066 (station509_N16-3)",
            "s0067 (station510_N16-4)",
        ]
        return field

    @pytest.mark.parametrize(
        "input_constraints, expected_result",
        [
            (
                [
                    ["E10-1", "E10-2", "E10-3", "E10-4"],
                    ["E13-1", "E13-2", "E13-3", "E13-4"],
                    ["E8-1", "E8-2", "E8-3", "E8-4"],
                    ["E9-1", "E9-2", "E9-3", "E9-4"],
                    ["N10-1", "N10-2", "N10-3", "N10-4"],
                    ["N13-1", "N13-2", "N13-3", "N13-4"],
                    ["N15-1", "N15-2", "N15-3", "N15-4"],
                    ["N16-1", "N16-2", "N16-3", "N16-4"],
                    ["N8-1", "N8-2", "N8-3", "N8-4"],
                    ["N9-1", "N9-2", "N9-3", "N9-4"],
                    ["S10-1", "S10-2", "S10-3", "S10-4", "S10-5", "S10-6"],
                    ["S13-1", "S13-2", "S13-3", "S13-4"],
                    ["S15-1", "S15-2", "S15-3", "S15-4", "S15-5", "S15-6"],
                    ["S16-1", "S16-2", "S16-3", "S16-4"],
                    ["S8-1", "S8-2", "S8-3", "S8-4", "S8-5", "S8-6"],
                    ["S9-1", "S9-2", "S9-3", "S9-4", "S9-5", "S9-6"],
                ],
                [
                    [
                        "s0026 (station387_E10-1)",
                        "s0027 (station388_E10-2)",
                        "s0028 (station389_E10-3)",
                        "s0029 (station390_E10-4)",
                    ],
                    [
                        "s0030 (station405_E13-1)",
                        "s0031 (station406_E13-2)",
                        "s0032 (station407_E13-3)",
                        "s0033 (station408_E13-4)",
                    ],
                    [
                        "s0000 (station315_E8-1)",
                        "s0001 (station316_E8-2)",
                        "s0002 (station317_E8-3)",
                        "s0003 (station318_E8-4)",
                    ],
                    [
                        "s0004 (station321_E9-1)",
                        "s0005 (station322_E9-2)",
                        "s0006 (station323_E9-3)",
                        "s0007 (station324_E9-4)",
                    ],
                    [
                        "s0052 (station471_N10-1)",
                        "s0053 (station472_N10-2)",
                        "s0054 (station473_N10-3)",
                        "s0055 (station474_N10-4)",
                    ],
                    [
                        "s0056 (station489_N13-1)",
                        "s0057 (station490_N13-2)",
                        "s0058 (station491_N13-3)",
                        "s0059 (station492_N13-4)",
                    ],
                    [
                        "s0060 (station501_N15-1)",
                        "s0061 (station502_N15-2)",
                        "s0062 (station503_N15-3)",
                        "s0063 (station504_N15-4)",
                    ],
                    [
                        "s0064 (station507_N16-1)",
                        "s0065 (station508_N16-2)",
                        "s0066 (station509_N16-3)",
                        "s0067 (station510_N16-4)",
                    ],
                    [
                        "s0018 (station375_N8-1)",
                        "s0019 (station376_N8-2)",
                        "s0020 (station377_N8-3)",
                        "s0021 (station378_N8-4)",
                    ],
                    [
                        "s0022 (station381_N9-1)",
                        "s0023 (station382_N9-2)",
                        "s0024 (station383_N9-3)",
                        "s0025 (station384_N9-4)",
                    ],
                    [
                        "s0034 (station429_S10-1)",
                        "s0035 (station430_S10-2)",
                        "s0036 (station431_S10-3)",
                        "s0037 (station432_S10-4)",
                        "s0038 (station433_S10-5)",
                        "s0039 (station434_S10-6)",
                    ],
                    [
                        "s0040 (station447_S13-1)",
                        "s0041 (station448_S13-2)",
                        "s0042 (station449_S13-3)",
                        "s0043 (station450_S13-4)",
                    ],
                    [
                        "s0044 (station460_S15-2)",
                        "s0045 (station461_S15-3)",
                        "s0046 (station463_S15-5)",
                        "s0047 (station464_S15-6)",
                    ],
                    [
                        "s0048 (station465_S16-1)",
                        "s0049 (station466_S16-2)",
                        "s0050 (station467_S16-3)",
                        "s0051 (station468_S16-4)",
                    ],
                    [
                        "s0008 (station345_S8-1)",
                        "s0009 (station346_S8-2)",
                        "s0010 (station347_S8-3)",
                        "s0011 (station348_S8-4)",
                        "s0012 (station349_S8-5)",
                        "s0013 (station350_S8-6)",
                    ],
                    [
                        "s0014 (station352_S9-2)",
                        "s0015 (station353_S9-3)",
                        "s0016 (station354_S9-4)",
                        "s0017 (station355_S9-5)",
                    ],
                ],
            )
        ],
    )
    def test_resolve_antenna_constraints_oskar(
        self, mock_ska_field_with_stations, input_constraints, expected_result
    ):
        """Test that the antenna constraints are resolved correctly."""
        with get_context(expected_result):
            result = list(
                mock_ska_field_with_stations._resolve_antenna_constraints(input_constraints)
            )
            assert result == expected_result
