import json
import logging
import shutil
from pathlib import Path

import casacore.tables as pt
import lsmtool
import numpy as np
import pytest
from losoto.h5parm import h5parm
from matplotlib import pyplot as plt
from lsmtool.testing import get_context

from rapthor.execution.pipeline.lifecycle import chunk_observations
from rapthor.lib.field import Field, _ensure_skymodel_write_units
from rapthor.lib.observation import Observation
from rapthor.lib.strategy import check_and_adjust_parameters, set_user_strategy
from rapthor.operations.predict import Predict
from tests.conftest import write_scalar_h5parm

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
        "RS106HBA",
        "RS205HBA",
        "RS305HBA",
        "RS306HBA",
        "RS503HBA",
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


def save_json(filename, data):
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)


@pytest.fixture
def field(parset_for_field_test):

    field = Field(parset_for_field_test)
    field.fast_timestep_sec = 32.0
    field.update_skymodels(1, True, target_flux=0.2)
    field.set_obs_parameters()
    field.define_imaging_sectors()
    field.define_outlier_sectors(1)
    return field


@pytest.fixture
def calibration_strategy_field():
    field = Field.__new__(Field)
    field.calibration_strategy = None
    return field


def test_scan_observations(field):
    assert field.fwhm_ra_deg == 4.500843683229519


def test_scan_observations_uses_mean_station_diameter(
    parset_for_field_test,
    test_ms,
    tmp_path,
):
    second_ms = tmp_path / "different_diameter.ms"
    shutil.copytree(test_ms, second_ms)

    with pt.table(f"{second_ms}::SPECTRAL_WINDOW", readonly=False, ack=False) as table:
        channel_freq = table.getcol("CHAN_FREQ")
        channel_width = table.getcol("CHAN_WIDTH")
        frequency_offset = float(
            channel_freq.max() - channel_freq.min() + 2 * abs(channel_width).max()
        )
        table.putcol("CHAN_FREQ", channel_freq + frequency_offset)
        table.putcol("REF_FREQUENCY", table.getcol("REF_FREQUENCY") + frequency_offset)

    with pt.table(f"{test_ms}::ANTENNA", ack=False) as table:
        first_diameter = float(table.getcol("DISH_DIAMETER")[0])
    with pt.table(f"{second_ms}::ANTENNA", readonly=False, ack=False) as table:
        table.putcol("DISH_DIAMETER", table.getcol("DISH_DIAMETER") * 2.0)

    parset = parset_for_field_test.copy()
    parset["mss"] = [test_ms, str(second_ms)]
    mixed_field = Field(parset, minimal=True)

    expected_diameter = 1.5 * first_diameter
    mean_reference_frequency = np.mean(
        [observation.referencefreq for observation in mixed_field.full_observations]
    )
    expected_fwhm_ra = 1.1 * (3.0e8 / mean_reference_frequency) / expected_diameter * 180.0 / np.pi
    assert mixed_field.diam == pytest.approx(expected_diameter)
    assert mixed_field.fwhm_ra_deg == pytest.approx(expected_fwhm_ra)


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
    chunked_endtime = full_obs.endtime - 2 * full_obs.timepersample
    assert obs.numsamples == 2
    assert obs.starttime == chunked_starttime
    assert obs.endtime == chunked_endtime


@pytest.mark.parametrize(
    "num_samples, expected_sizes",
    [
        (3594, [1198, 1198, 1198]),
        (3595, [1198, 1198, 1199]),
        (3596, [1198, 1199, 1199]),
        (4791, [1597, 1597, 1597]),
    ],
)
def test_chunk_observations_full_data_remainder(field, mocker, num_samples, expected_sizes):
    obs = field.full_observations[0]
    obs.endtime += (num_samples - obs.numsamples) * obs.timepersample
    obs.numsamples = num_samples
    obs.data_fraction = 1.0
    create_observation = mocker.patch("rapthor.lib.observation.Observation")

    field.chunk_observations(1198 * obs.timepersample, prefer_high_el_periods=False)

    boundaries = np.cumsum([0, *expected_sizes])
    starts = [call.kwargs["starttime"] for call in create_observation.call_args_list]
    ends = [call.kwargs["endtime"] for call in create_observation.call_args_list]
    assert starts == pytest.approx(
        obs.starttime + boundaries[:-1] * obs.timepersample, rel=0, abs=0.001
    )
    assert ends == pytest.approx(
        obs.endtime - (num_samples - boundaries[1:]) * obs.timepersample, rel=0, abs=0.001
    )


@pytest.mark.parametrize(
    "num_samples, max_chunks, expected_sizes",
    [(5, None, [2, 3]), (6, None, [2, 2, 2]), (6, 2, [3, 3]), (5, 3, [2, 3])],
)
def test_chunk_observations_with_real_scan(field, num_samples, max_chunks, expected_sizes):
    ms_filename = field.full_observations[0].ms_filename
    with pt.table(ms_filename, ack=False) as table:
        times = np.unique(table.getcol("TIME"))
    obs = Observation(ms_filename, starttime=times[0], endtime=times[num_samples - 1])
    field.full_observations = [obs]

    field.chunk_observations(
        2 * obs.timepersample, prefer_high_el_periods=False, max_chunks=max_chunks
    )

    assert [chunk.numsamples for chunk in field.observations] == expected_sizes
    selected_times = [
        times[(times >= chunk.starttime) & (times <= chunk.endtime)] for chunk in field.observations
    ]
    assert [len(chunk_times) for chunk_times in selected_times] == expected_sizes
    np.testing.assert_array_equal(np.concatenate(selected_times), times[:num_samples])


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


def test_update_allows_target_number_without_target_flux(field, monkeypatch):
    captured = {}

    def fake_update_skymodels(index, regroup_model, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(field, "update_skymodels", fake_update_skymodels)
    monkeypatch.setattr(field, "remove_skymodels", lambda: None)
    field.lofar_to_true_flux_ratio = 2.0
    field.lofar_to_true_flux_std = 0.1
    field.apply_normalizations = False

    field.update(
        {
            "do_calibrate": True,
            "calibration_strategy": {"di": ["fast_phase"]},
            "regroup_model": True,
            "target_flux": None,
            "max_directions": 1,
            "max_distance": None,
            "peel_bright_sources": False,
            "peel_outliers": False,
            "reweight": False,
            "compress_selfcal_images": False,
            "compress_final_images": False,
        },
        index=2,
    )

    assert captured["target_flux"] is None
    assert captured["target_number"] == 1


def test_update_sets_skymodel_flags_in_repeated_final_cycles(field, monkeypatch):
    captured = {}

    def fake_update_skymodels(index, regroup_model, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(field, "update_skymodels", fake_update_skymodels)
    monkeypatch.setattr(field, "remove_skymodels", lambda: None)

    class Sector:
        name = "sector_1"

    # Set one imaging sector and one outlier sector to simluate
    # the first final cycle
    field.imaging_sectors = [Sector()]
    field.outlier_sectors = [Sector()]

    field.update(
        {
            "do_calibrate": False,
            "regroup_model": False,
            "peel_bright_sources": False,
            "peel_outliers": False,
            "reweight": False,
            "compress_selfcal_images": False,
        },
        index=2,
        final_cycle_number=1,
    )

    assert captured["combine_current_and_initial"] is True
    assert field.imaged_sources_only is False

    # Remove the outlier sector to simluate the removal done by the predict
    # operation (in the first final cycle) before the second final cycle starts
    field.outlier_sectors = []

    field.update(
        {
            "do_calibrate": False,
            "regroup_model": False,
            "peel_bright_sources": False,
            "peel_outliers": False,
            "reweight": False,
            "compress_selfcal_images": False,
        },
        index=3,
        final_cycle_number=2,
    )

    assert captured["combine_current_and_initial"] is False
    assert field.imaged_sources_only is True


def _cycle_step(peel_outliers):
    return {
        "do_calibrate": True,
        "calibration_strategy": {"dd": ["fast_phase"]},
        # The first cycle regroups the input sky model as set in the parset
        "regroup_model": False,
        "target_flux": None,
        "max_directions": 5,
        "max_distance": None,
        "peel_outliers": peel_outliers,
        "peel_bright_sources": False,
        "reweight": False,
        "compress_selfcal_images": False,
        "compress_final_images": False,
    }


def _finish_cycle(field, cycle, tmp_path):
    """Finalize Predict without its workflow, and stand in for Image with sky models
    that hold the input sources inside each imaging sector."""
    if field.do_predict:
        for sector in field.sectors:
            sector.set_prediction_parameters()
        Predict("dd", field, cycle).finalize()
    for sector in field.imaging_sectors:
        skymodel = sector.filter_skymodel(lsmtool.load(field.parset["input_skymodel"]))
        skymodel.group("single")
        skymodel_file = tmp_path / f"{sector.name}_cycle{cycle}_true_sky.txt"
        skymodel.write(str(skymodel_file))
        sector.image_skymodel_file_true_sky = str(skymodel_file)
        sector.image_skymodel_file_apparent_sky = None


@pytest.mark.parametrize("peel_outliers", [True, False], ids=["peeled", "not-peeled"])
def test_final_passes_peel_outliers_once_from_the_input_data(
    parset_for_field_test, tmp_path, peel_outliers
):
    """The final pass peels the input data again; its repeat reuses the peeled data.

    The full data of a single node are used, so observations are not chunked.
    """
    parset = parset_for_field_test
    parset["cluster_specific"]["max_nodes"] = 1
    # A small imaging sector leaves sources of the input sky model outside it
    parset["imaging_specific"]["grid_width_ra_deg"] = 1.5
    parset["imaging_specific"]["grid_width_dec_deg"] = 1.5
    field = Field(parset)
    input_ms = [obs.ms_filename for obs in field.full_observations]
    step = _cycle_step(peel_outliers)

    passes = []
    for cycle, final_cycle_number in ((1, 0), (2, 1), (3, 2)):
        if final_cycle_number <= 1:
            chunk_observations(field, [step], 1.0)
        field.update(dict(step), cycle, final_cycle_number=final_cycle_number)
        calibration_skymodel = lsmtool.load(field.calibration_skymodel_file)
        passes.append(
            {
                "calibrated_ms": [obs.ms_filename for obs in field.observations],
                "peels_outliers": bool(field.outlier_sectors and field.peel_outliers),
                "calibration_sources": len(calibration_skymodel),
            }
        )
        _finish_cycle(field, cycle, tmp_path)

    assert [obs.ms_filename for obs in field.full_observations] == input_ms
    selfcal, final_pass, repeat = passes
    assert selfcal["calibrated_ms"] == input_ms
    assert final_pass["calibrated_ms"] == input_ms
    if peel_outliers:
        assert [entry["peels_outliers"] for entry in passes] == [True, True, False]
        predict_dir = Path(field.working_dir, "pipelines", "predict_2")
        assert all(Path(ms).parent == predict_dir for ms in repeat["calibrated_ms"])
        # The repeat calibrates only the imaged sources, as the outliers are gone
        assert repeat["calibration_sources"] < final_pass["calibration_sources"]
    else:
        assert [entry["peels_outliers"] for entry in passes] == [False, False, False]
        assert repeat["calibrated_ms"] == input_ms


def test_update_skymodels_allows_target_number_without_target_flux(parset_for_field_test):
    field = Field(parset_for_field_test)

    field.update_skymodels(1, True, target_flux=None, target_number=1)

    assert field.target_flux is not None
    assert len(field.calibrator_patch_names) == 1


def test_update_skymodels_ignores_empty_apparent_sky_without_patches(field, tmp_path, monkeypatch):
    empty_skymodel = "FORMAT = Name, Type, Ra, Dec, I\n"
    true_sky = tmp_path / "sector_1.true_sky.txt"
    apparent_sky = tmp_path / "sector_1.apparent_sky.txt"
    true_sky.write_text(empty_skymodel, encoding="utf-8")
    apparent_sky.write_text(empty_skymodel, encoding="utf-8")

    class Sector:
        name = "sector_1"
        image_skymodel_file_true_sky = true_sky
        image_skymodel_file_apparent_sky = apparent_sky

    class StopAfterSkyModelUpdate(Exception):
        """Stop after make_skymodels captures the apparent-sky argument."""

    captured = {}

    def fake_make_skymodels(skymodel_true_sky, **kwargs):
        captured.update(kwargs)
        raise StopAfterSkyModelUpdate

    field.imaging_sectors = [Sector()]
    field.imaged_sources_only = True
    monkeypatch.setattr(field, "make_skymodels", fake_make_skymodels)

    with pytest.raises(StopAfterSkyModelUpdate):
        field.update_skymodels(6, False)

    assert captured["skymodel_apparent_sky"] is None


def test_update_skymodels_handles_empty_previous_cycle_sky_model(field, tmp_path, monkeypatch):
    empty_skymodel = "FORMAT = Name, Type, Ra, Dec, I\n"
    true_sky = tmp_path / "sector_1.true_sky.txt"
    apparent_sky = tmp_path / "sector_1.apparent_sky.txt"
    true_sky.write_text(empty_skymodel, encoding="utf-8")
    apparent_sky.write_text(empty_skymodel, encoding="utf-8")

    class Sector:
        name = "sector_1"
        image_skymodel_file_true_sky = true_sky
        image_skymodel_file_apparent_sky = apparent_sky

        def __init__(self):
            self.region_calls = []

        def make_vertices_file(self):
            self.region_calls.append(("vertices", None))

        def make_region_file(self, region_file):
            self.region_calls.append(("region", region_file))

    sector = Sector()
    field.imaging_sectors = [sector]
    field.imaged_sources_only = True
    monkeypatch.setattr(field, "plot_overview", lambda *args, **kwargs: None)

    field.update_skymodels(6, False)

    assert field.calibrator_patch_names == []
    assert field.calibrator_fluxes == []
    assert field.calibrator_positions == {}
    assert field.num_patches == 0
    assert field.get_calibration_radius() == 0.0
    assert sector.region_calls == [
        ("vertices", None),
        ("region", str(Path(field.working_dir) / "regions" / "sector_1_region_ds9.reg")),
    ]
    assert field.outlier_sectors == []
    assert field.bright_source_sectors == []
    assert field.predict_sectors == []


def test_empty_skymodel_write_units_are_restored(tmp_path):
    empty_skymodel = tmp_path / "empty_skymodel.txt"
    empty_skymodel.write_text("FORMAT = Name, Type, Ra, Dec, I\n", encoding="utf-8")

    skymodel = lsmtool.load(str(empty_skymodel))
    _ensure_skymodel_write_units(skymodel)

    output_skymodel = tmp_path / "output_skymodel.txt"
    skymodel.write(str(output_skymodel), clobber=True)

    assert output_skymodel.read_text(encoding="utf-8").startswith("FORMAT = Name, Type, Ra, Dec, I")


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
    plt.close("all")
    field.plot_overview(plot_filename, show_calibration_patches=True)
    assert plot_path.exists()
    assert plt.get_fignums() == []


def test_plot_overview_initial(field):
    plot_filename = "initial_field_overview.png"
    plot_path = Path(field.parset["dir_working"]) / "plots" / plot_filename

    assert plot_path.exists()
    plot_path.unlink()  # Remove existing plot to test creation

    plt.close("all")
    field.plot_overview(plot_filename, show_initial_coverage=True)
    assert plot_path.exists()
    assert plt.get_fignums() == []


def test_plot_overview_initial_near_pole(field):
    plot_filename = "initial_field_overview.png"
    plot_path = Path(field.parset["dir_working"]) / "plots" / plot_filename

    assert plot_path.exists()
    plot_path.unlink()  # Remove existing plot to test creation

    field.dec = 89.5
    plt.close("all")
    field.plot_overview(plot_filename, show_initial_coverage=True)
    assert plot_path.exists()
    assert plt.get_fignums() == []


def _update_field(field, monkeypatch, index, **step):
    """Run Field.update() for one cycle without touching sky models."""
    monkeypatch.setattr(field, "update_skymodels", lambda *args, **kwargs: None)
    monkeypatch.setattr(field, "remove_skymodels", lambda: None)
    field.lofar_to_true_flux_ratio = 1.0
    field.lofar_to_true_flux_std = 0.1
    field.apply_normalizations = False
    step_dict = {
        "do_calibrate": True,
        "regroup_model": False,
        "target_flux": None,
        "max_directions": 1,
        "max_distance": None,
        "peel_bright_sources": False,
        "peel_outliers": False,
        "reweight": False,
        "compress_selfcal_images": False,
        "compress_final_images": False,
    }
    step_dict.update(step)
    field.update(step_dict, index=index)


@pytest.mark.parametrize(
    "strategy_setting",
    [{}, {"calibration_strategy": None}, {"calibration_strategy": {}}],
    ids=["missing", "none", "empty"],
)
def test_update_resolves_omitted_calibration_strategy_for_each_cycle(
    field, monkeypatch, caplog, strategy_setting
):
    """A cycle that omits the strategy gets the default, whatever earlier cycles used."""
    explicit_strategy = {"di": ["fast_phase"], "dd": []}

    with caplog.at_level("INFO", logger="rapthor:field"):
        _update_field(field, monkeypatch, 2)
    assert field.calibration_strategy == {
        "dd": ["fast_phase", "medium_phase", "slow_gains", "medium_phase"],
        "di": [],
    }
    assert field._calibration_strategy_defaulted is True
    assert (
        f"Calibration strategy for cycle 2 (default): {field.calibration_strategy}" in caplog.text
    )

    with caplog.at_level("INFO", logger="rapthor:field"):
        _update_field(field, monkeypatch, 3, calibration_strategy=explicit_strategy)
    assert field.calibration_strategy == explicit_strategy
    assert field._calibration_strategy_defaulted is False
    assert (
        f"Calibration strategy for cycle 3 (from the strategy): {explicit_strategy}" in caplog.text
    )

    # Omitted again: the explicit strategy of cycle 3 must not carry over
    _update_field(field, monkeypatch, 4, **strategy_setting)
    assert field.calibration_strategy == {
        "dd": ["fast_phase", "medium_phase", "slow_gains", "medium_phase"],
        "di": [],
    }
    assert field._calibration_strategy_defaulted is True

    assert all(
        sector.calibration_strategy == field.calibration_strategy
        for sector in field.imaging_sectors
    )


def test_update_marks_an_adjusted_omitted_strategy_as_default(
    field, custom_strategy, monkeypatch, caplog
):
    """A step that went through check_and_adjust_parameters() still counts as defaulted."""
    field.parset["strategy"] = str(custom_strategy)
    strategy_steps = set_user_strategy(field)
    del strategy_steps[1]["calibration_strategy"]
    adjusted_steps = check_and_adjust_parameters(field, strategy_steps)

    with caplog.at_level("INFO", logger="rapthor:field"):
        _update_field(field, monkeypatch, 2, **adjusted_steps[1])

    assert field.calibration_strategy == {
        "dd": ["fast_phase", "medium_phase", "slow_gains", "medium_phase"],
        "di": [],
    }
    assert field._calibration_strategy_defaulted is True
    assert "Calibration strategy for cycle 2 (default)" in caplog.text


def test_update_ignores_legacy_solve_flags_without_calibration_strategy(field, monkeypatch):
    """Legacy flags are translated when the strategy is loaded, not by Field.update()."""
    _update_field(field, monkeypatch, 2, do_slowgain_solve=False, do_fulljones_solve=True)
    assert field.calibration_strategy == {
        "dd": ["fast_phase", "medium_phase", "slow_gains", "medium_phase"],
        "di": [],
    }
    assert field._calibration_strategy_defaulted is True


@pytest.mark.parametrize(
    "strategy_setting",
    [{}, {"calibration_strategy": None}, {"calibration_strategy": {}}],
    ids=["missing", "none", "empty"],
)
@pytest.mark.parametrize("defaulted", [False, True], ids=["explicit-previous", "default-previous"])
def test_update_keeps_last_calibration_strategy_for_image_only_cycle(
    field, monkeypatch, strategy_setting, defaulted
):
    """An image-only cycle applies the solutions of the last calibration."""
    explicit_strategy = {"di": ["slow_gains"], "dd": ["fast_phase"]}
    _update_field(
        field, monkeypatch, 2, calibration_strategy=None if defaulted else explicit_strategy
    )
    previous_strategy = field.calibration_strategy

    _update_field(field, monkeypatch, 3, do_calibrate=False, **strategy_setting)
    assert field.calibration_strategy == previous_strategy
    assert field._calibration_strategy_defaulted is defaulted
    assert all(sector.calibration_strategy == previous_strategy for sector in field.imaging_sectors)

    # A later calibrating cycle without a strategy starts afresh from the default
    _update_field(field, monkeypatch, 4)
    assert field._calibration_strategy_defaulted is True
    assert field.calibration_strategy["dd"] == [
        "fast_phase",
        "medium_phase",
        "slow_gains",
        "medium_phase",
    ]


def test_update_allows_an_explicit_strategy_for_an_image_only_cycle(field, monkeypatch):
    _update_field(field, monkeypatch, 2)
    strategy = {"di": ["slow_gains"], "dd": []}

    _update_field(field, monkeypatch, 3, do_calibrate=False, calibration_strategy=strategy)

    assert field.calibration_strategy == strategy
    assert field._calibration_strategy_defaulted is False
    assert all(sector.calibration_strategy == strategy for sector in field.imaging_sectors)


def test_set_calibration_strategy_default(calibration_strategy_field):
    """The default calibration strategy is explicit and ignores legacy flags."""
    calibration_strategy_field.__dict__.update(
        {
            "do_calibrate": True,
            "do_slowgain_solve": False,
            "do_fulljones_solve": True,
        }
    )
    calibration_strategy_field.set_calibration_strategy()
    assert calibration_strategy_field.calibration_strategy == {
        "dd": ["fast_phase", "medium_phase", "slow_gains", "medium_phase"],
        "di": [],
    }
    assert calibration_strategy_field._calibration_strategy_defaulted is True


def test_set_calibration_strategy_user_provided(calibration_strategy_field):
    """Test that the calibration strategy is set correctly when provided.

    This captures the behaviour of the pipeline using the merged DD/DI classes.
    """
    user_provided_strategy = {
        "di": ["fast_phase", "medium_phase", "slow_gains", "full_jones"],
        "dd": ["fast_phase", "medium_phase", "slow_gains", "full_jones"],
    }
    step_dict = {
        "do_calibrate": True,
        "calibration_strategy": user_provided_strategy,
    }
    calibration_strategy_field.__dict__.update(step_dict)
    calibration_strategy_field.set_calibration_strategy()
    assert calibration_strategy_field.calibration_strategy == user_provided_strategy
    assert calibration_strategy_field._calibration_strategy_defaulted is False


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
def test_strategy_preserves_top_level_order(calibration_strategy_field, strategy_items):
    """Test that the order of the top-level keys in the calibration strategy is preserved when set."""
    user_provided_strategy = dict(strategy_items)
    calibration_strategy_field.__dict__.update(
        {
            "do_calibrate": True,
            "calibration_strategy": user_provided_strategy,
        }
    )
    calibration_strategy_field.set_calibration_strategy()

    assert list(calibration_strategy_field.calibration_strategy.items()) == strategy_items


@pytest.mark.parametrize("didd_order", [("di", "dd"), ("dd", "di")])
def test_set_calibration_strategy_preserves_order_of_di_vs_dd(
    calibration_strategy_field, didd_order
):
    """Test that the calibration strategy preserves the order of DI vs DD keys."""
    user_provided_strategy = {
        didd_order[0]: ["fast_phase", "medium_phase", "slow_gains", "full_jones"],
        didd_order[1]: ["fast_phase", "medium_phase", "slow_gains", "full_jones"],
    }
    step_dict = {"do_calibrate": True, "calibration_strategy": user_provided_strategy}
    calibration_strategy_field.__dict__.update(step_dict)
    calibration_strategy_field.set_calibration_strategy()
    assert list(calibration_strategy_field.calibration_strategy.keys()) == list(
        user_provided_strategy.keys()
    )
    assert calibration_strategy_field.calibration_strategy == user_provided_strategy


@pytest.mark.parametrize(
    "solve_order",
    [
        ("fast_phase", "medium_phase", "slow_gains", "full_jones"),
        ("full_jones", "slow_gains", "medium_phase", "fast_phase"),
    ],
)
def test_set_calibration_strategy_preserves_order_of_solves(
    calibration_strategy_field, solve_order
):
    """Test that the calibration strategy preserves the order of DI vs DD keys."""
    user_provided_strategy = {
        "di": list(solve_order),
        "dd": list(solve_order),
    }
    step_dict = {"do_calibrate": True, "calibration_strategy": user_provided_strategy}
    calibration_strategy_field.__dict__.update(step_dict)
    calibration_strategy_field.set_calibration_strategy()
    assert list(calibration_strategy_field.calibration_strategy.keys()) == list(
        user_provided_strategy.keys()
    )
    for key in user_provided_strategy.keys():
        assert list(calibration_strategy_field.calibration_strategy[key]) == list(
            user_provided_strategy[key]
        )
    assert calibration_strategy_field.calibration_strategy == user_provided_strategy


@pytest.fixture
def solution_state_field():
    """A bare Field with only the solution-file attributes that scan_h5parms reads."""
    field = Field.__new__(Field)
    field.h5parm_filename = None
    field.dd_h5parm_filename = None
    field.di_h5parm_filename = None
    field.fulljones_h5parm_filename = None
    field.apply_amplitudes = False
    field.dd_apply_amplitudes = False
    field.di_apply_amplitudes = False
    return field


ANTENNAS = ["CS001HBA0", "CS002HBA0"]


def test_scan_h5parms_keeps_dd_amplitudes_after_a_phase_only_di_product(
    solution_state_field, tmp_path
):
    """DD slow gains stay applied when a later DI phase solve becomes the current product."""
    field = solution_state_field
    field.dd_h5parm_filename = write_scalar_h5parm(
        tmp_path / "dd.h5", ANTENNAS, with_amplitudes=True
    )
    field.di_h5parm_filename = write_scalar_h5parm(tmp_path / "di.h5", ANTENNAS)
    # Calibrate.finalize() points h5parm_filename at the most recent product.
    field.h5parm_filename = field.di_h5parm_filename

    field.scan_h5parms()

    assert field.dd_apply_amplitudes is True
    assert field.di_apply_amplitudes is False
    assert field.apply_amplitudes is False


def test_scan_h5parms_keeps_di_amplitudes_after_a_phase_only_dd_product(
    solution_state_field, tmp_path
):
    field = solution_state_field
    field.dd_h5parm_filename = write_scalar_h5parm(tmp_path / "dd.h5", ANTENNAS)
    field.di_h5parm_filename = write_scalar_h5parm(
        tmp_path / "di.h5", ANTENNAS, with_amplitudes=True
    )
    field.h5parm_filename = field.dd_h5parm_filename

    field.scan_h5parms()

    assert field.dd_apply_amplitudes is False
    assert field.di_apply_amplitudes is True
    assert field.apply_amplitudes is False


def test_scan_h5parms_treats_a_lone_supplied_h5parm_as_the_dd_product(
    solution_state_field, tmp_path
):
    """An input h5parm without DI or DD ownership keeps the legacy single-flag behaviour."""
    field = solution_state_field
    field.h5parm_filename = write_scalar_h5parm(
        tmp_path / "input.h5", ANTENNAS, with_amplitudes=True
    )

    field.scan_h5parms()

    assert field.apply_amplitudes is True
    assert field.dd_apply_amplitudes is True
    assert field.di_apply_amplitudes is False


def test_scan_h5parms_resets_flags_when_products_are_cleared(solution_state_field, tmp_path):
    field = solution_state_field
    field.dd_apply_amplitudes = True
    field.di_apply_amplitudes = True
    field.apply_amplitudes = True

    field.scan_h5parms()

    assert field.apply_amplitudes is False
    assert field.dd_apply_amplitudes is False
    assert field.di_apply_amplitudes is False


def test_scan_h5parms_rejects_a_scalar_product_without_phases(solution_state_field, tmp_path):
    path = tmp_path / "amplitudes_only.h5"
    with h5parm(str(path), readonly=False) as solutions:
        solset = solutions.makeSolset("sol000")
        solset.makeSoltab(
            "amplitude",
            "amplitude000",
            axesNames=["ant"],
            axesVals=[ANTENNAS],
            vals=np.ones(len(ANTENNAS)),
            weights=np.ones(len(ANTENNAS)),
        )
    solution_state_field.dd_h5parm_filename = str(path)

    with pytest.raises(ValueError, match="phase000"):
        solution_state_field.scan_h5parms()


def test_scan_h5parms_derives_timesteps(solution_state_field, tmp_path):
    field = solution_state_field
    field.fast_timestep_sec = 0
    field.slow_timestep_sec = 0
    field.fulljones_timestep_sec = 0

    path = tmp_path / "field_solutions.h5"
    pols = np.array(["XX", "YY"])  # diagonal solve
    with h5parm(str(path), readonly=False) as solutions:
        solset = solutions.makeSolset("sol000")
        times = np.arange(0, 3600, 600)  # 600 s solution interval
        solset.makeSoltab(
            "amplitude",
            "amplitude000",
            axesNames=["ant", "time", "pol"],
            axesVals=[ANTENNAS, times, pols],
            vals=np.ones((len(ANTENNAS), len(times), len(pols))),
            weights=np.ones((len(ANTENNAS), len(times), len(pols))),
        )
        times = np.arange(0, 3600, 30)  # 30 s solution interval
        solset.makeSoltab(
            "phase",
            "phase000",
            axesNames=["ant", "time", "pol"],
            axesVals=[ANTENNAS, times, pols],
            vals=np.ones((len(ANTENNAS), len(times), len(pols))),
            weights=np.ones((len(ANTENNAS), len(times), len(pols))),
        )
    field.h5parm_filename = str(path)

    path = tmp_path / "fulljones_solutions.h5"
    pols = np.array(["XX", "XY", "YX", "YY"])  # full-Jones solve
    times = np.arange(0, 3600, 600)  # 600 s solution interval
    with h5parm(str(path), readonly=False) as solutions:
        solset = solutions.makeSolset("sol000")
        solset.makeSoltab(
            "amplitude",
            "amplitude000",
            axesNames=["ant", "time", "pol"],
            axesVals=[ANTENNAS, times, pols],
            vals=np.ones((len(ANTENNAS), len(times), len(pols))),
            weights=np.ones((len(ANTENNAS), len(times), len(pols))),
        )
        solset.makeSoltab(
            "phase",
            "phase000",
            axesNames=["ant", "time", "pol"],
            axesVals=[ANTENNAS, times, pols],
            vals=np.ones((len(ANTENNAS), len(times), len(pols))),
            weights=np.ones((len(ANTENNAS), len(times), len(pols))),
        )
    field.fulljones_h5parm_filename = str(path)

    field.scan_h5parms(derive_solution_timesteps=True)

    assert field.fast_timestep_sec == 30
    assert field.slow_timestep_sec == 600
    assert field.fulljones_timestep_sec == 600


def test_scan_h5parms_with_single_solution_in_time(solution_state_field, tmp_path):
    field = solution_state_field
    field.fast_timestep_sec = 8
    field.slow_timestep_sec = 32
    field.fulljones_timestep_sec = 32

    path = tmp_path / "field_solutions.h5"
    times = np.array([0])  # single solution
    pols = np.array(["XX", "YY"])  # diagonal solve
    with h5parm(str(path), readonly=False) as solutions:
        solset = solutions.makeSolset("sol000")
        solset.makeSoltab(
            "amplitude",
            "amplitude000",
            axesNames=["ant", "time", "pol"],
            axesVals=[ANTENNAS, times, pols],
            vals=np.ones((len(ANTENNAS), len(times), len(pols))),
            weights=np.ones((len(ANTENNAS), len(times), len(pols))),
        )
        solset.makeSoltab(
            "phase",
            "phase000",
            axesNames=["ant", "time", "pol"],
            axesVals=[ANTENNAS, times, pols],
            vals=np.ones((len(ANTENNAS), len(times), len(pols))),
            weights=np.ones((len(ANTENNAS), len(times), len(pols))),
        )
    field.h5parm_filename = str(path)

    path = tmp_path / "fulljones_solutions.h5"
    pols = np.array(["XX", "XY", "YX", "YY"])  # full-Jones solve
    with h5parm(str(path), readonly=False) as solutions:
        solset = solutions.makeSolset("sol000")
        solset.makeSoltab(
            "amplitude",
            "amplitude000",
            axesNames=["ant", "time", "pol"],
            axesVals=[ANTENNAS, times, pols],
            vals=np.ones((len(ANTENNAS), len(times), len(pols))),
            weights=np.ones((len(ANTENNAS), len(times), len(pols))),
        )
        solset.makeSoltab(
            "phase",
            "phase000",
            axesNames=["ant", "time", "pol"],
            axesVals=[ANTENNAS, times, pols],
            vals=np.ones((len(ANTENNAS), len(times), len(pols))),
            weights=np.ones((len(ANTENNAS), len(times), len(pols))),
        )
    field.fulljones_h5parm_filename = str(path)

    field.scan_h5parms(derive_solution_timesteps=True)

    # Check that initial values are unaltered
    assert field.fast_timestep_sec == 8
    assert field.slow_timestep_sec == 32
    assert field.fulljones_timestep_sec == 32


def test_scan_h5parms_with_screen_solutions(solution_state_field, tmp_path):
    field = solution_state_field
    field.fast_timestep_sec = 0
    field.slow_timestep_sec = 0

    path = tmp_path / "field_solutions.h5"
    pols = np.array(["XX", "YY"])  # diagonal solve
    with h5parm(str(path), readonly=False) as solutions:
        solset = solutions.makeSolset("coefficients000")
        times = np.arange(0, 3600, 600)  # 600 s solution interval
        solset.makeSoltab(
            "amplitude",
            "amplitude1_coefficients",
            axesNames=["ant", "time", "pol"],
            axesVals=[ANTENNAS, times, pols],
            vals=np.ones((len(ANTENNAS), len(times), len(pols))),
            weights=np.ones((len(ANTENNAS), len(times), len(pols))),
        )
        times = np.arange(0, 3600, 30)  # 30 s solution interval
        solset.makeSoltab(
            "phase",
            "phase_coefficients",
            axesNames=["ant", "time", "pol"],
            axesVals=[ANTENNAS, times, pols],
            vals=np.ones((len(ANTENNAS), len(times), len(pols))),
            weights=np.ones((len(ANTENNAS), len(times), len(pols))),
        )
    field.h5parm_filename = str(path)

    field.scan_h5parms(derive_solution_timesteps=True)

    assert field.fast_timestep_sec == 30
    assert field.slow_timestep_sec == 600


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

    @pytest.fixture
    def input_constraints(self, pytestconfig, request):
        # Resolve the input constraints filename if needed
        input_constraints = request.param
        if isinstance(input_constraints, str):
            return pytestconfig.resource_dir / input_constraints
        return input_constraints

    @pytest.mark.parametrize(
        "antenna, input_constraints, expected_result",
        [
            # Case: When `field.antenna_constraints` is True, load the HBA
            # constraints from file and match against the field stations. The
            # defaults constraints group the core stations together and leave
            # the remote stations unconstrained.
            pytest.param(
                "HBA",
                True,
                [["CS001HBA0", "CS002HBA0", "CS002HBA1", "CS004HBA1", "RS106HBA", "RS305HBA"]],
                id="LOFAR HBA constraints default",
            ),
            # Case: When `field.antenna_constraints` is False, do not load any
            # constraints.
            pytest.param("HBA", False, [], id="LOFAR HBA constraints False"),
            # Case: Passing a constraints file
            pytest.param(
                "HBA",
                "example_antenna_constraints.json",
                [
                    ["CS001HBA0", "CS002HBA0", "CS002HBA1", "CS004HBA1"],
                    ["RS106HBA", "RS208HBA", "RS305HBA", "RS307HBA"],
                ],
                id="LOFAR HBA constraints from file",
            ),
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
        indirect=["input_constraints"],
    )
    def test_antenna_constraints(
        self,
        mock_hba_field_with_stations,
        antenna,
        input_constraints,
        expected_result,
    ):
        """
        Test that the antenna constraints are loaded correctly from file for
        LOFAR HBA test dataset when required, and that an error is raised when
        we attempt to use antenna constraints for LBA on HBA data.
        """
        # Arrange
        mock_hba_field_with_stations.antenna = antenna
        mock_hba_field_with_stations.antenna_constraints = input_constraints

        with get_context(expected_result):
            # Act
            result = list(mock_hba_field_with_stations.resolve_antenna_constraints())
            # Assert
            assert result == expected_result

    @pytest.mark.parametrize(
        "input_constraints, expected_result",
        [
            # Case. All field stations are given in the constraints
            pytest.param(
                # input_constraints
                [
                    ["CS001HBA0", "CS002HBA0", "CS002HBA1", "CS004HBA1"],
                    ["RS106HBA", "RS208HBA", "RS305HBA", "RS307HBA"],
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
