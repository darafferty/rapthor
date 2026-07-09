"""
Tests for parset parsing and sky-model option adjustment.
"""

import ast
import string
import textwrap
from dataclasses import dataclass
from pathlib import Path

import pytest

from rapthor.lib.parset import check_and_adjust_skymodel_settings, parset_read

RESOURCE_DIR = Path(__file__).parents[1] / "resources"


@dataclass(frozen=True)
class ParsetScenario:
    parset: Path
    input_ms: Path
    working_dir: Path


@pytest.fixture
def parset_scenario(tmp_path):
    input_ms = tmp_path / "input.ms"
    input_ms.mkdir()
    working_dir = tmp_path / "work"
    working_dir.mkdir()
    parset = working_dir / "test.parset"
    parset.write_text(
        textwrap.dedent(
            f"""
            [global]
            input_ms = {input_ms}
            dir_working = {working_dir}
            """
        ),
        encoding="utf-8",
    )
    return ParsetScenario(parset=parset, input_ms=input_ms, working_dir=working_dir)


def _append_to_parset(parset_path, text):
    with parset_path.open("a", encoding="utf-8") as parset_file:
        parset_file.write(text)


def _render_resource_template(template_name, **values):
    return string.Template((RESOURCE_DIR / template_name).read_text(encoding="utf-8")).substitute(
        **values
    )


def _make_skymodel_settings(**overrides):
    parset_dict = {
        "input_skymodel": None,
        "generate_initial_skymodel": False,
        "download_initial_skymodel": False,
        "apparent_skymodel": None,
        "cluster_specific": {"allow_internet_access": True},
        "imaging_specific": {
            "astrometry_skymodel": None,
            "photometry_skymodel": None,
            "normalization_skymodels": None,
            "normalization_reference_frequencies": None,
        },
    }
    for key, value in overrides.items():
        if key in ("cluster_specific", "imaging_specific"):
            parset_dict[key].update(value)
        else:
            parset_dict[key] = value
    return parset_dict


def test_missing_parset_file(parset_scenario):
    parset_scenario.parset.unlink()

    with pytest.raises(FileNotFoundError):
        parset_read(str(parset_scenario.parset))


def test_empty_parset_file(parset_scenario):
    parset_scenario.parset.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match=r"Missing required option\(s\) in section \[global\]:"):
        parset_read(str(parset_scenario.parset))


def test_minimal_parset(parset_scenario):
    parset_dict = parset_read(str(parset_scenario.parset))

    assert parset_dict["dir_working"] == str(parset_scenario.working_dir)
    assert parset_dict["input_ms"] == str(parset_scenario.input_ms)


def test_misspelled_section(parset_scenario, caplog):
    section = "misspelled_section"
    _append_to_parset(parset_scenario.parset, f"\n[{section}]\n")
    caplog.set_level("WARNING", logger="rapthor:parset")

    parset_read(str(parset_scenario.parset))

    assert any(f"Section [{section}] is invalid" in message for message in caplog.messages)


def test_misspelled_option(parset_scenario, caplog):
    option = "misspelled_option"
    _append_to_parset(parset_scenario.parset, f"\n{option} = some value\n")
    caplog.set_level("WARNING", logger="rapthor:parset")

    parset_read(str(parset_scenario.parset))

    assert any(
        f"Option '{option}' in section [global] is invalid" in message
        for message in caplog.messages
    )


def test_deprecated_option(parset_scenario, caplog):
    section = "[cluster]"
    option = "dir_local"
    _append_to_parset(parset_scenario.parset, f"\n{section}\n{option} = some value\n")
    caplog.set_level("WARNING", logger="rapthor:parset")

    parset_read(str(parset_scenario.parset))

    assert any(
        f"Option '{option}' in section {section} is deprecated" in message
        for message in caplog.messages
    )


def test_fraction_out_of_range(parset_scenario):
    option = "selfcal_data_fraction"
    value = 1.1
    _append_to_parset(parset_scenario.parset, f"\n{option} = {value}\n")

    with pytest.raises(
        ValueError,
        match=f"The {option} parameter is {value}; it must be > 0 and <= 1",
    ):
        parset_read(str(parset_scenario.parset))


def test_invalid_idg_mode(parset_scenario):
    option = "idg_mode"
    value = "invalid"
    _append_to_parset(parset_scenario.parset, f"\n[imaging]\n{option} = {value}\n")

    with pytest.raises(ValueError, match=f"The option '{option}' must be one of"):
        parset_read(str(parset_scenario.parset))


def test_filter_skymodel_ncores_must_be_positive(parset_scenario):
    _append_to_parset(
        parset_scenario.parset,
        "\n[cluster]\nfilter_skymodel_ncores = -1\n",
    )

    with pytest.raises(
        ValueError,
        match="filter_skymodel_ncores.*greater than 0",
    ):
        parset_read(str(parset_scenario.parset))


def test_filter_skymodel_ncores_default_is_proposed_production_value(parset_scenario):
    parset = parset_read(str(parset_scenario.parset))

    assert parset["cluster_specific"]["filter_skymodel_ncores"] == 15


def test_filter_skymodel_ncores_zero_uses_max_threads(parset_scenario):
    _append_to_parset(
        parset_scenario.parset,
        "\n[cluster]\nmax_threads = 12\nfilter_skymodel_ncores = 0\n",
    )

    parset = parset_read(str(parset_scenario.parset))

    assert parset["cluster_specific"]["max_threads"] == 12
    assert parset["cluster_specific"]["filter_skymodel_ncores"] == 12


def test_unequal_sector_list_lengths(parset_scenario):
    _append_to_parset(parset_scenario.parset, "\n[imaging]\nsector_center_ra_list = [1]\n")

    with pytest.raises(
        ValueError,
        match="The options .* must all have the same number of entries",
    ):
        parset_read(str(parset_scenario.parset))


def test_default_parset_contents(parset_scenario, monkeypatch):
    monkeypatch.setattr("rapthor.lib.parset.misc.nproc", lambda: 8)
    parset_scenario.parset.write_text(
        _render_resource_template(
            "rapthor_minimal.parset.template",
            dir_working=parset_scenario.working_dir,
            input_ms=parset_scenario.input_ms,
        ),
        encoding="utf-8",
    )

    parset = parset_read(str(parset_scenario.parset))
    ref_parset = ast.literal_eval(
        _render_resource_template(
            "rapthor_minimal.parset_dict.template",
            dir_working=parset_scenario.working_dir,
            input_ms=parset_scenario.input_ms,
        )
    )

    assert parset == ref_parset


def test_complete_parset_contents(parset_scenario, monkeypatch):
    monkeypatch.setattr("rapthor.lib.parset.misc.nproc", lambda: 8)
    parset_scenario.parset.write_text(
        _render_resource_template(
            "rapthor_complete.parset.template",
            dir_working=parset_scenario.working_dir,
            input_ms=parset_scenario.input_ms,
        ),
        encoding="utf-8",
    )

    parset = parset_read(str(parset_scenario.parset))
    ref_parset = ast.literal_eval(
        _render_resource_template(
            "rapthor_complete.parset_dict.template",
            dir_working=parset_scenario.working_dir,
            input_ms=parset_scenario.input_ms,
        )
    )

    assert parset == ref_parset


def test_input_skymodel_not_found_raises():
    parset_dict = _make_skymodel_settings(input_skymodel="/nonexistent/skymodel.txt")

    with pytest.raises(FileNotFoundError):
        check_and_adjust_skymodel_settings(parset_dict)


def test_input_skymodel_exists(tmp_path):
    skymodel = tmp_path / "input.skymodel"
    skymodel.write_text("", encoding="utf-8")
    parset_dict = _make_skymodel_settings(input_skymodel=str(skymodel))

    check_and_adjust_skymodel_settings(parset_dict)


def test_input_skymodel_with_generate_disables_download(tmp_path, caplog):
    skymodel = tmp_path / "input.skymodel"
    skymodel.write_text("", encoding="utf-8")
    parset_dict = _make_skymodel_settings(
        input_skymodel=str(skymodel),
        generate_initial_skymodel=True,
        download_initial_skymodel=True,
    )
    caplog.set_level("WARNING", logger="rapthor:parset")

    check_and_adjust_skymodel_settings(parset_dict)

    assert parset_dict["download_initial_skymodel"] is False
    assert any("Sky model generation requested" in message for message in caplog.messages)


def test_input_skymodel_with_download_disables_download(tmp_path, caplog):
    skymodel = tmp_path / "input.skymodel"
    skymodel.write_text("", encoding="utf-8")
    parset_dict = _make_skymodel_settings(
        input_skymodel=str(skymodel),
        download_initial_skymodel=True,
    )
    caplog.set_level("WARNING", logger="rapthor:parset")

    check_and_adjust_skymodel_settings(parset_dict)

    assert parset_dict["download_initial_skymodel"] is False
    assert any("Sky model download requested" in message for message in caplog.messages)
    assert any("Disabling download" in message for message in caplog.messages)


def test_no_input_generate_requested(caplog):
    parset_dict = _make_skymodel_settings(generate_initial_skymodel=True)
    caplog.set_level("INFO", logger="rapthor:parset")

    check_and_adjust_skymodel_settings(parset_dict)

    assert any("Will automatically generate sky model" in message for message in caplog.messages)


def test_no_input_generate_requested_with_apparent_warns(caplog):
    parset_dict = _make_skymodel_settings(
        generate_initial_skymodel=True,
        apparent_skymodel="some_apparent.skymodel",
    )
    caplog.set_level("WARNING", logger="rapthor:parset")

    check_and_adjust_skymodel_settings(parset_dict)

    assert any("apparent sky model will not be used" in message for message in caplog.messages)


def test_no_input_download_requested(caplog):
    parset_dict = _make_skymodel_settings(download_initial_skymodel=True)
    caplog.set_level("INFO", logger="rapthor:parset")

    check_and_adjust_skymodel_settings(parset_dict)

    assert any("Will automatically download sky model" in message for message in caplog.messages)


def test_no_input_download_requested_with_apparent_warns(caplog):
    parset_dict = _make_skymodel_settings(
        download_initial_skymodel=True,
        apparent_skymodel="some_apparent.skymodel",
    )
    caplog.set_level("WARNING", logger="rapthor:parset")

    check_and_adjust_skymodel_settings(parset_dict)

    assert any("apparent sky model will not be used" in message for message in caplog.messages)


def test_no_input_no_generate_no_download_warns(caplog):
    parset_dict = _make_skymodel_settings()
    caplog.set_level("WARNING", logger="rapthor:parset")

    check_and_adjust_skymodel_settings(parset_dict)

    assert any("neither generation nor download" in message for message in caplog.messages)


def test_download_no_internet_raises():
    parset_dict = _make_skymodel_settings(
        download_initial_skymodel=True,
        cluster_specific={"allow_internet_access": False},
    )

    with pytest.raises(ValueError):
        check_and_adjust_skymodel_settings(parset_dict)


def test_internet_allowed_download_ok(caplog):
    parset_dict = _make_skymodel_settings(
        download_initial_skymodel=True,
        cluster_specific={"allow_internet_access": True},
    )
    caplog.set_level("INFO", logger="rapthor:parset")

    check_and_adjust_skymodel_settings(parset_dict)

    assert any("Will automatically download sky model" in message for message in caplog.messages)


def test_astrometry_skymodel_missing_no_internet_raises():
    parset_dict = _make_skymodel_settings(
        generate_initial_skymodel=True,
        cluster_specific={"allow_internet_access": False},
        imaging_specific={
            "astrometry_skymodel": "/nonexistent/astro.skymodel",
            "photometry_skymodel": None,
            "normalization_skymodels": None,
            "normalization_reference_frequencies": None,
        },
    )

    with pytest.raises(FileNotFoundError):
        check_and_adjust_skymodel_settings(parset_dict)


def test_photometry_skymodel_missing_no_internet_raises():
    parset_dict = _make_skymodel_settings(
        generate_initial_skymodel=True,
        cluster_specific={"allow_internet_access": False},
        imaging_specific={
            "astrometry_skymodel": None,
            "photometry_skymodel": "/nonexistent/photo.skymodel",
            "normalization_skymodels": None,
            "normalization_reference_frequencies": None,
        },
    )

    with pytest.raises(FileNotFoundError):
        check_and_adjust_skymodel_settings(parset_dict)


def test_normalization_skymodel_missing_no_internet_raises():
    parset_dict = _make_skymodel_settings(
        generate_initial_skymodel=True,
        cluster_specific={"allow_internet_access": False},
        imaging_specific={
            "astrometry_skymodel": None,
            "photometry_skymodel": None,
            "normalization_skymodels": [
                "/nonexistent/norm.skymodel",
                "/nonexistent/norm2.skymodel",
            ],
            "normalization_reference_frequencies": None,
        },
    )

    with pytest.raises(FileNotFoundError):
        check_and_adjust_skymodel_settings(parset_dict)


def test_astrometry_skymodel_exists_no_internet_ok(tmp_path, caplog):
    skymodel = tmp_path / "astrometry.skymodel"
    skymodel.write_text("", encoding="utf-8")
    parset_dict = _make_skymodel_settings(
        cluster_specific={"allow_internet_access": False},
        imaging_specific={
            "astrometry_skymodel": str(skymodel),
            "photometry_skymodel": None,
        },
    )
    caplog.set_level("WARNING", logger="rapthor:parset")

    check_and_adjust_skymodel_settings(parset_dict)

    assert any("The photometry check will be skipped" in message for message in caplog.messages)


def test_photometry_skymodel_exists_no_internet_ok(tmp_path, caplog):
    skymodel = tmp_path / "photometry.skymodel"
    skymodel.write_text("", encoding="utf-8")
    parset_dict = _make_skymodel_settings(
        cluster_specific={"allow_internet_access": False},
        imaging_specific={
            "astrometry_skymodel": None,
            "photometry_skymodel": str(skymodel),
        },
    )
    caplog.set_level("WARNING", logger="rapthor:parset")

    check_and_adjust_skymodel_settings(parset_dict)

    assert any("The astrometry check will be skipped" in message for message in caplog.messages)


def test_normalization_skymodel_exists_no_internet_ok(tmp_path, caplog):
    skymodel = tmp_path / "normalization.skymodel"
    skymodel.write_text("", encoding="utf-8")
    parset_dict = _make_skymodel_settings(
        cluster_specific={"allow_internet_access": False},
        imaging_specific={
            "normalization_skymodels": [str(skymodel), str(skymodel)],
            "normalization_reference_frequencies": [str(142000000.0), str(142001000.0)],
        },
    )
    caplog.set_level("WARNING", logger="rapthor:parset")

    check_and_adjust_skymodel_settings(parset_dict)

    assert any("The astrometry check will be skipped" in message for message in caplog.messages)
    assert any("The photometry check will be skipped" in message for message in caplog.messages)


def test_single_normalization_skymodel_raises_error(tmp_path):
    skymodel = tmp_path / "normalization.skymodel"
    skymodel.write_text("", encoding="utf-8")
    parset_dict = _make_skymodel_settings(
        cluster_specific={"allow_internet_access": False},
        imaging_specific={
            "normalization_skymodels": str(skymodel),
            "normalization_reference_frequencies": [str(142000000.0)],
        },
    )

    with pytest.raises(ValueError):
        check_and_adjust_skymodel_settings(parset_dict)


def test_only_one_existing_path_for_normalization_skymodels_raises_error(tmp_path):
    skymodel = tmp_path / "normalization.skymodel"
    skymodel.write_text("", encoding="utf-8")
    parset_dict = _make_skymodel_settings(
        cluster_specific={"allow_internet_access": False},
        imaging_specific={
            "normalization_skymodels": [str(skymodel), "/nonexistent/norm.skymodel"],
            "normalization_reference_frequencies": [str(142000000.0), str(142001000.0)],
        },
    )

    with pytest.raises(FileNotFoundError):
        check_and_adjust_skymodel_settings(parset_dict)


def test_normalization_reference_frequencies_missing_raises_error(tmp_path):
    skymodel = tmp_path / "normalization.skymodel"
    skymodel.write_text("", encoding="utf-8")
    parset_dict = _make_skymodel_settings(
        cluster_specific={"allow_internet_access": False},
        imaging_specific={
            "normalization_skymodels": [str(skymodel), str(skymodel)],
            "normalization_reference_frequencies": None,
        },
    )

    with pytest.raises(ValueError):
        check_and_adjust_skymodel_settings(parset_dict)


def test_normalization_reference_frequencies_wrong_length_raises_error(tmp_path):
    skymodel = tmp_path / "normalization.skymodel"
    skymodel.write_text("", encoding="utf-8")
    parset_dict = _make_skymodel_settings(
        cluster_specific={"allow_internet_access": False},
        imaging_specific={
            "normalization_skymodels": [str(skymodel), str(skymodel)],
            "normalization_reference_frequencies": [str(142000000.0)],
        },
    )

    with pytest.raises(ValueError):
        check_and_adjust_skymodel_settings(parset_dict)


def test_normalization_reference_frequencies_must_be_distinct(tmp_path):
    skymodel = tmp_path / "normalization.skymodel"
    skymodel.write_text("", encoding="utf-8")
    parset_dict = _make_skymodel_settings(
        cluster_specific={"allow_internet_access": False},
        imaging_specific={
            "normalization_skymodels": [str(skymodel), str(skymodel)],
            "normalization_reference_frequencies": [134375000.0, 134375000.0],
        },
    )

    with pytest.raises(ValueError, match="at least two distinct frequencies"):
        check_and_adjust_skymodel_settings(parset_dict)


def test_normalization_parameters_tuple(tmp_path):
    skymodel = tmp_path / "normalization.skymodel"
    skymodel.write_text("", encoding="utf-8")
    parset_dict = _make_skymodel_settings(
        cluster_specific={"allow_internet_access": False},
        imaging_specific={
            "normalization_skymodels": (str(skymodel), str(skymodel)),
            "normalization_reference_frequencies": (142000000.0, 142001000.0),
        },
    )

    check_and_adjust_skymodel_settings(parset_dict)


def test_normalization_parameters_set_raises_error(tmp_path):
    skymodel = tmp_path / "normalization.skymodel"
    skymodel.write_text("", encoding="utf-8")
    parset_dict = _make_skymodel_settings(
        cluster_specific={"allow_internet_access": False},
        imaging_specific={
            "normalization_skymodels": {str(skymodel), str(skymodel)},
            "normalization_reference_frequencies": {142000000.0, 142001000.0},
        },
    )

    with pytest.raises(ValueError):
        check_and_adjust_skymodel_settings(parset_dict)


def test_diagnostic_skymodel_empty_no_internet_ok(caplog):
    parset_dict = _make_skymodel_settings(
        cluster_specific={"allow_internet_access": False},
        imaging_specific={
            "astrometry_skymodel": None,
            "photometry_skymodel": None,
            "normalization_skymodels": None,
            "normalization_reference_frequencies": None,
        },
    )
    caplog.set_level("WARNING", logger="rapthor:parset")

    check_and_adjust_skymodel_settings(parset_dict)

    assert any("The astrometry check will be skipped" in message for message in caplog.messages)
    assert any("The photometry check will be skipped" in message for message in caplog.messages)
