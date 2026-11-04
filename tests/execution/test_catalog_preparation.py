"""Tests for preparing sky models for a run without internet access."""

from pathlib import Path

import numpy as np
import pytest

import rapthor.execution.image.catalog_preparation as catalog_preparation
from rapthor.execution.image.catalog_cli import main
from rapthor.execution.image.catalog_preparation import parset_options, prepare_catalogs
from rapthor.execution.image.flux_normalization import get_field_phase_center

ALL_SURVEYS = ["TGSS", "PANSTARRS", "VLSSR", "WENSS"]


class FakeDownload:
    """Stand in for the LSMTool download, writing a file for each survey it is asked for."""

    def __init__(self, failing=()):
        self.calls = []
        self.failing = failing

    def __call__(self, cone_params, skymodel_path, overwrite, survey):
        self.calls.append({"cone": cone_params, "survey": survey, "overwrite": overwrite})
        Path(skymodel_path).write_text("partial" if survey in self.failing else survey)
        if survey in self.failing:
            raise OSError(f"{survey} server is down")


@pytest.fixture
def fake_download(monkeypatch):
    download = FakeDownload()
    monkeypatch.setattr(catalog_preparation, "download_skymodel", download)
    return download


@pytest.fixture
def measurement_set(tmp_path, monkeypatch):
    path = tmp_path / "input.ms"
    path.mkdir()
    monkeypatch.setattr(
        catalog_preparation,
        "get_field_phase_center",
        lambda ms_file: (np.radians(-10.0), np.radians(50.0)),
    )
    return str(path)


def test_prepare_catalogs_downloads_every_survey_around_the_pointing(
    measurement_set, tmp_path, fake_download
):
    output_dir = tmp_path / "catalogs"

    catalogs = prepare_catalogs(measurement_set, str(output_dir), radius_deg=3.5)

    assert [call["survey"] for call in fake_download.calls] == ALL_SURVEYS
    for call in fake_download.calls:
        assert call["cone"]["ra"] == pytest.approx(350.0)
        assert call["cone"]["dec"] == pytest.approx(50.0)
        assert call["cone"]["radius"] == 3.5
    assert catalogs.failures == {}
    assert catalogs.paths == {
        survey: str(output_dir / f"{survey}_model.txt") for survey in ALL_SURVEYS
    }
    for survey, path in catalogs.paths.items():
        assert Path(path).read_text() == survey
    assert sorted(path.name for path in output_dir.iterdir()) == sorted(
        f"{survey}_model.txt" for survey in ALL_SURVEYS
    )


def test_prepare_catalogs_reads_the_pointing_of_the_measurement_set(tmp_path, test_ms, monkeypatch):
    download = FakeDownload()
    monkeypatch.setattr(catalog_preparation, "download_skymodel", download)
    ra_rad, dec_rad = get_field_phase_center(test_ms)

    prepare_catalogs(test_ms, str(tmp_path / "catalogs"))

    assert download.calls[0]["cone"]["ra"] == pytest.approx(np.degrees(ra_rad))
    assert download.calls[0]["cone"]["dec"] == pytest.approx(np.degrees(dec_rad))
    assert download.calls[0]["cone"]["radius"] == 5.0


def test_prepare_catalogs_keeps_what_is_present_unless_told_to_overwrite(
    measurement_set, tmp_path, fake_download
):
    output_dir = tmp_path / "catalogs"
    output_dir.mkdir()
    tgss = output_dir / "TGSS_model.txt"
    tgss.write_text("earlier download")

    catalogs = prepare_catalogs(measurement_set, str(output_dir))

    assert [call["survey"] for call in fake_download.calls] == ALL_SURVEYS[1:]
    assert catalogs.paths["TGSS"] == str(tgss)
    assert tgss.read_text() == "earlier download"

    prepare_catalogs(measurement_set, str(output_dir), overwrite=True)

    assert tgss.read_text() == "TGSS"


def test_a_failed_download_leaves_no_file_and_does_not_stop_the_others(
    measurement_set, tmp_path, monkeypatch
):
    monkeypatch.setattr(
        catalog_preparation, "download_skymodel", FakeDownload(failing=("PANSTARRS",))
    )
    output_dir = tmp_path / "catalogs"

    catalogs = prepare_catalogs(measurement_set, str(output_dir))

    assert catalogs.failures == {"PANSTARRS": "PANSTARRS server is down"}
    assert sorted(catalogs.paths) == ["TGSS", "VLSSR", "WENSS"]
    assert sorted(path.name for path in output_dir.iterdir()) == [
        "TGSS_model.txt",
        "VLSSR_model.txt",
        "WENSS_model.txt",
    ]


def test_a_failed_overwrite_keeps_the_earlier_file(measurement_set, tmp_path, monkeypatch):
    monkeypatch.setattr(catalog_preparation, "download_skymodel", FakeDownload(failing=("TGSS",)))
    output_dir = tmp_path / "catalogs"
    output_dir.mkdir()
    tgss = output_dir / "TGSS_model.txt"
    tgss.write_text("earlier download")

    catalogs = prepare_catalogs(measurement_set, str(output_dir), overwrite=True)

    assert "TGSS" in catalogs.failures
    assert tgss.read_text() == "earlier download"


@pytest.mark.parametrize("radius", [0.0, -1.0])
def test_prepare_catalogs_rejects_a_radius_that_is_not_positive(
    measurement_set, tmp_path, fake_download, radius
):
    with pytest.raises(ValueError, match="radius must be positive"):
        prepare_catalogs(measurement_set, str(tmp_path / "catalogs"), radius_deg=radius)
    assert fake_download.calls == []


def test_prepare_catalogs_rejects_a_missing_measurement_set(tmp_path, fake_download):
    with pytest.raises(FileNotFoundError, match="missing.ms"):
        prepare_catalogs(str(tmp_path / "missing.ms"), str(tmp_path / "catalogs"))
    assert fake_download.calls == []


def test_parset_options_name_the_sky_models_and_their_frequencies(
    measurement_set, tmp_path, fake_download
):
    output_dir = tmp_path / "catalogs"

    options = parset_options(prepare_catalogs(measurement_set, str(output_dir)))

    assert options.splitlines() == [
        "[imaging]",
        f"photometry_skymodel = {output_dir}/TGSS_model.txt",
        f"astrometry_skymodel = {output_dir}/PANSTARRS_model.txt",
        f"normalization_skymodels = [{output_dir}/VLSSR_model.txt, {output_dir}/WENSS_model.txt]",
        "normalization_reference_frequencies = [74000000.0, 327000000.0]",
        "",
        "[cluster]",
        "allow_internet_access = False",
    ]


def test_main_prints_the_parset_options(measurement_set, tmp_path, fake_download, capsys):
    output_dir = tmp_path / "catalogs"

    assert main([measurement_set, str(output_dir), "--radius", "2"]) == 0

    assert fake_download.calls[0]["cone"]["radius"] == 2.0
    assert f"photometry_skymodel = {output_dir}/TGSS_model.txt" in capsys.readouterr().out


def test_main_reports_failed_downloads_without_parset_options(
    measurement_set, tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(catalog_preparation, "download_skymodel", FakeDownload(failing=("WENSS",)))

    assert main([measurement_set, str(tmp_path / "catalogs")]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "could not download WENSS" in captured.err


def test_main_reports_a_missing_measurement_set(tmp_path, fake_download):
    with pytest.raises(SystemExit, match="Measurement Set not found"):
        main([str(tmp_path / "missing.ms"), str(tmp_path / "catalogs")])


def test_command_is_installed():
    pyproject = (Path(__file__).parents[2] / "pyproject.toml").read_text(encoding="utf-8")

    assert 'fetch_skymodels = "rapthor.execution.image.catalog_cli:main"' in pyproject
