"""Tests for preparing sky models for a run without internet access."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.table import Table

import rapthor.execution.image.catalog_preparation as catalog_preparation
from rapthor.execution.image.catalog_cli import main
from rapthor.execution.image.catalog_preparation import parset_options, prepare_catalogs
from rapthor.execution.image.diagnostic_calculation import check_astrometry
from rapthor.execution.image.flux_normalization import get_field_phase_center

ALL_SURVEYS = ["TGSS", "PANSTARRS", "VLSSR", "WENSS"]


class FakeDownload:
    """Stand in for the LSMTool download, writing a file for each survey it is asked for."""

    def __init__(self, failing=()):
        self.calls = []
        self.failing = failing

    def __call__(self, cone_params, skymodel_path, overwrite, survey):
        self.calls.append(
            {
                "cone": cone_params,
                "survey": survey,
                "overwrite": overwrite,
                "path": skymodel_path,
            }
        )
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


def test_each_download_writes_to_its_own_temporary_file(measurement_set, tmp_path, fake_download):
    output_dir = tmp_path / "catalogs"

    # Runs that overlap would otherwise write to, move or remove the same file
    prepare_catalogs(measurement_set, str(output_dir))
    prepare_catalogs(measurement_set, str(output_dir), overwrite=True)

    tgss_paths = [Path(call["path"]) for call in fake_download.calls if call["survey"] == "TGSS"]
    assert len(tgss_paths) == 2
    assert tgss_paths[0] != tgss_paths[1]
    final_path = output_dir / "TGSS_model.txt"
    assert final_path not in tgss_paths
    # On the same file system as the sky model, so that moving it into place is atomic
    assert all(path.is_relative_to(output_dir) for path in tgss_paths)


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


def test_astrometry_check_uses_a_downloaded_sky_model(
    measurement_set, tmp_path, monkeypatch, mocker
):
    """The sky model from a download, in which LSMTool groups the sources, can be compared."""
    ra, dec = 10.0, 50.0
    offsets_deg = np.linspace(-0.05, 0.05, 10)
    dec_shift_deg = 1 / 3600

    def write_survey_model(_cone_params, _survey, skymodel_path):
        lines = [
            f"s{index}, POINT, {ra + offset}, {dec + offset + dec_shift_deg}, 1.0"
            for index, offset in enumerate(offsets_deg)
        ]
        Path(skymodel_path).write_text(
            "\n".join(["FORMAT = Name, Type, Ra, Dec, I, ReferenceFrequency=150e6", *lines])
        )

    monkeypatch.setattr(
        "lsmtool.download_skymodel.download_skymodel_from_survey", write_survey_model
    )
    catalogs = prepare_catalogs(measurement_set, str(tmp_path / "catalogs"))
    assert catalogs.failures == {}
    input_catalog = tmp_path / "catalog.fits"
    Table(
        {
            "Source_id": np.arange(len(offsets_deg)),
            "RA": ra + offsets_deg,
            "DEC": dec + offsets_deg,
            "Isl_Total_flux": np.ones(len(offsets_deg)),
            "DC_Maj": np.full(len(offsets_deg), 1 / 3600),
            "E_RA": np.full(len(offsets_deg), 0.1 / 3600),
            "E_DEC": np.full(len(offsets_deg), 0.1 / 3600),
        }
    ).write(input_catalog, format="fits")
    mocker.patch("rapthor.execution.image.diagnostic_calculation.plot_astrometry_offsets")
    output_root = tmp_path / "check"
    # The comparison writes its statistics to the current directory
    monkeypatch.chdir(tmp_path)

    result = check_astrometry(
        SimpleNamespace(ra=ra, dec=dec),
        input_catalog,
        SimpleNamespace(freq=150e6, img_data=np.zeros((100, 100)), img_hdr={"CDELT1": 0.01}),
        None,
        min_number=5,
        output_root=output_root,
        comparison_skymodel=catalogs.paths["PANSTARRS"],
        allow_internet_access=False,
    )

    assert abs(result["meanDecOffsetDeg"]) == pytest.approx(dec_shift_deg, rel=0.01)
    assert result["meanRAOffsetDeg"] == pytest.approx(0.0, abs=dec_shift_deg / 10)
    assert output_root.with_suffix(".astrometry_offsets.json").exists()


def test_command_is_installed():
    pyproject = (Path(__file__).parents[2] / "pyproject.toml").read_text(encoding="utf-8")

    assert 'fetch_skymodels = "rapthor.execution.image.catalog_cli:main"' in pyproject
