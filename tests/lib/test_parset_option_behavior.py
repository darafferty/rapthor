import logging

import pytest

from rapthor.lib.parset import parset_read


def _write_parset(tmp_path, sections):
    input_ms = tmp_path / "input.ms"
    input_ms.mkdir()

    merged_sections = {
        "global": {
            "input_ms": input_ms,
            "dir_working": tmp_path / "work",
        }
    }
    for section, options in sections.items():
        merged_sections.setdefault(section, {}).update(options)

    parset_path = tmp_path / "input.parset"
    lines = []
    for section, options in merged_sections.items():
        lines.append(f"[{section}]")
        for option, value in options.items():
            lines.append(f"{option} = {value}")
        lines.append("")
    parset_path.write_text("\n".join(lines), encoding="utf-8")
    return parset_path


def _read_parset(tmp_path, sections):
    return parset_read(_write_parset(tmp_path, sections), use_log_file=False)


def _section_value(parset, section, option):
    if section == "global":
        return parset[option]
    return parset[f"{section}_specific"][option]


@pytest.mark.parametrize(
    ("section", "option", "value", "expected"),
    [
        pytest.param(
            "global",
            "separation_tolerance_arcsec",
            "0.2",
            0.2,
            id="pointing-separation-tolerance",
        ),
        pytest.param(
            "global",
            "download_initial_skymodel_radius",
            "3.5",
            3.5,
            id="download-radius",
        ),
        pytest.param(
            "global",
            "download_initial_skymodel_server",
            "LOTSS",
            "LOTSS",
            id="download-server",
        ),
        pytest.param(
            "global",
            "download_overwrite_skymodel",
            "True",
            True,
            id="download-overwrite",
        ),
        pytest.param(
            "global",
            "input_fulljones_h5parm",
            "/data/fulljones.h5",
            "/data/fulljones.h5",
            id="input-fulljones-h5parm",
        ),
        pytest.param(
            "global",
            "input_normalization_h5parm",
            "/data/normalization.h5",
            "/data/normalization.h5",
            id="input-normalization-h5parm",
        ),
        pytest.param(
            "calibration",
            "use_included_skymodels",
            "True",
            True,
            id="included-skymodels",
        ),
        pytest.param(
            "calibration",
            "fulljones_smoothnessconstraint",
            "1.5",
            1.5,
            id="fulljones-smoothness",
        ),
    ],
)
def test_less_common_global_and_calibration_options_are_parsed(
    tmp_path, section, option, value, expected
):
    parset = _read_parset(tmp_path, {section: {option: value}})

    assert _section_value(parset, section, option) == expected


def test_custom_imaging_grid_options_are_parsed(tmp_path):
    parset = _read_parset(
        tmp_path,
        {
            "imaging": {
                "mem_gb": "64.0",
                "grid_center_ra": "14h41m01.884",
                "grid_center_dec": "+35d30m31.52",
                "grid_nsectors_ra": "3",
                "skip_corner_sectors": "True",
            }
        },
    )

    imaging = parset["imaging_specific"]
    assert imaging["mem_gb"] == 64.0
    assert imaging["grid_center_ra"] == pytest.approx(220.25785)
    assert imaging["grid_center_dec"] == pytest.approx(35.50875555555555)
    assert imaging["grid_nsectors_ra"] == 3
    assert imaging["skip_corner_sectors"] is True


def test_custom_sector_geometry_options_are_parsed(tmp_path):
    parset = _read_parset(
        tmp_path,
        {
            "imaging": {
                "sector_center_ra_list": "[14h41m01.884, 14h13m23.234]",
                "sector_center_dec_list": "[+35d30m31.52, +37d21m56.86]",
                "sector_width_ra_deg_list": "[0.532, 0.127]",
                "sector_width_dec_deg_list": "[0.533, 0.128]",
            }
        },
    )

    imaging = parset["imaging_specific"]
    assert imaging["sector_center_ra_list"] == pytest.approx([220.25785, 213.3468083333333])
    assert imaging["sector_center_dec_list"] == pytest.approx(
        [35.50875555555555, 37.36579444444445]
    )
    assert imaging["sector_width_ra_deg_list"] == [0.532, 0.127]
    assert imaging["sector_width_dec_deg_list"] == [0.533, 0.128]


def test_mismatched_time_frequency_smearing_options_warn(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="rapthor:parset"):
        parset = _read_parset(
            tmp_path,
            {
                "calibration": {"correct_time_frequency_smearing": "True"},
                "imaging": {"correct_time_frequency_smearing": "False"},
            },
        )

    assert parset["calibration_specific"]["correct_time_frequency_smearing"] is True
    assert parset["imaging_specific"]["correct_time_frequency_smearing"] is False
    assert "Correction for time and frequency smearing is enabled" in caplog.text
