import configparser
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

SCRIPT_PATH = Path(__file__).parents[2] / "scripts" / "dev" / "run_saved_cwl_equivalence.py"


def load_equivalence_script():
    spec = importlib.util.spec_from_file_location("run_saved_cwl_equivalence", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _header(**overrides):
    header = fits.Header()
    header["CTYPE1"] = "RA---SIN"
    header["CTYPE2"] = "DEC--SIN"
    header["CUNIT1"] = "deg"
    header["CUNIT2"] = "deg"
    header["CRVAL1"] = 10.0
    header["CRVAL2"] = 20.0
    header["CRPIX1"] = 1.0
    header["CRPIX2"] = 1.0
    header["CDELT1"] = -0.01
    header["CDELT2"] = 0.01
    header.update(overrides)
    return header


def _write_fits(path, data, header=None):
    fits.PrimaryHDU(data=np.asarray(data, dtype=np.float32), header=header or _header()).writeto(
        path, overwrite=True
    )


def test_saved_normalization_reference_frequencies_are_valid_for_current_contract():
    module = load_equivalence_script()
    parser = configparser.ConfigParser()
    parser["imaging"] = {
        "normalization_reference_frequencies": "[150000000.0, 150000000.0]",
    }

    module._set_saved_normalization_reference_frequencies(parser)

    assert parser["imaging"]["normalization_reference_frequencies"] == (
        "[142000000.0, 142001000.0]"
    )


def _compare_fits(module, tmp_path, reference_data, current_data, *, current_header=None):
    reference = tmp_path / "reference.fits"
    current = tmp_path / "current.fits"
    _write_fits(reference, reference_data)
    _write_fits(current, current_data, header=current_header)
    result = module.ComparisonResult("synthetic", tmp_path / "run")

    module._compare_fits(reference, current, result, atol=1e-6, rtol=1e-3)

    return result


def test_fits_equivalence_records_residual_metrics_for_small_drift(tmp_path):
    module = load_equivalence_script()
    reference = np.arange(9, dtype=float).reshape(3, 3) + 1.0
    current = reference + 1e-5

    result = _compare_fits(module, tmp_path, reference, current)

    assert result.failures == []
    metric = result.product_statistics["fits"][0]
    assert metric["kind"] == "image"
    assert metric["shape"] == [3, 3]
    assert metric["max_abs_delta"] == pytest.approx(1e-5, rel=1e-2)
    assert metric["p99_abs_delta"] == pytest.approx(1e-5, rel=1e-2)
    assert metric["residual_rms_over_reference_rms"] < 1e-5


def test_fits_equivalence_catches_spatial_change_with_matching_aggregate_stats(tmp_path):
    module = load_equivalence_script()
    reference = np.array([[0.0, 1.0], [2.0, 3.0]])
    current = np.array([[3.0, 2.0], [1.0, 0.0]])

    result = _compare_fits(module, tmp_path, reference, current)

    assert any("FITS image pixels differ" in failure for failure in result.failures)
    assert not any("FITS mean differs" in failure for failure in result.failures)
    assert result.product_statistics["fits"][0]["max_abs_delta"] == pytest.approx(3.0)


def test_fits_equivalence_allows_sparse_near_zero_pixel_outliers(tmp_path):
    module = load_equivalence_script()
    reference = np.zeros((100, 100), dtype=float)
    reference[0, 0] = 1.0
    current = reference.copy()
    current[37, 42] = 5e-6

    result = _compare_fits(module, tmp_path, reference, current)

    assert result.failures == []
    metric = result.product_statistics["fits"][0]
    assert metric["max_abs_delta"] == pytest.approx(5e-6)
    assert metric["p99_abs_delta"] == pytest.approx(0.0)


def test_fits_equivalence_allows_small_relative_image_jitter(tmp_path):
    module = load_equivalence_script()
    reference = np.linspace(-0.5, 0.5, 10_000, dtype=float).reshape(100, 100)
    jitter = np.sin(np.arange(reference.size, dtype=float)).reshape(reference.shape) * 2.0e-6
    jitter[::17, ::19] += 1.0e-5
    jitter -= np.mean(jitter)
    current = reference + jitter

    result = _compare_fits(module, tmp_path, reference, current)

    assert result.failures == []
    metric = result.product_statistics["fits"][0]
    assert metric["residual_rms_over_reference_rms"] < 1.0e-5
    assert metric["p99_abs_delta"] > 1.0e-6


def test_fits_equivalence_catches_systematic_near_zero_pixel_drift(tmp_path):
    module = load_equivalence_script()
    reference = np.zeros((100, 100), dtype=float)
    current = np.full((100, 100), 2e-6, dtype=float)

    result = _compare_fits(module, tmp_path, reference, current)

    assert any("FITS image pixels differ" in failure for failure in result.failures)


def test_fits_equivalence_catches_wcs_header_changes(tmp_path):
    module = load_equivalence_script()
    data = np.ones((2, 2), dtype=float)

    result = _compare_fits(module, tmp_path, data, data, current_header=_header(CRVAL1=10.5))

    assert any("FITS WCS/header key CRVAL1 differs" in failure for failure in result.failures)


def test_fits_equivalence_catches_nan_mask_changes(tmp_path):
    module = load_equivalence_script()
    reference = np.array([[np.nan, 1.0], [2.0, 3.0]])
    current = np.array([[0.0, 1.0], [2.0, 3.0]])

    result = _compare_fits(module, tmp_path, reference, current)

    assert any("FITS finite/NaN mask differs" in failure for failure in result.failures)


def test_fits_equivalence_records_cube_plane_metrics(tmp_path):
    module = load_equivalence_script()
    reference = np.ones((2, 2, 2), dtype=float)
    current = reference.copy()
    current[1, 0, 0] += 5e-7

    result = _compare_fits(module, tmp_path, reference, current)

    assert result.failures == []
    planes = result.product_statistics["fits"][0]["planes"]
    assert [plane["leading_index"] for plane in planes] == [[0], [1]]
    assert planes[0]["max_abs_delta"] == pytest.approx(0.0)
    assert planes[1]["max_abs_delta"] == pytest.approx(5e-7, rel=0.3)


def test_equivalence_markdown_report_includes_fits_residual_table(tmp_path):
    module = load_equivalence_script()
    result = _compare_fits(
        module,
        tmp_path,
        np.ones((2, 2), dtype=float),
        np.ones((2, 2), dtype=float) + 1e-5,
    )
    result.warnings.append("output-record optional artifact basenames differ for image_1")
    report = {"run_root": str(tmp_path)}

    markdown = module._render_markdown_report(report, [result])

    assert "## FITS Residual Metrics" in markdown
    assert "| `synthetic` | `current.fits` |" in markdown
    assert "## Warnings" in markdown
    assert "- `synthetic`: output-record optional artifact basenames differ for image_1" in markdown


def _compare_text_product(module, tmp_path, reference_text, current_text, filename="facets.reg"):
    reference = tmp_path / "reference" / filename
    current = tmp_path / "current" / filename
    reference.parent.mkdir()
    current.parent.mkdir()
    reference.write_text(reference_text.strip() + "\n", encoding="utf-8")
    current.write_text(current_text.strip() + "\n", encoding="utf-8")
    result = module.ComparisonResult("synthetic", tmp_path / "run")

    module._compare_text_product(reference, current, result, atol=1e-6, rtol=1e-3)

    return result


def test_ds9_region_equivalence_ignores_label_placement(tmp_path):
    module = load_equivalence_script()
    reference = """
    # Region file format: DS9 version 4.0
    global color=green
    fk5
    polygon(1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 1.0, 4.0)
    point(2.0, 3.0) # text={Patch_A}
    """
    current = """
    # Region file format: DS9 version 4.0
    fk5
    polygon(1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 1.0, 4.0) # text={Patch_A}
    """

    result = _compare_text_product(module, tmp_path, reference, current)

    assert result.failures == []
    assert result.product_statistics["text"] == [
        {
            "product": "facets.reg",
            "kind": "ds9_region",
            "reference_shapes": 1,
            "current_shapes": 1,
            "reference_labels": 1,
            "current_labels": 1,
        }
    ]


def test_ds9_region_equivalence_catches_geometry_changes(tmp_path):
    module = load_equivalence_script()
    reference = "fk5\npolygon(1.0, 2.0, 3.0, 2.0, 3.0, 4.0) # text={Patch_A}"
    current = "fk5\npolygon(1.0, 2.0, 3.0, 2.0, 3.1, 4.0) # text={Patch_A}"

    result = _compare_text_product(module, tmp_path, reference, current)

    assert result.failures == ["DS9 region geometry differs for facets.reg"]


def test_ds9_region_equivalence_catches_label_changes(tmp_path):
    module = load_equivalence_script()
    reference = "fk5\npolygon(1.0, 2.0, 3.0, 2.0, 3.0, 4.0) # text={Patch_A}"
    current = "fk5\npolygon(1.0, 2.0, 3.0, 2.0, 3.0, 4.0) # text={Patch_B}"

    result = _compare_text_product(module, tmp_path, reference, current)

    assert result.failures == ["DS9 region labels differ for facets.reg"]


def _write_output_record(root, operation, data):
    output_dir = root / "pipelines" / operation
    output_dir.mkdir(parents=True)
    (output_dir / ".outputs.json").write_text(json.dumps(data), encoding="utf-8")


def test_output_record_comparison_warns_for_metadata_shape_only(tmp_path):
    module = load_equivalence_script()
    reference = tmp_path / "reference"
    current = tmp_path / "current"
    _write_output_record(
        reference,
        "image_1",
        {"image": {"class": "File", "basename": "field-MFS-image.fits"}},
    )
    _write_output_record(
        current,
        "image_1",
        {"image": [{"class": "File", "basename": "field-MFS-image.fits"}]},
    )
    result = module.ComparisonResult("synthetic", tmp_path / "run")

    module._compare_output_records(reference, current, result)

    assert result.failures == []
    assert result.warnings == ["output-record metadata shape differs for image_1"]
    assert result.product_statistics["output_records"] == [
        {
            "operation": "image_1",
            "kind": "metadata_shape",
            "reference_count": 1,
            "current_count": 1,
            "missing_basenames": [],
            "extra_basenames": [],
        }
    ]


def test_output_record_comparison_fails_for_product_basename_changes(tmp_path):
    module = load_equivalence_script()
    reference = tmp_path / "reference"
    current = tmp_path / "current"
    _write_output_record(
        reference,
        "image_1",
        {"image": {"class": "File", "basename": "field-MFS-image.fits"}},
    )
    _write_output_record(
        current,
        "image_1",
        {"image": {"class": "File", "basename": "field-MFS-residual.fits"}},
    )
    result = module.ComparisonResult("synthetic", tmp_path / "run")

    module._compare_output_records(reference, current, result)

    assert result.warnings == []
    assert result.failures == ["output-record product basenames differ for image_1"]
    assert result.product_statistics["output_records"][0]["missing_basenames"] == [
        "field-MFS-image.fits"
    ]
    assert result.product_statistics["output_records"][0]["extra_basenames"] == [
        "field-MFS-residual.fits"
    ]


@pytest.mark.parametrize(
    ("reference_name", "current_name"),
    [
        ("fast_phase_dir[Patch].png", "scalarphase_dir[Patch].png"),
        ("fulljones_gains.h5", "fulljones_solutions.h5"),
    ],
)
def test_output_record_comparison_warns_for_auxiliary_artifact_name_changes(
    tmp_path,
    reference_name,
    current_name,
):
    module = load_equivalence_script()
    reference = tmp_path / "reference"
    current = tmp_path / "current"
    _write_output_record(
        reference,
        "calibrate_1",
        {"plot": {"class": "File", "basename": reference_name}},
    )
    _write_output_record(
        current,
        "calibrate_1",
        {"plot": {"class": "File", "basename": current_name}},
    )
    result = module.ComparisonResult("synthetic", tmp_path / "run")

    module._compare_output_records(reference, current, result)

    assert result.failures == []
    assert result.warnings == ["output-record auxiliary artifact basenames differ for calibrate_1"]
    assert result.product_statistics["output_records"][0]["kind"] == (
        "auxiliary_artifact_basenames"
    )


def test_output_record_comparison_warns_for_optional_artifact_drift(tmp_path):
    module = load_equivalence_script()
    reference = tmp_path / "reference"
    current = tmp_path / "current"
    _write_output_record(
        reference,
        "image_1",
        {
            "image": {"class": "File", "basename": "sector_1-MFS-image-pb.fits"},
            "visibilities": {"class": "Directory", "basename": "test.ms.sector_1.prep"},
        },
    )
    _write_output_record(
        current,
        "image_1",
        {
            "image": [
                {"class": "File", "basename": "sector_1-MFS-image-pb.fits"},
                {"class": "File", "basename": "sector_1-MFS-image-pb-ast.fits"},
            ],
            "visibilities": {"class": "Directory", "basename": "test.sector_1_prep.ms"},
        },
    )
    result = module.ComparisonResult("synthetic", tmp_path / "run")

    module._compare_output_records(reference, current, result)

    assert result.failures == []
    assert result.warnings == ["output-record optional artifact basenames differ for image_1"]
    assert result.product_statistics["output_records"] == [
        {
            "operation": "image_1",
            "kind": "optional_artifact_basenames",
            "reference_count": 2,
            "current_count": 3,
            "missing_basenames": ["test.ms.sector_1.prep"],
            "extra_basenames": [
                "sector_1-MFS-image-pb-ast.fits",
                "test.sector_1_prep.ms",
            ],
        }
    ]


def test_reference_scenarios_skip_stale_fulljones_reference_by_default(tmp_path, monkeypatch):
    module = load_equivalence_script()
    for name in ["dd_only_calibration", "di_full_jones_calibration"]:
        (tmp_path / name).mkdir()
    monkeypatch.setattr(module, "REFERENCE_ROOT", tmp_path)

    default_scenarios = module._reference_scenarios([], include_stale=False)
    explicit_stale_scenarios = module._reference_scenarios(
        ["di_full_jones_calibration"],
        include_stale=False,
    )

    assert [path.name for path in default_scenarios] == ["dd_only_calibration"]
    assert [path.name for path in explicit_stale_scenarios] == ["di_full_jones_calibration"]
