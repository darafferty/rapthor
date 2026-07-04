import importlib.util
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
    report = {"run_root": str(tmp_path)}

    markdown = module._render_markdown_report(report, [result])

    assert "## FITS Residual Metrics" in markdown
    assert "| `synthetic` | `current.fits` |" in markdown
