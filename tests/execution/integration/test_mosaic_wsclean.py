"""Draw model mosaics with real WSClean from sector sky models."""

import shutil

import numpy as np
import pytest
from astropy.io import fits as pyfits
from astropy.wcs import WCS

from rapthor.execution.config import ExecutionConfig
from rapthor.execution.mosaic.model_rendering import render_model_mosaic_with_wsclean
from rapthor.lib.records import file_record

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(shutil.which("wsclean") is None, reason="WSClean is not available"),
]

FREQUENCY_HZ = 150.0e6
SOURCE_FLUX_JY = 1.5
SKYMODEL_HEADER = (
    "FORMAT = Name, Type, Patch, Ra, Dec, I, SpectralIndex='[]', LogarithmicSI, "
    f"ReferenceFrequency='{FREQUENCY_HZ}', MajorAxis, MinorAxis, Orientation\n"
)
# One point source, 0.1 deg north of the template centre
SKYMODEL_SOURCE = (
    "\n , , Patch_1, 12:00:00.0, +45.00.00.0\n"
    f"src_a, POINT, Patch_1, 12:00:00.0, +45.06.00.0, {SOURCE_FLUX_JY}, [-0.7], true, "
    f"{FREQUENCY_HZ}, , , \n"
)


@pytest.mark.parametrize("has_sources", [[False, True], [False, False]], ids=["mixed", "all-empty"])
def test_wsclean_draws_model_mosaic_from_empty_sector_models(tmp_path, has_sources):
    # WSClean rounds odd image sizes up to even ones, so use an even size
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---SIN", "DEC--SIN"]
    wcs.wcs.cdelt = [-0.01, 0.01]
    wcs.wcs.crval = [180.0, 45.0]
    wcs.wcs.crpix = [33.0, 33.0]
    header = wcs.to_header()
    header["FREQ"] = FREQUENCY_HZ
    template = tmp_path / "mosaic_1_template.fits"
    pyfits.PrimaryHDU(data=np.zeros((64, 64)), header=header).writeto(template)
    sector_skymodels = []
    for index, has_source in enumerate(has_sources, start=1):
        path = tmp_path / f"sector_{index}.true_sky.txt"
        path.write_text(SKYMODEL_HEADER + (SKYMODEL_SOURCE if has_source else ""))
        sector_skymodels.append(str(path))
    output = tmp_path / "mosaic_1-MFS-model-pb.fits"

    # The default single CPU per task checks that two threads are still used:
    # with one, WSClean would draw nothing
    result = render_model_mosaic_with_wsclean(
        sector_skymodels,
        str(template),
        str(output),
        str(tmp_path),
        ExecutionConfig(task_runner="sync"),
    )

    assert result == file_record(output)
    with pyfits.open(output) as hdul:
        data = np.squeeze(hdul[0].data)
        output_wcs = WCS(hdul[0].header).celestial
    assert data.shape == (64, 64)
    if not any(has_sources):
        assert np.count_nonzero(data) == 0
        return
    # An off-pixel source is spread over nearby pixels, so the sum of a small
    # image differs slightly from the component flux
    assert np.isclose(data.sum(), SOURCE_FLUX_JY, rtol=0.05)
    peak_y, peak_x = np.unravel_index(np.argmax(data), data.shape)
    expected_x, expected_y = output_wcs.world_to_pixel_values(180.0, 45.1)
    assert abs(peak_x - expected_x) <= 1
    assert abs(peak_y - expected_y) <= 1
