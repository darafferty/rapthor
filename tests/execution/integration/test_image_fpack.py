"""Real fpack checks for restarting image compression."""

import shutil

import numpy as np
import pytest
from astropy.io import fits

from rapthor.execution.config import ExecutionConfig
from rapthor.execution.image.outputs import compress_image_records
from rapthor.lib.records import file_record

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(shutil.which("fpack") is None, reason="fpack is not available"),
]


def test_compression_restart_replaces_a_stopped_output(tmp_path):
    images = [tmp_path / "sector_1-MFS-I-image.fits", tmp_path / "sector_1-MFS-I-model-pb.fits"]
    for image in images:
        fits.PrimaryHDU(np.random.default_rng(1).random((64, 64), dtype=np.float32)).writeto(image)
    # fpack was stopped while writing the first compressed image.
    (tmp_path / "sector_1-MFS-I-image.fits.fz").write_bytes(b"SIMPLE")

    def compress():
        return compress_image_records(
            [file_record(images[0])],
            [file_record(images[1])],
            str(tmp_path),
            ExecutionConfig(task_runner="sync"),
        )

    records = compress()
    assert records == (
        [file_record(f"{images[0]}.fz")],
        [file_record(f"{images[1]}.fz")],
    )
    for image in images:
        with fits.open(f"{image}.fz") as compressed, fits.open(image) as original:
            assert compressed[1].data.shape == original[0].data.shape
            # Only the model image is compressed without loss
            exact = np.array_equal(compressed[1].data, original[0].data)
            assert exact is image.name.endswith("model-pb.fits")

    # A resumed run compresses the complete images again
    assert compress() == records
