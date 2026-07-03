"""Input preparation helpers for committed benchmark scenarios."""

from __future__ import annotations

import shutil
import tarfile
import tempfile
from pathlib import Path

import requests

TEST_MS_ARCHIVE_URL = "https://support.astron.nl/software/ci_data/rapthor/tDDECal.in_MS.tgz"
TEST_MS_ARCHIVE_DIRNAME = "tDDECal.MS"
TEST_MS_DIRNAME = "test.ms"


def download_test_ms(destination: Path) -> None:
    """Download the shared small Measurement Set used by tests and benchmarks."""
    response = requests.get(TEST_MS_ARCHIVE_URL, timeout=300)
    response.raise_for_status()

    with tempfile.TemporaryDirectory(dir=destination.parent) as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        archive_path = tmp_dir / "test.ms.tgz"
        archive_path.write_bytes(response.content)

        with tarfile.open(archive_path, "r:gz") as archive:
            archive.extractall(tmp_dir)

        extracted_dir = tmp_dir / TEST_MS_ARCHIVE_DIRNAME
        if not extracted_dir.exists():
            raise FileNotFoundError(f"Downloaded archive did not contain {TEST_MS_ARCHIVE_DIRNAME}")

        try:
            shutil.move(extracted_dir.as_posix(), destination.as_posix())
        except shutil.Error:
            if not destination.exists():
                raise


def ensure_test_ms(resource_dir: Path) -> Path:
    """Return the shared test Measurement Set, downloading it if needed."""
    destination = Path(resource_dir) / TEST_MS_DIRNAME
    if destination.exists():
        return destination

    download_test_ms(destination)
    return destination
