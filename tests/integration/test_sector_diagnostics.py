"""Integration tests for the shared_facet_rw feature."""

import json
import subprocess
from pathlib import Path

import pytest

from .utils import get_working_dir_from_parset, update_parset_path


@pytest.mark.integration
@pytest.mark.parametrize(
    "generated_parset_path",
    [
        (
            "tests/resources/integration_template.parset",
            "tests/resources/integration_true_sky.txt",
            "tests/resources/integration_apparent_sky.txt",
        )
    ],
    indirect=True,
)
def test_rapthor_generates_diagnostics_per_facets(generated_parset_path, single_loop_strategy_path):
    """Test a single self-calibration loop end to end."""

    updated_parset_path = update_parset_path(
        generated_parset_path,
        {
            "allow_internet_access": "False",
            "strategy": str(single_loop_strategy_path),
        },
    )
    print(
        "-" * 80,
        "Rapthor will be executed on: ",
        get_working_dir_from_parset(updated_parset_path),
        "-" * 80,
    )

    command = ["rapthor", str(updated_parset_path)]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    output = f"{result.stdout}\n{result.stderr}"

    assert result.returncode == 0, f"Rapthor failed with output:\n{output}"
    assert "Operation calibrate_1 completed" in output
    assert "Operation image_1 completed" in output
    assert "Operation mosaic_1 completed" in output
    assert "Rapthor has finished :)" in output
    diagnostics_file = (
        Path(get_working_dir_from_parset(updated_parset_path))
        / "plots"
        / "image_1"
        / "sector_1.image_diagnostics.json"
    )

    assert diagnostics_file.exists()
    with diagnostics_file.open() as f_stream:
        diagnostics = json.load(f_stream)
        assert "facets_rms" in diagnostics
        assert "Patch_0" in diagnostics["facets_rms"]
        for type in ("flat_noise", "beam_corrected"):
            assert type in diagnostics["facets_rms"]["Patch_0"]
            for metric in ("mean", "median", "std", "min", "max"):
                assert metric in diagnostics["facets_rms"]["Patch_0"][type]
