"""Integration tests for the Rapthor pipeline restart behavior."""

import os

import pytest
import subprocess
from pathlib import Path

from .utils import (
    get_working_dir_from_parset,
    update_parset_path,
    get_wsclean_output_mtimes,
    make_failing_filter_skymodel,
)


@pytest.fixture()
def injected_failing_filterskymodel_env(tmp_path):
    fake_bin_dir = tmp_path / "fake_bin"
    fake_bin_dir.mkdir()
    make_failing_filter_skymodel(fake_bin_dir)

    modified_env = os.environ.copy()
    modified_env["PATH"] = f"{fake_bin_dir}:{modified_env['PATH']}"
    return modified_env


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
def test_rapthor_restart_after_filter_failure_skips_wsclean(
    generated_parset_path,
    single_loop_strategy_path,
    injected_failing_filterskymodel_env,
):
    """Verify that after a filter_skymodel failure, restarting Rapthor does not rerun WSClean and that previous WSClean outputs are reused, not regenerated."""

    failing_parset_path = update_parset_path(
        generated_parset_path,
        {
            "allow_internet_access": "False",
            "strategy": str(single_loop_strategy_path),
        },
    )

    first_result = subprocess.run(
        ["rapthor", str(failing_parset_path)],
        capture_output=True,
        text=True,
        check=False,
        env=injected_failing_filterskymodel_env,
    )
    first_output = f"{first_result.stdout}\n{first_result.stderr}"
    assert first_result.returncode != 0, (
        "First run should fail in filter_skymodel via injected wrapper script"
    )

    working_dir = get_working_dir_from_parset(failing_parset_path)
    image_pipeline_dir = Path(working_dir) / "pipelines" / "image_1"
    wsclean_outputs_after_first_run = get_wsclean_output_mtimes(image_pipeline_dir)
    assert wsclean_outputs_after_first_run, (
        "Expected WSClean output products before restart, but none were found. "
        f"First run output was:\n{first_output}"
    )

    # Use now the default environment which as the working filter_skymodel
    second_result = subprocess.run(
        ["rapthor", str(failing_parset_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    second_output = f"{second_result.stdout}\n{second_result.stderr}"
    assert second_result.returncode == 0, f"Second run should succeed. Output was:\n{second_output}"
    assert "Rapthor has finished :)" in second_output

    wsclean_outputs_after_second_run = get_wsclean_output_mtimes(image_pipeline_dir)
    assert set(wsclean_outputs_after_second_run) == set(wsclean_outputs_after_first_run), (
        "WSClean output product set changed after restart. "
        f"Before: {sorted(wsclean_outputs_after_first_run)}; "
        f"after: {sorted(wsclean_outputs_after_second_run)}"
    )
    assert wsclean_outputs_after_second_run == wsclean_outputs_after_first_run, (
        "WSClean output product timestamps changed after restart, suggesting WSClean reran."
    )
