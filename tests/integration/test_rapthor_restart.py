"""Integration tests for the Rapthor pipeline restart behavior."""

import json
import os
import subprocess
from pathlib import Path

import pytest

from .utils import (
    get_working_dir_from_parset,
    get_wsclean_output_mtimes,
    make_failing_filter_skymodel,
    update_parset_path,
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

    working_dir = Path(get_working_dir_from_parset(generated_parset_path))
    local_scratch = working_dir.parent / "local-scratch"
    global_scratch = working_dir.parent / "global-scratch"
    failing_parset_path = update_parset_path(
        generated_parset_path,
        {
            "allow_internet_access": "False",
            "strategy": str(single_loop_strategy_path),
            "local_scratch_dir": str(local_scratch),
            "global_scratch_dir": str(global_scratch),
            "keep_temporary_files": "False",
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
        "First run should fail in filter_skymodel via the injected Python shim"
    )

    working_dir = get_working_dir_from_parset(failing_parset_path)
    image_pipeline_dir = Path(working_dir) / "pipelines" / "image_1"
    assert not image_pipeline_dir.is_symlink()
    assert not list(image_pipeline_dir.parent.glob(".*.scratch"))
    assert list(global_scratch.iterdir()) == []
    wsclean_outputs_after_first_run = get_wsclean_output_mtimes(image_pipeline_dir)
    channel_images = list(image_pipeline_dir.glob("*-0???-*.fits"))
    assert channel_images, "Failed workflows must retain channel images for recovery"
    assert wsclean_outputs_after_first_run, (
        "Expected WSClean output products before restart, but none were found. "
        f"First run output was:\n{first_output}"
    )

    # Use the default environment, which has the working skymodel filter adapter.
    second_result = subprocess.run(
        ["rapthor", str(failing_parset_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    second_output = f"{second_result.stdout}\n{second_result.stderr}"
    assert second_result.returncode == 0, f"Second run should succeed. Output was:\n{second_output}"
    assert "Rapthor has finished :)" in second_output
    assert not image_pipeline_dir.is_symlink()
    assert not list(image_pipeline_dir.parent.glob(".*.scratch"))
    assert list(global_scratch.iterdir()) == []
    assert all(not path.exists() for path in channel_images)
    assert not list(image_pipeline_dir.glob("*-MFS-psf.fits"))
    records = [
        json.loads(line)
        for line in (Path(working_dir) / "logs" / "commands.jsonl").read_text().splitlines()
    ]
    wsclean_records = [record for record in records if "-temp-dir" in record["command"]]
    assert wsclean_records
    for record in wsclean_records:
        command = record["command"]
        temp_dir = Path(command[command.index("-temp-dir") + 1])
        assert temp_dir.is_relative_to(local_scratch)
        assert record["environment"]["TMPDIR"] == str(temp_dir)
        assert not temp_dir.exists()

    wsclean_outputs_after_second_run = get_wsclean_output_mtimes(image_pipeline_dir)
    assert set(wsclean_outputs_after_second_run) == set(wsclean_outputs_after_first_run), (
        "WSClean output product set changed after restart. "
        f"Before: {sorted(wsclean_outputs_after_first_run)}; "
        f"after: {sorted(wsclean_outputs_after_second_run)}"
    )
    assert wsclean_outputs_after_second_run == wsclean_outputs_after_first_run, (
        "WSClean output product timestamps changed after restart, suggesting WSClean reran."
    )
