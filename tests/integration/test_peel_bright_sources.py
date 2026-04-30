"""Integration tests for the Rapthor pipeline when using the peel_bright_sources option."""

import pytest
import subprocess
from pathlib import Path

from .utils import find_step_logs, get_working_dir_from_parset, update_parset_path


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
def test_rapthor_run_single_loop_peel_bright_sources(
    generated_parset_path, single_loop_strategy_path_peel_bright_sources
):
    """Test a single self-calibration loop with peel_bright_sources enabled and disabled."""
    strategy_path, peel_bright_sources = single_loop_strategy_path_peel_bright_sources

    updated_parset_path = update_parset_path(
        generated_parset_path,
        {
            "allow_internet_access": "False",
            "strategy": str(strategy_path),
        },
    )

    command = ["rapthor", str(updated_parset_path)]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    output = f"{result.stdout}\n{result.stderr}"
    assert result.returncode == 0
    assert "Operation calibrate_1 completed" in output
    assert "Operation image_1 completed" in output
    assert "Operation mosaic_1 completed" in output

    # Verify bright-source peeling actually ran by checking wsclean_restore step logs.
    working_dir = get_working_dir_from_parset(updated_parset_path)
    image_logs_dir = Path(working_dir) / "logs" / "image_1"
    restore_pb_logs = find_step_logs(image_logs_dir, "restore_pb.wsclean_restore.cwl")
    restore_nonpb_logs = find_step_logs(image_logs_dir, "restore_nonpb.wsclean_restore.cwl")
    restore_pb_stdout_logs = sorted(
        image_logs_dir.glob("subpipeline_parset.cwl.restore_pb.wsclean_restore.cwl.stdout_*.log")
    )
    restore_nonpb_stdout_logs = sorted(
        image_logs_dir.glob("subpipeline_parset.cwl.restore_nonpb.wsclean_restore.cwl.stdout_*.log")
    )

    if peel_bright_sources:
        assert restore_pb_logs, f"No restore_pb wsclean_restore logs found in {image_logs_dir}"
        assert restore_nonpb_logs, (
            f"No restore_nonpb wsclean_restore logs found in {image_logs_dir}"
        )
        assert restore_pb_stdout_logs, f"No restore_pb stdout logs found in {image_logs_dir}"
        assert restore_nonpb_stdout_logs, f"No restore_nonpb stdout logs found in {image_logs_dir}"
        assert "completed success" in restore_pb_logs[-1].read_text()
        assert "completed success" in restore_nonpb_logs[-1].read_text()
        assert "wsclean" in restore_pb_logs[-1].read_text()
        assert "wsclean" in restore_nonpb_logs[-1].read_text()
    else:
        # With when: $(inputs.peel_bright_sources), Toil still emits CWLJob logs,
        # but command stdout logs are not created and there is no "completed success".
        assert not restore_pb_stdout_logs, (
            "restore_pb stdout log should not exist when peeling is disabled"
        )
        assert not restore_nonpb_stdout_logs, (
            "restore_nonpb stdout log should not exist when peeling is disabled"
        )
        assert "completed success" not in restore_pb_logs[-1].read_text(), (
            "restore_pb wsclean_restore should have been skipped when peeling is disabled"
        )
        assert "completed success" not in restore_nonpb_logs[-1].read_text(), (
            "restore_nonpb wsclean_restore should have been skipped when peeling is disabled"
        )

    assert "Rapthor has finished :)" in output
