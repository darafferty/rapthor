"""Integration tests for wsclean parallel gridding feature."""

import os
import re
import stat
import subprocess
import textwrap
from pathlib import Path

import pytest

from .utils import get_working_dir_from_parset, parse_cmd_args_from_logs, update_parset_path


@pytest.fixture
def mock_mpi_run(tmp_path, monkeypatch):
    script = tmp_path / "mpirun"
    script.write_text(
        textwrap.dedent("""\
        #!/usr/bin/env python3
        import sys
        import os
        index = sys.argv.index("wsclean-mp")                      
        
        os.system("wsclean "+" ".join(sys.argv[index+1:]))
                                      
    """)
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)

    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ.get("PATH", ""))
    return script


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
@pytest.mark.parametrize("parallel_gridding_tasks", ["1", "2"])
@pytest.mark.parametrize("dde_method", ["full", "single"])
@pytest.mark.parametrize("use_mpi", ["True", "False"])
def test_rapthor_parallel_gridding(
    use_mpi,
    dde_method,
    parallel_gridding_tasks,
    generated_parset_path,
    single_loop_strategy_path,
    mock_mpi_run,
):
    """
    Test a single self-calibration loop end to end.
    Checking if the parallel gridding parameter is passed properly.
    """

    updated_parset_path = update_parset_path(
        generated_parset_path,
        {
            "allow_internet_access": "False",
            "use_mpi": use_mpi,
            "dde_method": dde_method,
            "parallel_gridding_tasks": parallel_gridding_tasks,
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

    working_dir = get_working_dir_from_parset(updated_parset_path)
    image_logs_dir = Path(working_dir) / "logs" / "image_1"
    mpi_tag = "_mpi_" if use_mpi else ""
    dde_tag = "no_dde" if dde_method == "single" else "facets"
    expected_log_file = (
        f"CWLJob_subpipeline_parset.cwl.image.wsclean{mpi_tag}image_{dde_tag}.cwl_000.log"
    )
    wsclean_log_file = image_logs_dir / expected_log_file

    wsclean_cmd = "wsclean-mp" if use_mpi else "wsclean"
    parsed_cmd = parse_cmd_args_from_logs(wsclean_log_file, wsclean_cmd)
    assert re.search(rf"-parallel-gridding\s+{parallel_gridding_tasks}\b", parsed_cmd)
