import subprocess
from pathlib import Path

import pytest

from .utils import (
    find_step_logs,
    get_working_dir_from_parset,
    parse_dp3_args_from_log,
    update_parset_path,
)


@pytest.mark.integration
@pytest.mark.xfail(reason="calibration_strategy still not implemented fully", strict=True)
@pytest.mark.parametrize(
    "generated_parset_path,single_loop_strategy_with_calibration_strategy",
    [
        (
            (
                "tests/resources/integration_template.parset",
                "tests/resources/integration_true_sky.txt",
                "tests/resources/integration_apparent_sky.txt",
            ),
            {"di": ["fast_phase"], "dd": []},
        )
    ],
    indirect=["generated_parset_path", "single_loop_strategy_with_calibration_strategy"],
)
def test_rapthor_run_single_loop_calibrate_di_fast_phase(
    generated_parset_path, single_loop_strategy_with_calibration_strategy
):
    """Test a single selfcal loop with DP3 when only fast phase
    direction-independent calibration is specified in the
    strategy.
    """

    updated_parset_path = update_parset_path(
        generated_parset_path,
        {
            "allow_internet_access": "False",
            "strategy": str(single_loop_strategy_with_calibration_strategy),
        },
    )

    working_dir = get_working_dir_from_parset(updated_parset_path)
    print("---Rapthor working dir: ", working_dir)

    command = ["rapthor", str(updated_parset_path)]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    output = f"{result.stdout}\n{result.stderr}"
    assert result.returncode == 0, f"Rapthor failed with output:\n{output}"
    assert "Operation calibrate_1 completed" in output
    assert "Operation predict_1 completed" in output
    assert "Operation image_1 completed" in output
    assert "Operation mosaic_1 completed" in output
    assert "Rapthor has finished :)" in output

    calibrate_logs_dir = Path(working_dir) / "logs" / "calibrate_1"
    calibrate_log = find_step_logs(calibrate_logs_dir, "ddecal_solve.cwl")
    assert calibrate_log, "Expected calibration logs to be present"
    dp3_arguments = parse_dp3_args_from_log(calibrate_log[0])

    assert "steps" in dp3_arguments
    assert "solve1.directions" not in dp3_arguments, "Expected only direction independent run"
    assert "solve1" in dp3_arguments["steps"]
    assert "solve2" not in dp3_arguments["steps"]
    assert "solve3" not in dp3_arguments["steps"]
    assert "solve4" not in dp3_arguments["steps"]
    assert "avg" not in dp3_arguments["steps"]
    assert "scalarphase" in dp3_arguments["solve1.mode"]
