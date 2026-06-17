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
    assert "Operation calibrate_di_1 completed" in output
    assert "Operation predict_di_1 completed" in output
    assert "Operation image_1 completed" in output
    assert "Operation mosaic_1 completed" in output
    assert "Operation predict_1 completed" not in output
    assert "Rapthor has finished :)" in output

    calibrate_logs_dir = Path(working_dir) / "logs" / "calibrate_di_1"
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


# fast medium slow for DI


@pytest.mark.integration
@pytest.mark.parametrize(
    "generated_parset_path,single_loop_strategy_with_calibration_strategy",
    [
        (
            (
                "tests/resources/integration_template.parset",
                "tests/resources/integration_true_sky.txt",
                "tests/resources/integration_apparent_sky.txt",
            ),
            {"di": ["slow_gains"], "dd": []},
        )
    ],
    indirect=["generated_parset_path", "single_loop_strategy_with_calibration_strategy"],
)
def test_rapthor_run_single_loop_calibrate_di_slow_gains(
    generated_parset_path, single_loop_strategy_with_calibration_strategy
):
    """Test a single selfcal loop with DP3 when only slow gains
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
    assert "Operation calibrate_di_1 completed" in output
    assert "Operation predict_di_1 completed" in output
    assert "Operation image_1 completed" in output
    assert "Operation mosaic_1 completed" in output
    assert "Rapthor has finished :)" in output

    calibrate_logs_dir = Path(working_dir) / "logs" / "calibrate_di_1"
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
    assert "diagonal" in dp3_arguments["solve1.mode"]
    assert "slow_gains_di_0.h5parm" == dp3_arguments["solve1.h5parm"]
    assert int(dp3_arguments["solve1.solint"]) == 60  # 600 s strategy interval / 10 s samples


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
def test_rapthor_run_di_fast_phase_medium_phase(
    generated_parset_path, single_loop_strategy_path_calibrate_di_fast_medium_phase
):
    """
    Test a single selfcal loop with DP3.
    DI fast_phase and medium_phase solves are performed
    """

    updated_parset_path = update_parset_path(
        generated_parset_path,
        {
            "allow_internet_access": "False",
            "strategy": str(single_loop_strategy_path_calibrate_di_fast_medium_phase),
        },
    )

    working_dir = get_working_dir_from_parset(updated_parset_path)
    print("---Rapthor working dir: ", working_dir)

    command = ["rapthor", str(updated_parset_path)]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    output = f"{result.stdout}\n{result.stderr}"
    assert result.returncode == 0, f"Rapthor failed with output:\n{output}"
    assert "Operation calibrate_di_1 completed" in output
    assert "Operation predict_di_1 completed" in output
    assert "Operation image_1 completed" in output
    assert "Operation mosaic_1 completed" in output
    assert "Rapthor has finished :)" in output

    calibrate_di_logs_dir = Path(working_dir) / "logs" / "calibrate_di_1"
    calibrate_di_log = find_step_logs(calibrate_di_logs_dir, "ddecal_solve.cwl")
    assert calibrate_di_log, "Expected DI calibration logs to be present"
    dp3_arguments = parse_dp3_args_from_log(calibrate_di_log[0])

    assert "steps" in dp3_arguments
    assert "solve1" in dp3_arguments["steps"]
    assert "solve2" in dp3_arguments["steps"]
    assert "solve3" not in dp3_arguments["steps"]
    assert "fast_phase_di_0.h5parm" == dp3_arguments["solve1.h5parm"]
    assert "medium1_phase_di_0.h5parm" == dp3_arguments["solve2.h5parm"]
    assert "scalarphase" == dp3_arguments["solve1.mode"]
    assert "scalarphase" == dp3_arguments["solve2.mode"]
    assert int(dp3_arguments["solve1.solint"]) < int(dp3_arguments["solve2.solint"])


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
def test_rapthor_run_di_fast_phase_medium_slow(
    generated_parset_path, single_loop_strategy_path_calibrate_di_fast_medium_slow
):
    """
    Test a single selfcal loop with DP3.
    DI fast_phase, medium_phase, and slow gains solves are performed
    """

    updated_parset_path = update_parset_path(
        generated_parset_path,
        {
            "allow_internet_access": "False",
            "strategy": str(single_loop_strategy_path_calibrate_di_fast_medium_slow),
        },
    )

    working_dir = get_working_dir_from_parset(updated_parset_path)
    print("---Rapthor working dir: ", working_dir)

    command = ["rapthor", str(updated_parset_path)]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    output = f"{result.stdout}\n{result.stderr}"
    assert result.returncode == 0, f"Rapthor failed with output:\n{output}"
    assert "Operation calibrate_di_1 completed" in output
    assert "Operation predict_di_1 completed" in output
    assert "Operation image_1 completed" in output
    assert "Operation mosaic_1 completed" in output
    assert "Rapthor has finished :)" in output

    calibrate_di_logs_dir = Path(working_dir) / "logs" / "calibrate_di_1"
    calibrate_di_log = find_step_logs(calibrate_di_logs_dir, "ddecal_solve.cwl")
    assert calibrate_di_log, "Expected DI calibration logs to be present"
    dp3_arguments = parse_dp3_args_from_log(calibrate_di_log[0])

    assert "steps" in dp3_arguments
    assert "solve1" in dp3_arguments["steps"]
    assert "solve2" in dp3_arguments["steps"]
    assert "solve3" in dp3_arguments["steps"]
    assert "solve4" in dp3_arguments["steps"]
    assert "fast_phase_di_0.h5parm" == dp3_arguments["solve1.h5parm"]
    assert "medium1_phase_di_0.h5parm" == dp3_arguments["solve2.h5parm"]
    assert "slow_gains_di_0.h5parm" == dp3_arguments["solve3.h5parm"]
    assert "medium2_phase_di_0.h5parm" == dp3_arguments["solve4.h5parm"]
    assert "scalarphase" == dp3_arguments["solve1.mode"]
    assert "scalarphase" == dp3_arguments["solve2.mode"]
    assert "diagonal" == dp3_arguments["solve3.mode"]
    assert "scalarphase" == dp3_arguments["solve4.mode"]
    assert "solve3.solint" in dp3_arguments
    assert int(dp3_arguments["solve1.solint"]) < int(dp3_arguments["solve2.solint"])
    assert int(dp3_arguments["solve2.solint"]) < int(dp3_arguments["solve3.solint"])


@pytest.mark.integration
@pytest.mark.parametrize(
    "generated_parset_path,single_loop_strategy_with_calibration_strategy",
    [
        (
            (
                "tests/resources/integration_template.parset",
                "tests/resources/integration_true_sky.txt",
                "tests/resources/integration_apparent_sky.txt",
            ),
            {"di": ["full_jones"], "dd": []},
        )
    ],
    indirect=["generated_parset_path", "single_loop_strategy_with_calibration_strategy"],
)
def test_rapthor_run_single_loop_calibrate_di_full_jones(
    generated_parset_path, single_loop_strategy_with_calibration_strategy
):
    """Test a single selfcal loop with DP3 when only full-Jones
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
    assert "Operation calibrate_di_1 completed" in output

    calibrate_logs_dir = Path(working_dir) / "logs" / "calibrate_di_1"
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
    assert "fulljones" in dp3_arguments["solve1.mode"]


@pytest.mark.integration
@pytest.mark.parametrize(
    "generated_parset_path,single_loop_strategy_with_calibration_strategy,expected_order",
    [
        (
            (
                "tests/resources/integration_template.parset",
                "tests/resources/integration_true_sky.txt",
                "tests/resources/integration_apparent_sky.txt",
            ),
            {"di": ["full_jones"], "dd": ["fast_phase"]},
            ["predict_di_1", "calibrate_di_1", "calibrate_1"],
        ),
        (
            (
                "tests/resources/integration_template.parset",
                "tests/resources/integration_true_sky.txt",
                "tests/resources/integration_apparent_sky.txt",
            ),
            {"dd": ["fast_phase"], "di": ["full_jones"]},
            ["calibrate_1", "predict_di_1", "calibrate_di_1"],
        ),
    ],
    indirect=["generated_parset_path", "single_loop_strategy_with_calibration_strategy"],
)
def test_rapthor_run_mixed_di_dd_order(
    generated_parset_path,
    single_loop_strategy_with_calibration_strategy,
    expected_order,
):
    """Test mixed DI/DD calibration order and branch-to-branch solution application."""

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
    for operation in expected_order:
        assert f"Operation {operation} completed" in output
    operation_positions = [
        output.index(f"Operation {operation} completed") for operation in expected_order
    ]
    assert operation_positions == sorted(operation_positions)
    assert "Operation image_1 completed" in output
    assert "Operation mosaic_1 completed" in output
    assert "Rapthor has finished :)" in output

    if expected_order[0] == "predict_di_1":
        calibrate_logs_dir = Path(working_dir) / "logs" / "calibrate_1"
        calibrate_log = find_step_logs(calibrate_logs_dir, "ddecal_solve.cwl")
        assert calibrate_log, "Expected DD calibration logs to be present"
        dp3_arguments = parse_dp3_args_from_log(calibrate_log[0])
        assert "applycal" in dp3_arguments["steps"]
        assert "fulljones" in dp3_arguments["applycal.steps"]
    else:
        predict_logs_dir = Path(working_dir) / "logs" / "predict_di_1"
        predict_log = find_step_logs(predict_logs_dir, "predict_model_data.cwl")
        assert predict_log, "Expected DI prediction logs to be present"
        dp3_arguments = parse_dp3_args_from_log(predict_log[0])
        assert "fastphase" in dp3_arguments["predict.applycal.steps"]


@pytest.mark.integration
@pytest.mark.parametrize(
    "generated_parset_path",
    [
            (
                "tests/resources/integration_template.parset",
                "tests/resources/integration_true_sky.txt",
                "tests/resources/integration_apparent_sky.txt",
            ),
    ],
    indirect=["generated_parset_path"],
) 
def test_rapthor_run_multi_cycles(
    generated_parset_path,
    two_loop_strategy_with_calibration_strategy,
):
    """Test two cycle DI/DD calibration order and branch-to-branch solution application."""

    updated_parset_path = update_parset_path(
        generated_parset_path,
        {
            "allow_internet_access": "False",
            "strategy": str(two_loop_strategy_with_calibration_strategy),
        }
    )
    
    working_dir = get_working_dir_from_parset(updated_parset_path)
    print("---Rapthor working dir: ", working_dir)

    command = ["rapthor", str(updated_parset_path)]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    output = f"{result.stdout}\n{result.stderr}"
    assert result.returncode == 0, f"Rapthor failed with output:\n{output}"

    assert "Operation calibrate_di_1 completed" in output
    assert "Operation predict_di_1 completed" in output
    assert "Operation image_1 completed" in output
    assert "Operation mosaic_1 completed" in output
    assert "Operation calibrate_2 completed" in output
    assert "Operation predict_2 completed" in output
    assert "Operation image_2 completed" in output
    assert "Operation mosaic_2 completed" in output
    assert "Rapthor has finished :)" in output