import configparser
import subprocess
from pathlib import Path

import numpy as np
import pytest
from losoto.h5parm import h5parm

from .utils import (
    find_command_records,
    first_command_arguments,
    get_working_dir_from_parset,
    update_parset_path,
)

# The 1 MHz slow-gain interval of the template covers all 8 channels (24 kHz
# wide) of the test MS
SLOW_SOLINT_CHANNELS = 8


def _phases(path):
    with h5parm(str(path), readonly=True) as solutions:
        return solutions.getSolset("sol000").getSoltab("phase000").val


def _set_calibration_averaging(parset_path, average_visibilities):
    """Turn the averaging before the calibration solves on or off.

    The integration template only sets the imaging average_visibilities option,
    which update_parset_path would change instead, so the calibration one is set
    here.
    """
    parset = configparser.ConfigParser()
    parset.read(parset_path)
    parset["calibration"]["average_visibilities"] = str(average_visibilities)
    with Path(parset_path).open("w") as fp:
        parset.write(fp)


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
def test_rapthor_run_dd_fast_phase_medium_phase(generated_parset_path, single_loop_strategy_path):
    """Test a single selfcal loop with DP3.
    DD calibration fast gains and medium gains are performed.

    This is the default single-loop run (peel_bright_sources off and
    shared_facet_rw off), so it also checks that bright-source peeling does
    not run; keep it the only test with these default settings.
    """

    updated_parset_path = update_parset_path(
        generated_parset_path,
        {
            "allow_internet_access": "False",
            "strategy": str(single_loop_strategy_path),
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

    dp3_arguments = first_command_arguments(
        working_dir,
        operation="calibrate_1",
        executable="DP3",
    )

    assert "steps" in dp3_arguments
    assert "solve1" in dp3_arguments["steps"]
    assert "solve2" in dp3_arguments["steps"]
    assert "solve3" not in dp3_arguments["steps"]
    assert "fast_phase_0.h5parm" == dp3_arguments["solve1.h5parm"]
    assert "medium1_phase_0.h5parm" == dp3_arguments["solve2.h5parm"]
    assert "scalarphase" == dp3_arguments["solve1.mode"]
    assert "scalarphase" == dp3_arguments["solve2.mode"]
    assert int(dp3_arguments["solve1.solint"]) < int(dp3_arguments["solve2.solint"])

    # The medium phases must reach the active solutions, not only the fast phases
    solutions = Path(working_dir) / "solutions" / "calibrate_1"
    assert not np.array_equal(
        _phases(solutions / "field-solutions.h5"),
        _phases(solutions / "field-solutions-fast-phase.h5"),
        equal_nan=True,
    )

    restore_commands = find_command_records(
        working_dir,
        operation="image_1",
        executable="wsclean",
        contains="-restore-list",
    )
    assert restore_commands == [], (
        "WSClean restore commands should not run when bright-source peeling is disabled"
    )


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
def test_rapthor_run_dd_fast_medium_slow_gains(
    generated_parset_path, single_loop_strategy_path_fast_medium_slow
):
    """Test a single selfcal loop with DP3.
    DD calibration fast gains, medium gains, and slow gains are performed
    """

    updated_parset_path = update_parset_path(
        generated_parset_path,
        {
            "allow_internet_access": "False",
            "strategy": str(single_loop_strategy_path_fast_medium_slow),
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

    dp3_arguments = first_command_arguments(
        working_dir,
        operation="calibrate_1",
        executable="DP3",
    )

    assert "steps" in dp3_arguments
    assert "solve1" in dp3_arguments["steps"]
    assert "solve2" in dp3_arguments["steps"]
    assert "solve3" in dp3_arguments["steps"]
    assert "solve4" in dp3_arguments["steps"]
    assert "fast_phase_0.h5parm" == dp3_arguments["solve1.h5parm"]
    assert "medium1_phase_0.h5parm" == dp3_arguments["solve2.h5parm"]
    assert "slow_gain_0.h5parm" == dp3_arguments["solve3.h5parm"]
    assert "medium2_phase_0.h5parm" == dp3_arguments["solve4.h5parm"]
    assert "scalarphase" == dp3_arguments["solve1.mode"]
    assert "scalarphase" == dp3_arguments["solve2.mode"]
    assert "diagonal" == dp3_arguments["solve3.mode"]
    assert "scalarphase" == dp3_arguments["solve4.mode"]
    assert int(dp3_arguments["solve1.solint"]) < int(dp3_arguments["solve2.solint"])
    assert int(dp3_arguments["solve2.solint"]) < int(dp3_arguments["solve3.solint"])


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
@pytest.mark.parametrize("average_visibilities", [True, False], ids=["averaged", "unaveraged"])
def test_rapthor_run_dd_slow_gains(
    generated_parset_path, single_loop_strategy_path_calibrate_dd_slow, average_visibilities
):
    """Test a single selfcal loop with DP3.
    DD calibration slow gains are performed, with and without averaging the
    visibilities before the solve
    """

    updated_parset_path = update_parset_path(
        generated_parset_path,
        {
            "allow_internet_access": "False",
            "strategy": str(single_loop_strategy_path_calibrate_dd_slow),
        },
    )
    _set_calibration_averaging(updated_parset_path, average_visibilities)

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

    dp3_arguments = first_command_arguments(
        working_dir,
        operation="calibrate_1",
        executable="DP3",
    )

    assert "steps" in dp3_arguments
    assert "solve1" in dp3_arguments["steps"]
    assert "solve2" not in dp3_arguments["steps"]
    assert "solve3" not in dp3_arguments["steps"]
    assert "slow_gain_0.h5parm" == dp3_arguments["solve1.h5parm"]
    assert "diagonal" == dp3_arguments["solve1.mode"]

    # DP3 counts the solution intervals in the samples that reach the solve, so
    # after averaging they shrink by the averaging steps
    steps = dp3_arguments["steps"].strip("[]").split(",")
    if average_visibilities:
        assert steps[0] == "avg", "The averaging must come before every other step"
        timestep = int(dp3_arguments["avg.timestep"])
        freqstep = int(dp3_arguments["avg.freqstep"])
        assert timestep > 1, "Expected the 10 s samples to be averaged in time"
    else:
        assert "avg" not in steps
        assert "avg.timestep" not in dp3_arguments
        assert "avg.freqstep" not in dp3_arguments
        timestep = freqstep = 1
    # 600 s strategy interval / 10 s samples
    assert int(dp3_arguments["solve1.solint"]) * timestep == 60
    assert int(dp3_arguments["solve1.nchan"]) * freqstep == SLOW_SOLINT_CHANNELS
