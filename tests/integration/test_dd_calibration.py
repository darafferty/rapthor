import logging
from pathlib import Path

import pytest

from rapthor.process import run as run_rapthor
from rapthor.testing import assert_logged

from .utils import (
    find_step_logs,
    get_working_dir_from_parset,
    parse_dp3_args_from_log,
    update_parset_path,
)


def _test_calibrate(parset_path, caplog):
    """
    Run the Rapthor calibration workflow with the given parset file and check
    that the job completed and that the expected messages appear in the output.

    Returns:
        dict: Parsed DP3 arguments from the calibration log.
    """
    working_dir = get_working_dir_from_parset(parset_path)
    print("---Rapthor working dir: ", working_dir)

    expected_messages = [
        "Operation calibrate_1 completed",
        "Operation predict_1 completed",
        "Operation image_1 completed",
        "Operation mosaic_1 completed",
        "Rapthor has finished :)",
    ]
    with assert_logged(caplog, "rapthor", logging.INFO, *expected_messages):
        run_rapthor(str(parset_path))

    calibrate_logs_dir = Path(working_dir) / "logs" / "calibrate_1"
    calibrate_log = find_step_logs(calibrate_logs_dir, "ddecal_solve.cwl")
    assert calibrate_log, "Expected calibration logs to be present"

    return parse_dp3_args_from_log(calibrate_log[0])


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
def test_rapthor_run_dd_fast_phase_medium_phase(
    generated_parset_path, single_loop_strategy_path, caplog
):
    """Test a single selfcal loop with DP3.
    ddecal fast_gains and medium gains are performed
    """

    updated_parset_path = update_parset_path(
        generated_parset_path,
        {
            "allow_internet_access": "False",
            "strategy": str(single_loop_strategy_path),
        },
    )

    dp3_arguments = _test_calibrate(updated_parset_path, caplog)

    assert "steps" in dp3_arguments
    assert "solve1" in dp3_arguments["steps"]
    assert "solve2" in dp3_arguments["steps"]
    assert "solve3" not in dp3_arguments["steps"]
    assert "fast_phase_0.h5parm" == dp3_arguments["solve1.h5parm"]
    assert "medium1_phase_0.h5parm" == dp3_arguments["solve2.h5parm"]
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
def test_rapthor_run_dd_fast_medium_slow_gains(
    generated_parset_path, single_loop_strategy_path_fast_medium_slow, caplog
):
    """Test a single selfcal loop with DP3.
    ddecal fast_gains, medium gains, and slow gains are performed
    """

    updated_parset_path = update_parset_path(
        generated_parset_path,
        {
            "allow_internet_access": "False",
            "strategy": str(single_loop_strategy_path_fast_medium_slow),
        },
    )

    dp3_arguments = _test_calibrate(updated_parset_path, caplog)

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
def test_rapthor_run_dd_slow_gains(
    generated_parset_path, single_loop_strategy_path_calibrate_dd_slow, caplog
):
    """Test a single selfcal loop with DP3.
    ddecal slow gains are performed
    """

    updated_parset_path = update_parset_path(
        generated_parset_path,
        {
            "allow_internet_access": "False",
            "strategy": str(single_loop_strategy_path_calibrate_dd_slow),
        },
    )

    dp3_arguments = _test_calibrate(updated_parset_path, caplog)

    assert "steps" in dp3_arguments
    assert "solve1" in dp3_arguments["steps"]
    assert "solve2" not in dp3_arguments["steps"]
    assert "solve3" not in dp3_arguments["steps"]
    assert "slow_gain_0.h5parm" == dp3_arguments["solve1.h5parm"]
    assert "diagonal" == dp3_arguments["solve1.mode"]
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
def test_rapthor_run_dd_wsclean_predict_fast_phase_medium_phase(
    generated_parset_path, single_loop_strategy_path, caplog
):
    """Test a single selfcal loop with DP3.
    ddecal fast_gains and medium gains are performed
    """

    updated_parset_path = update_parset_path(
        generated_parset_path,
        {
            "allow_internet_access": "False",
            "use_wsclean_predict": "True",
            "strategy": str(single_loop_strategy_path),
        },
    )

    dp3_arguments = _test_calibrate(updated_parset_path, caplog)

    assert "steps" in dp3_arguments
    assert "solve1" in dp3_arguments["steps"]
    assert "solve2" in dp3_arguments["steps"]
    assert "solve3" not in dp3_arguments["steps"]
    assert "fast_phase_0.h5parm" == dp3_arguments["solve1.h5parm"]
    assert "medium1_phase_0.h5parm" == dp3_arguments["solve2.h5parm"]
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
def test_rapthor_run_dd_with_antenna_constraints(
    generated_parset_path, single_loop_strategy_path, caplog
):
    """Test a single selfcal loop with DP3, using antenna constraints."""

    updated_parset_path = update_parset_path(
        generated_parset_path,
        {
            "allow_internet_access": "False",
            "antenna_constraints": "[[CS001HBA0, CS002HBA0, CS002HBA1, CS004HBA1]]",
            "strategy": str(single_loop_strategy_path),
        },
    )

    dp3_arguments = _test_calibrate(updated_parset_path, caplog)
    assert (
        dp3_arguments["solve1_antennaconstraint"]
        == "[[CS001HBA0, CS002HBA0, CS002HBA1, CS004HBA1]]"
    )
    assert (
        dp3_arguments["solve2_antennaconstraint"]
        == "[[CS001HBA0, CS002HBA0, CS002HBA1, CS004HBA1]]"
    )
    assert dp3_arguments["solve3_antennaconstraint"] == "[]"
    assert dp3_arguments["solve4_antennaconstraint"] == "[]"
