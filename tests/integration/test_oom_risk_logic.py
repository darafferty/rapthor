"""Integration coverage for strict DP3 calibration memory checking."""

import runpy
import subprocess

import pytest

from .utils import make_rapthor_command, update_parset_path

INTEGRATION_PARSET = (
    "tests/resources/integration_template.parset",
    "tests/resources/integration_true_sky.txt",
    "tests/resources/integration_apparent_sky.txt",
)


@pytest.mark.integration
@pytest.mark.parametrize("generated_parset_path", [INTEGRATION_PARSET], indirect=True)
def test_cli_stops_before_calibration_when_strict_oom_check_fails(
    generated_parset_path,
    single_loop_strategy_path,
):
    """Strict preflight checking exits non-zero without starting calibration."""
    strategy_steps = runpy.run_path(single_loop_strategy_path)["strategy_steps"]
    strategy_steps[0]["max_directions"] = 100_000_000
    strict_strategy_path = single_loop_strategy_path.with_name("strict_oom_strategy.py")
    strict_strategy_path.write_text(f"strategy_steps = {strategy_steps!r}\n")

    strict_parset_path = update_parset_path(
        generated_parset_path,
        {
            "allow_internet_access": "False",
            "fail_on_calibration_oom_risk": "True",
            "mem_per_node_gb": "1",
            "strategy": str(strict_strategy_path),
        },
    )

    result = subprocess.run(
        make_rapthor_command(strict_parset_path),
        capture_output=True,
        text=True,
        check=False,
    )
    output = f"{result.stdout}\n{result.stderr}"

    assert result.returncode != 0
    assert "pre-flight max_directions upper bound" in output
    assert "likely out of memory by" in output
    assert "Rapthor is stopping because fail_on_calibration_oom_risk=True" in output
    assert "Operation calibrate" not in output
