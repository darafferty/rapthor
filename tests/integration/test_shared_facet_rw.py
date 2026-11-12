"""Integration tests for the shared_facet_rw feature."""

import pytest

from .utils import (
    find_command_records,
    get_working_dir_from_parset,
    run_rapthor,
    update_parset_path,
)


@pytest.mark.integration
# The shared_facet_rw=False case is the default single-loop run, which
# tests/integration/test_dd_calibration.py checks.
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
def test_rapthor_run_single_loop(generated_parset_path, single_loop_strategy_path):
    """Test a single self-calibration loop end to end with shared facet I/O."""

    updated_parset_path = update_parset_path(
        generated_parset_path,
        {
            "allow_internet_access": "False",
            "strategy": str(single_loop_strategy_path),
            # Keep the three patches of the input sky model, so that there are
            # as many facets as parallel gridding tasks and sharing is used
            "regroup_input_skymodel": "False",
            "shared_facet_rw": "True",
        },
    )

    output = run_rapthor(updated_parset_path)

    assert "Operation calibrate_1 completed" in output
    assert "Operation image_1 completed" in output
    assert "Operation mosaic_1 completed" in output
    wsclean_commands = find_command_records(
        get_working_dir_from_parset(updated_parset_path),
        operation="image_1",
        executable="wsclean",
        contains="-facet-regions",
    )
    assert wsclean_commands
    for record in wsclean_commands:
        assert "-shared-facet-reads" in record.command
        assert "-shared-facet-writes" in record.command
