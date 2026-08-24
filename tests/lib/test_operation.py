"""
Test cases for the `rapthor.lib.operation` module.
"""

from pathlib import Path

import pytest

from rapthor.lib.field import Field
from rapthor.lib.operation import Operation, extract_log_errors


@pytest.fixture
def mock_create_cwl_runner(mocker, request):
    # Mock the `create_cwl_runner` function to return a mock runner that
    # simulates running a subprocess.
    param = getattr(request, "param", {})
    returncode = param.get("returncode", 0)
    parse_outputs_return_value = param.get("parse_outputs_return_value", None)
    return mocker.patch(
        "rapthor.lib.operation.create_cwl_runner",
        return_value=mocker.Mock(  # call returns context manager
            __enter__=mocker.Mock(  # context returns mock runner
                return_value=mocker.Mock(  # mock runner
                    run=mocker.Mock(return_value=(returncode == 0)),  # simulate run method
                    parse_outputs=mocker.Mock(return_value=parse_outputs_return_value),
                ),
            ),
            __exit__=mocker.Mock(return_value=None),
        ),
    )


class TestOperation:
    """
    Test cases for the `Operation` class in the `rapthor.lib.operation` module.
    """

    @pytest.fixture
    def operation(self, mocker, parset):
        """
        Fixture to create an `Operation` instance for testing.
        """
        field = mocker.Mock(Field)
        field.parset = parset
        return Operation(field, 0, "image")

    @pytest.mark.parametrize(
        "mock_create_cwl_runner",
        [
            {"returncode": 0, "parse_outputs_return_value": {"mock": "outputs"}},
        ],
        indirect=True,
    )
    def test_run(self, operation, mock_create_cwl_runner):
        """
        Test that the `run` method correctly executes the operation and returns
        True when successful.
        """

        operation.run()

        mock_create_cwl_runner.assert_called_once_with(operation.cwl_runner, operation)
        assert operation.outputs == {"mock": "outputs"}
        assert Path(operation.done_file).exists()

    @pytest.mark.parametrize(
        "mock_create_cwl_runner",
        [
            {"returncode": 1},
        ],
        indirect=True,
    )
    def test_run_failure(self, operation, mock_create_cwl_runner):
        """
        Test that the `run` method correctly handles a failure during execution
        and raises a RuntimeError.
        """

        with pytest.raises(RuntimeError):
            operation.run()

        assert operation.outputs == {}
        assert not Path(operation.done_file).exists()

    def test_handle_failure(self, mocker, request, parset):
        """
        Test that the `handle_failure` method correctly raises a RuntimeError
        with the appropriate message when an operation fails.
        """
        field = mocker.Mock(Field)
        field.parset = parset
        operation = Operation(field, 0, "image")
        operation.log_dir = str(request.config.resource_dir)
        operation.pipeline_log_file = str(
            request.config.resource_dir / "failed_workflow_sample.log"
        )
        with pytest.raises(
            RuntimeError, match="antennaconstraint given that should constrain a group of antennas"
        ):
            operation.handle_failure()


def test_extract_log_errors(request):
    """
    Test that the `extract_log_errors` function correctly parses error
    messages from a log file.
    """
    log_file = request.config.resource_dir / "failed_workflow_sample.log"
    records = extract_log_errors(log_file)
    assert len(records) == 6
    assert (
        "std exception detected: Error: antennaconstraint given that should "
        "constrain a group of antennas with one antenna in it. This does not make "
        "sense (did you forget using two square brackets? [[ ant1, ant2 ]] )"
    ) in records[2]
