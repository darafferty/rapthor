import json
import shlex
from pathlib import Path

import pytest
from prefect.testing.utilities import prefect_test_harness

import rapthor.execution.flows.calibrate as calibrate_module
from rapthor.execution.config import ExecutionConfig
from rapthor.execution.flows.calibrate import (
    build_collect_h5parms_command,
    build_ddecal_solve_command,
    build_plot_solutions_command,
    calibrate_chunk_task,
    calibrate_flow,
    calibrate_payload_from_inputs,
    normalized_collect_h5parms_command,
    normalized_ddecal_solve_command,
    normalized_plot_solutions_command,
    run_calibrate_flow,
)
from rapthor.execution.outputs import directory_record, file_record, validate_output_record

FIXTURE_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fake_calibrate_shell_operation_cls():
    class FakeCalibrateShellOperation:
        instances = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.instances.append(self)

        def run(self):
            tokens = shlex.split(self.kwargs["commands"][0])
            cwd = Path(self.kwargs["cwd"])
            if tokens[0] == "DP3":
                output_name = next(
                    token.split("=", 1)[1] for token in tokens if token.startswith("solve1.h5parm=")
                )
                output_path = cwd / output_name
                output_path.write_text("h5parm")
                return "OK"
            if tokens[:2] == ["H5parm_collector.py", "-c"]:
                output_name = next(
                    token.split("=", 1)[1] for token in tokens if token.startswith("--outh5parm=")
                )
                output_path = cwd / output_name
                output_path.write_text("collected")
                return "OK"
            if tokens[0] == "plotrapthor":
                output_path = cwd / f"{tokens[2]}_solutions.png"
                output_path.write_text("plot")
                return "OK"
            raise AssertionError(f"Unexpected command: {tokens[0]}")

    return FakeCalibrateShellOperation


class NoOutputShellOperation:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def run(self):
        return "OK"


def _di_fulljones_input_parms():
    return {
        "timechunk_filename": [
            directory_record("/data/obs_0.ms"),
            directory_record("/data/obs_1.ms"),
        ],
        "data_colname": "DATA",
        "modeldatacolumn": "[MODEL_DATA]",
        "starttime": ["50000.0", "50010.0"],
        "ntimes": [10, 12],
        "dp3_steps": "[solve1]",
        "output_solve1_h5parm": [
            "fulljones_gain_0.h5parm",
            "fulljones_gain_1.h5parm",
        ],
        "collected_solve1_h5parm": "fulljones_solutions.h5",
        "solint_solve1_timestep": [5, 6],
        "solint_solve1_freqstep": [2, 3],
        "solve1_mode": "fulljones",
        "smoothnessconstraint_fulljones": 1.5,
        "llssolver": "qr",
        "maxiter": 50,
        "propagatesolutions": True,
        "solveralgorithm": "directionsolve",
        "solverlbfgs_dof": 200.0,
        "solverlbfgs_iter": 4,
        "solverlbfgs_minibatches": 1,
        "stepsize": 0.2,
        "stepsigma": 0.0,
        "tolerance": 0.0001,
        "uvlambdamin": 80.0,
        "correctfreqsmearing": False,
        "correcttimesmearing": True,
        "max_threads": 4,
    }


def _di_fulljones_solve_slots():
    return [
        {
            "slot": 1,
            "h5parm": "fulljones_gain_0.h5parm",
            "solint": 5,
            "mode": "fulljones",
            "nchan": 2,
            "llssolver": "qr",
            "maxiter": 50,
            "propagatesolutions": True,
            "initialsolutions_soltab": "[phase000]",
            "solveralgorithm": "directionsolve",
            "solverlbfgs_dof": 200.0,
            "solverlbfgs_iter": 4,
            "solverlbfgs_minibatches": 1,
            "stepsize": 0.2,
            "stepsigma": 0.0,
            "tolerance": 0.0001,
            "uvlambdamin": 80.0,
            "smoothnessconstraint": 1.5,
            "antennaconstraint": "[]",
            "correctfreqsmearing": False,
            "correcttimesmearing": True,
            "keepmodel": "True",
        }
    ]


def test_calibrate_command_builders_match_reference_fixtures():
    commands = json.loads((FIXTURE_DIR / "cwl_reference_commands.json").read_text())

    assert (
        normalized_ddecal_solve_command(
            msin="obs_0.ms",
            data_colname="DATA",
            starttime="50000.0",
            ntimes=10,
            steps="[solve1]",
            solve_slots=_di_fulljones_solve_slots(),
            numthreads=4,
            modeldatacolumn="[MODEL_DATA]",
        )
        == commands["calibrate"]["ddecal_di_fulljones"]
    )
    assert (
        normalized_collect_h5parms_command(
            ["fulljones_gain_0.h5parm", "fulljones_gain_1.h5parm"],
            "fulljones_solutions.h5",
        )
        == commands["calibrate"]["collect_fulljones"]
    )
    assert (
        normalized_plot_solutions_command("fulljones_solutions.h5", "phase")
        == commands["calibrate"]["plot_fulljones_phase"]
    )


def test_calibrate_command_builders_create_cwl_equivalent_tokens():
    assert build_collect_h5parms_command(
        ["fulljones_gain_0.h5parm", "fulljones_gain_1.h5parm"],
        "fulljones_solutions.h5",
    ) == [
        "H5parm_collector.py",
        "-c",
        "fulljones_gain_0.h5parm,fulljones_gain_1.h5parm",
        "--outh5parm=fulljones_solutions.h5",
    ]
    assert build_plot_solutions_command("fulljones_solutions.h5", "phase") == [
        "plotrapthor",
        "fulljones_solutions.h5",
        "phase",
    ]
    assert (
        build_ddecal_solve_command(
            msin="obs_0.ms",
            data_colname="DATA",
            starttime="50000.0",
            ntimes=10,
            steps="[solve1]",
            solve_slots=_di_fulljones_solve_slots(),
            numthreads=4,
            modeldatacolumn="[MODEL_DATA]",
        )[0]
        == "DP3"
    )


def test_calibrate_payload_from_inputs_builds_di_fulljones_payload(tmp_path):
    payload = calibrate_payload_from_inputs("di", _di_fulljones_input_parms(), tmp_path)

    assert payload["mode"] == "di"
    assert payload["pipeline_working_dir"] == str(tmp_path)
    assert payload["collected_h5parm"] == "fulljones_solutions.h5"
    assert payload["collected_h5parm_path"] == str(tmp_path / "fulljones_solutions.h5")
    assert payload["chunks"] == [
        {
            "msin": "/data/obs_0.ms",
            "starttime": "50000.0",
            "ntimes": 10,
            "output_h5parm": "fulljones_gain_0.h5parm",
            "output_h5parm_path": str(tmp_path / "fulljones_gain_0.h5parm"),
            "solve1_solint": 5,
            "solve1_nchan": 2,
        },
        {
            "msin": "/data/obs_1.ms",
            "starttime": "50010.0",
            "ntimes": 12,
            "output_h5parm": "fulljones_gain_1.h5parm",
            "output_h5parm_path": str(tmp_path / "fulljones_gain_1.h5parm"),
            "solve1_solint": 6,
            "solve1_nchan": 3,
        },
    ]


@pytest.mark.parametrize(
    "mode, updates, match",
    [
        ("dd", {}, "Only DI calibration payloads"),
        ("di", {"solve1_mode": "scalarphase"}, "Only DI full-Jones"),
        ("di", {"dp3_steps": "[solve1,solve2]"}, "Only single-solve"),
    ],
)
def test_calibrate_payload_from_inputs_rejects_unsupported_slice(tmp_path, mode, updates, match):
    input_parms = _di_fulljones_input_parms()
    input_parms.update(updates)

    with pytest.raises(ValueError, match=match):
        calibrate_payload_from_inputs(mode, input_parms, tmp_path)


def test_calibrate_chunk_task_runs_with_mocked_shell(tmp_path, fake_calibrate_shell_operation_cls):
    payload = calibrate_payload_from_inputs("di", _di_fulljones_input_parms(), tmp_path)
    task_fn = getattr(calibrate_chunk_task, "fn", calibrate_chunk_task)

    output = task_fn(
        payload,
        payload["chunks"][0],
        execution_config=ExecutionConfig(task_runner="sync"),
        shell_operation_cls=fake_calibrate_shell_operation_cls,
    )

    assert output == file_record(tmp_path / "fulljones_gain_0.h5parm")
    assert fake_calibrate_shell_operation_cls.instances[0].kwargs["cwd"] == str(tmp_path)


def test_run_calibrate_flow_supports_di_fulljones(tmp_path, fake_calibrate_shell_operation_cls):
    payload = calibrate_payload_from_inputs("di", _di_fulljones_input_parms(), tmp_path)

    outputs = run_calibrate_flow(
        payload,
        execution_config=ExecutionConfig(task_runner="sync"),
        shell_operation_cls=fake_calibrate_shell_operation_cls,
    )

    expected_solution = file_record(tmp_path / "fulljones_solutions.h5")
    assert outputs == {
        "combined_solutions": expected_solution,
        "fast_phase_solutions": expected_solution,
        "fast_phase_plots": [file_record(tmp_path / "phase_solutions.png")],
    }
    for value in outputs.values():
        validate_output_record(value)

    commands = [
        shlex.split(instance.kwargs["commands"][0])
        for instance in fake_calibrate_shell_operation_cls.instances
    ]
    assert [command[0] for command in commands] == [
        "DP3",
        "DP3",
        "H5parm_collector.py",
        "plotrapthor",
    ]
    assert "solve1.mode=fulljones" in commands[0]
    assert "solve1.h5parm=fulljones_gain_0.h5parm" in commands[0]
    assert "numthreads=4" in commands[0]
    assert (
        commands[2][2]
        == f"{tmp_path / 'fulljones_gain_0.h5parm'},{tmp_path / 'fulljones_gain_1.h5parm'}"
    )


def test_calibrate_prefect_flow_entrypoint_runs_with_mocked_shell(
    tmp_path, monkeypatch, fake_calibrate_shell_operation_cls
):
    monkeypatch.setattr(
        "rapthor.execution.shell._load_shell_operation_cls",
        lambda: fake_calibrate_shell_operation_cls,
    )

    with prefect_test_harness(server_startup_timeout=None):
        outputs = calibrate_flow(
            calibrate_payload_from_inputs("di", _di_fulljones_input_parms(), tmp_path),
            execution_config=ExecutionConfig(task_runner="sync"),
        )

    assert outputs["combined_solutions"] == file_record(tmp_path / "fulljones_solutions.h5")
    assert len(fake_calibrate_shell_operation_cls.instances) == 4


def test_calibrate_prefect_tasks_submit_all_chunks_before_collect(monkeypatch, tmp_path):
    payload = calibrate_payload_from_inputs("di", _di_fulljones_input_parms(), tmp_path)
    events = []

    class FakeFuture:
        def __init__(self, index):
            self.index = index

        def result(self):
            events.append(f"result-{self.index}")
            return file_record(tmp_path / f"fulljones_gain_{self.index}.h5parm")

    def fake_submit(payload_arg, chunk, execution_config=None):
        assert payload_arg is payload
        assert execution_config == ExecutionConfig(task_runner="sync")
        events.append(f"submit-{chunk['output_h5parm']}")
        return FakeFuture(len(events) - 1)

    def fake_collect(payload_arg, solve_records, execution_config, shell_operation_cls=None):
        _ = shell_operation_cls
        events.append("collect")
        assert payload_arg is payload
        assert execution_config == ExecutionConfig(task_runner="sync")
        assert solve_records == [
            file_record(tmp_path / "fulljones_gain_0.h5parm"),
            file_record(tmp_path / "fulljones_gain_1.h5parm"),
        ]
        return {"combined_solutions": file_record(tmp_path / "fulljones_solutions.h5")}

    monkeypatch.setattr(calibrate_module.calibrate_chunk_task, "submit", fake_submit)
    monkeypatch.setattr(calibrate_module, "_collect_and_plot_fulljones", fake_collect)

    outputs = calibrate_module._run_calibrate_prefect_tasks(
        payload,
        execution_config=ExecutionConfig(task_runner="sync"),
    )

    assert outputs == {"combined_solutions": file_record(tmp_path / "fulljones_solutions.h5")}
    assert events == [
        "submit-fulljones_gain_0.h5parm",
        "submit-fulljones_gain_1.h5parm",
        "result-0",
        "result-1",
        "collect",
    ]


def test_run_calibrate_flow_fails_when_expected_output_is_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="DI full-Jones h5parm"):
        run_calibrate_flow(
            calibrate_payload_from_inputs("di", _di_fulljones_input_parms(), tmp_path),
            execution_config=ExecutionConfig(task_runner="sync"),
            shell_operation_cls=NoOutputShellOperation,
        )


def test_calibrate_reference_output_fixture_matches_output_contract():
    outputs = json.loads((FIXTURE_DIR / "cwl_reference_outputs.json").read_text())

    for value in outputs["calibrate_di_fulljones"].values():
        validate_output_record(value)
