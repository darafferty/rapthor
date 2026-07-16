import json
import shlex
from pathlib import Path

import numpy as np
import pytest
from prefect.testing.utilities import prefect_test_harness

import rapthor.execution.calibrate.collection as calibrate_collection
import rapthor.execution.calibrate.flow as calibrate_module
import rapthor.execution.calibrate.prediction as calibrate_prediction
from rapthor.execution.calibrate.builders import calibrate_payload_from_inputs
from rapthor.execution.calibrate.commands import (
    PLOT_SOLUTIONS_MODULE,
    CalibrationSolveOptions,
    DrawModelOptions,
    IdgcalScreenSolveOptions,
    WscleanPredictOptions,
    build_calibration_solve_command,
    build_collect_h5parms_command,
    build_draw_model_command,
    build_idgcal_solve_phase_and_gain_command,
    build_idgcal_solve_phase_command,
    build_plot_solutions_command,
    build_wsclean_predict_command,
)
from rapthor.execution.calibrate.flow import (
    calibrate_chunk_task,
    calibrate_flow,
)
from rapthor.execution.calibrate.solves import (
    build_calibrate_chunk_command,
    initialsolutions_soltab,
)
from rapthor.execution.commands import normalize_command
from rapthor.execution.config import ExecutionConfig
from rapthor.lib.field import Field as RapthorField
from rapthor.lib.records import directory_record, file_record, validate_output_record
from rapthor.operations.calibrate.base import Calibrate
from tests.execution.conftest import run_flow_for_test

FIXTURE_DIR = Path(__file__).parent / "fixtures"
PLOT_SOLUTIONS_COMMAND_PREFIX = ["python3", "-m", PLOT_SOLUTIONS_MODULE]
PLOT_SOLUTIONS_COMMAND_NAME = " ".join(PLOT_SOLUTIONS_COMMAND_PREFIX)


def _is_plot_solutions_command(command: list[str]) -> bool:
    """Return whether command tokens invoke the plotting adapter."""
    return command[: len(PLOT_SOLUTIONS_COMMAND_PREFIX)] == PLOT_SOLUTIONS_COMMAND_PREFIX


def _plot_solutions_args(command: list[str]) -> list[str]:
    """Return plotting adapter arguments without the Python module prefix."""
    assert _is_plot_solutions_command(command)
    return command[len(PLOT_SOLUTIONS_COMMAND_PREFIX) :]


def _command_name(command: list[str]) -> str:
    """Return a readable command name for assertions."""
    if _is_plot_solutions_command(command):
        return PLOT_SOLUTIONS_COMMAND_NAME
    return command[0]


def _command_names(commands: list[list[str]]) -> list[str]:
    """Return readable command names for assertion lists."""
    return [_command_name(command) for command in commands]


def _calibration_command_names_after_collect_split(
    *plot_counts: int,
    prefix=None,
) -> list[str]:
    """Return expected command names after solve-slot collection is task-split."""
    names = list(prefix or []) + ["DP3", "DP3"]
    names.extend(["H5parm_collector.py"] * len(plot_counts))
    for plot_count in plot_counts:
        names.extend([PLOT_SOLUTIONS_COMMAND_NAME] * plot_count)
    return names


def _add_explicit_solve_metadata(input_parms):
    """Add operation-style solve metadata to hand-built payload fixtures."""
    medium_count = 0
    for step in input_parms["dp3_steps"].strip("[]").split(","):
        step = step.strip()
        if not step.startswith("solve"):
            continue

        slot = int(step.removeprefix("solve"))
        output_name = input_parms[f"output_solve{slot}_h5parm"][0]
        solve_mode = input_parms[f"solve{slot}_mode"]
        if output_name.startswith("fulljones_gain_") and solve_mode == "fulljones":
            solve_type = "full_jones"
            solution_label = "fulljones"
            medium_index = None
        elif output_name.startswith("fast_phase_") and solve_mode == "scalarphase":
            solve_type = "fast_phase"
            solution_label = "fast"
            medium_index = None
        elif (
            output_name.startswith(("medium1_phase_", "medium2_phase_"))
            and solve_mode == "scalarphase"
        ):
            solve_type = "medium_phase"
            medium_count += 1
            medium_index = medium_count
            solution_label = f"medium{medium_count}"
        elif output_name.startswith(("slow_gain_", "slow_gains_di_")) and solve_mode == "diagonal":
            solve_type = "slow_gains"
            solution_label = "slow"
            medium_index = None
        else:
            raise AssertionError(f"Unhandled solve fixture: {step} {output_name} {solve_mode}")

        input_parms[f"solve{slot}_type"] = solve_type
        input_parms[f"solve{slot}_solution_label"] = solution_label
        input_parms[f"solve{slot}_medium_index"] = medium_index
        input_parms.setdefault(f"solve{slot}_initialsolutions_h5parm", None)
    return input_parms


@pytest.fixture
def fake_calibrate_shell_operation_cls():
    class FakeCalibrateShellOperation:
        instances = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.instances.append(self)

        def run(self):
            tokens = shlex.split(self.kwargs["commands"][0])
            cwd = Path(self.kwargs["working_dir"])
            if tokens[0] == "wsclean":
                if "-draw-model" in tokens:
                    root = tokens[tokens.index("-name") + 1]
                    numterms = int(tokens[tokens.index("-draw-spectral-terms") + 1])
                    for index in range(numterms):
                        (cwd / f"{root}-term-{index}.fits").write_text("model")
                elif "-predict" not in tokens:
                    raise AssertionError(f"Unexpected WSClean command: {tokens}")
                return "OK"
            if tokens[0] == "DP3":
                for token in tokens:
                    if token.startswith("solve") and ".h5parm=" in token:
                        output_path = cwd / token.split("=", 1)[1]
                        output_path.write_text("h5parm")
                return "OK"
            if tokens[:2] == ["H5parm_collector.py", "-c"]:
                output_name = next(
                    token.split("=", 1)[1] for token in tokens if token.startswith("--outh5parm=")
                )
                output_path = cwd / output_name
                output_path.write_text("collected")
                return "OK"
            if _is_plot_solutions_command(tokens):
                plot_args = _plot_solutions_args(tokens)
                root = next(
                    (token.split("=", 1)[1] for token in plot_args if token.startswith("--root=")),
                    f"{plot_args[1]}_",
                )
                output_path = cwd / f"{root}solutions.png"
                output_path.write_text("plot")
                return "OK"
            raise AssertionError(f"Unexpected command: {tokens[0]}")

    return FakeCalibrateShellOperation


@pytest.fixture(autouse=True)
def fake_direct_calibrate_helpers(monkeypatch):
    calls = {
        "adjust_h5parm_sources": [],
        "collect_screen_h5parms": [],
        "combine_h5parms": [],
        "make_region_file": [],
        "patch_names_from_region": [],
        "frequency_chunks_for_ms": [],
        "process_gains": [],
    }

    def fake_adjust_h5parm_source_coordinates(skymodel, h5parm_file, solset_name="sol000"):
        calls["adjust_h5parm_sources"].append(
            {
                "skymodel": skymodel,
                "h5parm_file": h5parm_file,
                "solset_name": solset_name,
            }
        )
        Path(h5parm_file).write_text("adjusted")

    def fake_collect_screen_h5parms(h5parm_files, output_h5parm, overwrite=False):
        calls["collect_screen_h5parms"].append(
            {
                "h5parm_files": list(h5parm_files),
                "output_h5parm": output_h5parm,
                "overwrite": overwrite,
            }
        )
        Path(output_h5parm).write_text("screens")

    def fake_combine_h5parms(
        h5parm1,
        h5parm2,
        outh5parm,
        mode,
        solset1="sol000",
        solset2="sol000",
        reweight=False,
        cal_names=None,
        cal_fluxes=None,
    ):
        calls["combine_h5parms"].append(
            {
                "h5parm1": h5parm1,
                "h5parm2": h5parm2,
                "outh5parm": outh5parm,
                "mode": mode,
                "solset1": solset1,
                "solset2": solset2,
                "reweight": reweight,
                "cal_names": cal_names,
                "cal_fluxes": cal_fluxes,
            }
        )
        Path(outh5parm).write_text("combined")

    def fake_make_ds9_region_from_skymodel(
        skymodel,
        ra_mid,
        dec_mid,
        width_ra,
        width_dec,
        region_file,
        *,
        enclose_names=True,
    ):
        calls["make_region_file"].append(
            {
                "skymodel": skymodel,
                "ra_mid": ra_mid,
                "dec_mid": dec_mid,
                "width_ra": width_ra,
                "width_dec": width_dec,
                "region_file": region_file,
                "enclose_names": enclose_names,
            }
        )
        Path(region_file).write_text("region")

    def fake_process_gain_solutions(h5parmfile, **kwargs):
        calls["process_gains"].append({"h5parmfile": h5parmfile, **kwargs})
        Path(h5parmfile).write_text("processed")

    def fake_patch_names_from_region(region_file):
        calls["patch_names_from_region"].append(region_file)
        return ["patch1", "patch2"]

    def fake_frequency_chunks_for_ms(
        msin,
        fallback_frequency_bandwidth,
        *,
        max_bandwidth_hz,
    ):
        calls["frequency_chunks_for_ms"].append(
            {
                "msin": msin,
                "fallback_frequency_bandwidth": list(fallback_frequency_bandwidth),
                "max_bandwidth_hz": max_bandwidth_hz,
            }
        )
        return [
            {
                "frequency_bandwidth": [150000000.0, 1000000.0],
                "channel_range": (0, 3),
            }
        ]

    monkeypatch.setattr(
        calibrate_collection,
        "adjust_h5parm_source_coordinates",
        fake_adjust_h5parm_source_coordinates,
    )
    monkeypatch.setattr(
        calibrate_collection,
        "collect_screen_h5parms",
        fake_collect_screen_h5parms,
    )
    monkeypatch.setattr(calibrate_collection, "combine_h5parms", fake_combine_h5parms)
    monkeypatch.setattr(
        calibrate_prediction,
        "make_ds9_region_from_skymodel",
        fake_make_ds9_region_from_skymodel,
    )
    monkeypatch.setattr(
        calibrate_collection,
        "process_gain_solutions",
        fake_process_gain_solutions,
    )
    monkeypatch.setattr(
        calibrate_prediction,
        "_patch_names_from_region",
        fake_patch_names_from_region,
    )
    monkeypatch.setattr(
        calibrate_prediction,
        "_frequency_chunks_for_ms",
        fake_frequency_chunks_for_ms,
    )
    return calls


class NoOutputShellOperation:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def run(self):
        return "OK"


class FailingShellOperation:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def run(self):
        raise RuntimeError("calibrate failed")


class CalibrateObservationStub:
    channels_are_regular = True


class CalibrateFieldStub:
    solution_cycle_number = RapthorField.solution_cycle_number

    def __init__(self, tmp_path):
        self.parset = {
            "dir_working": str(tmp_path / "working"),
            "cluster_specific": {
                "debug_workflow": False,
                "keep_temporary_files": False,
                "max_nodes": 1,
                "batch_system": "single_machine",
                "cpus_per_task": 1,
                "mem_per_node_gb": 0,
                "dir_local": None,
                "local_scratch_dir": None,
                "global_scratch_dir": None,
                "use_container": False,
                "container_type": "docker",
                "max_cores": 1,
                "max_threads": 4,
                "prefect_task_runner": "sync",
            },
        }
        self.observations = [CalibrateObservationStub(), CalibrateObservationStub()]
        self.ntimechunks = 2
        self.calibration_diagnostics = []
        self.calibration_skymodel_file = str(tmp_path / "calibration.skymodel")
        Path(self.calibration_skymodel_file).write_text("skymodel")
        self.ra = 123.0
        self.dec = 45.0
        self.sector_bounds_deg = "[0,0,1,1]"
        self.sector_bounds_mid_deg = "[0.5,0.5]"
        self.smoothnessconstraint_fulljones = 1.5
        self.fast_smoothnessconstraint = 1200000.0
        self.medium_smoothnessconstraint = 2400000.0
        self.slow_smoothnessconstraint = 3600000.0
        self.fast_smoothnessrefdistance = 2500.0
        self.medium_smoothnessrefdistance = 3500.0
        self.max_normalization_delta = 0.3
        self.scale_normalization_delta = True
        self.llssolver = "qr"
        self.maxiter = 50
        self.propagatesolutions = True
        self.solveralgorithm = "directionsolve"
        self.onebeamperpatch = False
        self.stepsize = 0.2
        self.stepsigma = 0.0
        self.tolerance = 0.0001
        self.solve_min_uv_lambda = 80.0
        self.parallelbaselines = False
        self.sagecalpredict = False
        self.solverlbfgs_dof = 200.0
        self.solverlbfgs_iter = 4
        self.solverlbfgs_minibatches = 1
        self.correct_smearing_in_calibration = True
        self.fast_datause = "full"
        self.medium_datause = "full"
        self.slow_datause = "full"
        self.data_colname = "DATA"
        self.calibrator_patch_names = ["patch1"]
        self.calibrator_fluxes = [10.0]
        self.antenna = "HBA"
        self.stations = []
        self.calibrate_bda_timebase = 0
        self.calibrate_bda_frequencybase = 0
        self.calibration_strategy = {"dd": ["fast_phase", "medium_phase"], "di": ["full_jones"]}
        self._calibration_strategy_defaulted = False
        self.apply_diagonal_solutions = False
        self.apply_amplitudes = False
        self.generate_screens = False
        self.use_image_based_predict = False
        self.use_wsclean_predict = False
        self.wsclean_predict_bw = 2.0e6
        self.apply_normalizations = False
        self.normalize_h5parm = None
        self.fulljones_h5parm_filename = None
        self.h5parm_filename = None
        self.dd_h5parm_filename = None
        self.di_h5parm_filename = None
        self.fast_phases_h5parm_filename = None
        self.medium1_phases_h5parm_filename = None
        self.medium2_phases_h5parm_filename = None
        self.slow_gains_h5parm_filename = None
        self.di_fast_phases_h5parm_filename = None
        self.di_medium1_phases_h5parm_filename = None
        self.di_medium2_phases_h5parm_filename = None
        self.di_slow_gains_h5parm_filename = None
        self.scan_h5parms_calls = 0
        self._obs_parameters = {
            "predict_di_output_filename": ["obs_0_predict.ms", "obs_1_predict.ms"],
            "timechunk_filename": ["dd_obs_0.ms", "dd_obs_1.ms"],
            "starttime": ["50000.0", "50010.0"],
            "ntimes": [10, 12],
            "bda_maxinterval": [8.0, 9.0],
            "bda_minchannels": [1, 1],
            "solint_fulljones_timestep": [5, 6],
            "solint_fulljones_freqstep": [2, 3],
            "solint_fast_timestep": [5, 6],
            "solint_fast_freqstep": [2, 3],
            "solint_slow_timestep": [11, 12],
            "solint_slow_freqstep": [7, 8],
            "solint_medium_timestep": [9, 10],
            "solint_medium_freqstep": [5, 6],
            "fast_solutions_per_direction": [[1], [1]],
            "medium_solutions_per_direction": [[1], [1]],
            "slow_solutions_per_direction": [[1], [1]],
            "fast_smoothness_dd_factors": [[1.0], [1.0]],
            "medium_smoothness_dd_factors": [[1.0], [1.0]],
            "slow_smoothness_dd_factors": [[1.0], [1.0]],
            "fast_smoothnessreffrequency": [150000000.0, 151000000.0],
            "medium_smoothnessreffrequency": [152000000.0, 153000000.0],
        }

    def set_obs_parameters(self):
        return None

    def get_obs_parameters(self, name):
        return self._obs_parameters[name]

    def scan_h5parms(self):
        self.scan_h5parms_calls += 1


def _expected_di_fulljones_operation_outputs(operation):
    pipeline_dir = Path(operation.pipeline_working_dir)
    solution = file_record(pipeline_dir / "fulljones_solutions.h5")
    return {
        "combined_solutions": solution,
        "fulljones_solutions": solution,
        "fulljones_phase_plots": [file_record(pipeline_dir / "fulljones_phase_solutions.png")],
    }


def _expected_di_scalar_phase_operation_outputs(operation):
    pipeline_dir = Path(operation.pipeline_working_dir)
    return {
        "combined_solutions": file_record(pipeline_dir / "combined_solve1_solve2_di.h5parm"),
        "fast_phase_solutions": file_record(pipeline_dir / "fast_phases_di.h5parm"),
        "medium1_phase_solutions": file_record(pipeline_dir / "medium1_phases_di.h5parm"),
        "fast_phase_plots": [file_record(pipeline_dir / "phase_solutions.png")],
        "medium1_phase_plots": [file_record(pipeline_dir / "medium1_phase_solutions.png")],
    }


def _expected_dd_fast_medium_operation_outputs(operation):
    pipeline_dir = Path(operation.pipeline_working_dir)
    return {
        "combined_solutions": file_record(pipeline_dir / "combined_fast_medium1_phases.h5parm"),
        "fast_phase_solutions": file_record(pipeline_dir / "fast_phases.h5parm"),
        "medium1_phase_solutions": file_record(pipeline_dir / "medium1_phases.h5parm"),
        "fast_phase_plots": [file_record(pipeline_dir / "phase_solutions.png")],
        "medium1_phase_plots": [file_record(pipeline_dir / "medium1_phase_solutions.png")],
    }


def _expected_dd_slow_operation_outputs(operation):
    pipeline_dir = Path(operation.pipeline_working_dir)
    return {
        "combined_solutions": file_record(pipeline_dir / "combined_solutions.h5"),
        "fast_phase_solutions": file_record(pipeline_dir / "fast_phases.h5parm"),
        "medium1_phase_solutions": file_record(pipeline_dir / "medium1_phases.h5parm"),
        "slow_gain_solutions": file_record(pipeline_dir / "slow_gains.h5parm"),
        "medium2_phase_solutions": file_record(pipeline_dir / "medium2_phases.h5parm"),
        "fast_phase_plots": [file_record(pipeline_dir / "phase_solutions.png")],
        "medium1_phase_plots": [file_record(pipeline_dir / "medium1_phase_solutions.png")],
        "slow_phase_plots": [file_record(pipeline_dir / "slow_phase_solutions.png")],
        "slow_amp_plots": [file_record(pipeline_dir / "slow_amplitude_solutions.png")],
        "medium2_phase_plots": [file_record(pipeline_dir / "medium2_phase_solutions.png")],
    }


def _expected_dd_screen_operation_outputs(operation):
    pipeline_dir = Path(operation.pipeline_working_dir)
    return {"combined_solutions": file_record(pipeline_dir / "combined_solutions.h5")}


def _patch_dd_model_metadata(monkeypatch):
    monkeypatch.setattr(
        "rapthor.operations.calibrate.base.misc.get_max_spectral_terms",
        lambda *args, **kwargs: 1,
    )
    monkeypatch.setattr(
        Calibrate,
        "_get_model_image_parameters",
        lambda self: (
            (150000000.0, 1000000.0),
            ("12:00:00.0", "+45.00.00.0"),
            (1024, 1024),
            0.001,
        ),
    )


def _configure_dd_preapply_products(field, tmp_path):
    solutions_dir = tmp_path / "input_solutions"
    solutions_dir.mkdir()
    di_h5parm = solutions_dir / "di-solutions.h5"
    fulljones_h5parm = solutions_dir / "fulljones-solutions.h5"
    normalize_h5parm = solutions_dir / "normalize-solutions.h5"
    for path in (di_h5parm, fulljones_h5parm, normalize_h5parm):
        path.write_text("input")
    field.di_h5parm_filename = str(di_h5parm)
    field.fulljones_h5parm_filename = str(fulljones_h5parm)
    field.normalize_h5parm = str(normalize_h5parm)
    field.apply_amplitudes = True
    field.apply_normalizations = True
    return di_h5parm, fulljones_h5parm, normalize_h5parm


def _configure_dd_multidirection(field):
    field.calibrator_patch_names = ["patch1", "patch2"]
    field.calibrator_fluxes = [10.0, 5.0]
    field._obs_parameters.update(
        {
            "fast_solutions_per_direction": [[1, 1], [1, 1]],
            "medium_solutions_per_direction": [[1, 1], [1, 1]],
            "slow_solutions_per_direction": [[1, 1], [1, 1]],
            "fast_smoothness_dd_factors": [[1.0, 2.0], [1.5, 2.5]],
            "medium_smoothness_dd_factors": [[2.0, 3.0], [2.5, 3.5]],
            "slow_smoothness_dd_factors": [[3.0, 4.0], [3.5, 4.5]],
        }
    )


def _command_tokens(shell_operation_cls):
    return [
        shlex.split(instance.kwargs["commands"][0]) for instance in shell_operation_cls.instances
    ]


def _command_arguments(tokens):
    return {
        key: value for key, _, value in (token.partition("=") for token in tokens if "=" in token)
    }


def _materialize_calibrate_operation_outputs(value):
    if isinstance(value, dict) and "class" not in value:
        for item in value.values():
            _materialize_calibrate_operation_outputs(item)
        return
    if isinstance(value, list):
        for item in value:
            _materialize_calibrate_operation_outputs(item)
        return
    path = Path(value["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("calibrate")


def _di_fulljones_input_parms():
    return _add_explicit_solve_metadata(
        {
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
            "max_normalization_delta": 0.3,
        }
    )


def _di_scalar_phase_input_parms():
    input_parms = _di_fulljones_input_parms()
    input_parms.update(
        {
            "dp3_steps": "[solve1,solve2]",
            "output_solve1_h5parm": [
                "fast_phase_di_0.h5parm",
                "fast_phase_di_1.h5parm",
            ],
            "output_solve2_h5parm": [
                "medium1_phase_di_0.h5parm",
                "medium1_phase_di_1.h5parm",
            ],
            "collected_solve1_h5parm": "fast_phases_di.h5parm",
            "collected_solve2_h5parm": "medium1_phases_di.h5parm",
            "combined_phase_1_2_h5parm": "combined_solve1_solve2_di.h5parm",
            "solint_solve1_timestep": [5, 6],
            "solint_solve2_timestep": [7, 8],
            "solint_solve1_freqstep": [2, 3],
            "solint_solve2_freqstep": [4, 5],
            "solve1_mode": "scalarphase",
            "solve2_mode": "scalarphase",
            "solve1_solutions_per_direction": [None, None],
            "solve2_solutions_per_direction": [None, None],
            "solve1_smoothness_dd_factors": [None, None],
            "solve2_smoothness_dd_factors": [None, None],
            "solve1_smoothnessconstraint": 0,
            "solve2_smoothnessconstraint": 0,
            "solve1_smoothnessreffrequency": [0, 0],
            "solve2_smoothnessreffrequency": [0, 0],
            "solve1_smoothnessrefdistance": None,
            "solve2_smoothnessrefdistance": None,
            "solve1_antennaconstraint": "[]",
            "solve2_antennaconstraint": "[]",
            "calibrator_patch_names": [],
            "calibrator_fluxes": [],
        }
    )
    return _add_explicit_solve_metadata(input_parms)


def _di_fast_phase_input_parms():
    input_parms = _di_scalar_phase_input_parms()
    input_parms.update(
        {
            "dp3_steps": "[solve1]",
            "output_solve1_h5parm": [
                "fast_phase_di_0.h5parm",
                "fast_phase_di_1.h5parm",
            ],
            "collected_solve1_h5parm": "fast_phases_di.h5parm",
        }
    )
    return _add_explicit_solve_metadata(input_parms)


def _di_slow_input_parms():
    input_parms = _di_scalar_phase_input_parms()
    input_parms.update(
        {
            "dp3_steps": "[solve1]",
            "output_solve1_h5parm": [
                "slow_gains_di_0.h5parm",
                "slow_gains_di_1.h5parm",
            ],
            "collected_solve1_h5parm": "slow_gains_di.h5parm",
            "solint_solve1_timestep": [11, 12],
            "solint_solve1_freqstep": [7, 8],
            "solve1_mode": "diagonal",
            "max_normalization_delta": 0.25,
            "scale_normalization_delta": "False",
            "phase_center_ra": 123.0,
            "phase_center_dec": 45.0,
        }
    )
    return _add_explicit_solve_metadata(input_parms)


def _di_phase_slow_input_parms():
    input_parms = _di_scalar_phase_input_parms()
    input_parms.update(
        {
            "dp3_steps": "[solve1,solve2,solve3]",
            "output_solve3_h5parm": [
                "slow_gains_di_0.h5parm",
                "slow_gains_di_1.h5parm",
            ],
            "collected_solve3_h5parm": "slow_gains_di.h5parm",
            "combined_h5parms": "combined_di_solutions.h5parm",
            "solint_solve3_timestep": [11, 12],
            "solint_solve3_freqstep": [7, 8],
            "solve3_mode": "diagonal",
            "solve3_solutions_per_direction": [None, None],
            "solve3_smoothness_dd_factors": [None, None],
            "solve3_smoothnessconstraint": 0,
            "solve3_smoothnessreffrequency": [0, 0],
            "solve3_smoothnessrefdistance": None,
            "solve3_antennaconstraint": "[]",
            "max_normalization_delta": 0.25,
            "scale_normalization_delta": "False",
            "phase_center_ra": 123.0,
            "phase_center_dec": 45.0,
        }
    )
    return _add_explicit_solve_metadata(input_parms)


def _di_scalar_slow_fulljones_input_parms():
    input_parms = _di_phase_slow_input_parms()
    input_parms.update(
        {
            "dp3_steps": "[solve1,solve2,solve3,solve4]",
            "output_solve4_h5parm": [
                "fulljones_gain_0.h5parm",
                "fulljones_gain_1.h5parm",
            ],
            "collected_solve4_h5parm": "fulljones_solutions.h5",
            "solint_solve4_timestep": [13, 14],
            "solint_solve4_freqstep": [9, 10],
            "solve4_mode": "fulljones",
            "solve4_solutions_per_direction": [None, None],
            "solve4_smoothness_dd_factors": [None, None],
            "solve4_smoothnessconstraint": 1.5,
            "solve4_smoothnessreffrequency": [0, 0],
            "solve4_smoothnessrefdistance": None,
            "solve4_antennaconstraint": "[]",
        }
    )
    return _add_explicit_solve_metadata(input_parms)


def _dd_fast_phase_input_parms():
    input_parms = _di_fulljones_input_parms()
    input_parms.update(
        {
            "timechunk_filename": [
                directory_record("/data/dd_obs_0.ms"),
                directory_record("/data/dd_obs_1.ms"),
            ],
            "modeldatacolumn": None,
            "calibration_skymodel_file": file_record("/data/calibration.skymodel"),
            "solve_directions": ["patch1", "patch2"],
            "dp3_steps": "[solve1]",
            "output_solve1_h5parm": [
                "fast_phase_0.h5parm",
                "fast_phase_1.h5parm",
            ],
            "collected_solve1_h5parm": "fast_phases.h5parm",
            "solint_solve1_timestep": [3, 4],
            "solint_solve1_freqstep": [1, 2],
            "solve1_mode": "scalarphase",
            "solve1_solutions_per_direction": [[1, 1], [1, 1]],
            "solve1_smoothness_dd_factors": [[1.0, 2.0], [1.5, 2.5]],
            "solve1_smoothnessconstraint": 1200000.0,
            "solve1_smoothnessreffrequency": [150000000.0, 151000000.0],
            "solve1_smoothnessrefdistance": 2500.0,
            "solve1_antennaconstraint": "[[CS001HBA0,CS002HBA0]]",
            "solve1_datause": "full",
            "solve1_initialsolutions_h5parm": None,
            "applycal_steps": None,
            "applycal_h5parm": None,
            "fulljones_h5parm": None,
            "normalize_h5parm": None,
            "bda_timebase": 0.0,
            "bda_frequencybase": 0.0,
            "bda_maxinterval": [8.0, 9.0],
            "bda_minchannels": [1, 1],
            "onebeamperpatch": True,
            "parallelbaselines": False,
            "sagecalpredict": False,
            "calibrator_patch_names": ["patch1", "patch2"],
            "calibrator_fluxes": [10.0, 5.0],
        }
    )
    return _add_explicit_solve_metadata(input_parms)


def _dd_slow_input_parms():
    input_parms = _dd_fast_phase_input_parms()
    input_parms.update(
        {
            "dp3_steps": "[solve1]",
            "output_solve1_h5parm": [
                "slow_gain_0.h5parm",
                "slow_gain_1.h5parm",
            ],
            "collected_solve1_h5parm": "slow_gains.h5parm",
            "solint_solve1_timestep": [11, 12],
            "solint_solve1_freqstep": [7, 8],
            "solve1_mode": "diagonal",
            "solve1_smoothnessconstraint": 3600000.0,
            "solve1_smoothnessreffrequency": [0, 0],
            "solve1_smoothnessrefdistance": None,
            "max_normalization_delta": 0.25,
            "scale_normalization_delta": "False",
            "phase_center_ra": 123.0,
            "phase_center_dec": 45.0,
        }
    )
    return _add_explicit_solve_metadata(input_parms)


def _dd_slow_medium_input_parms():
    input_parms = _dd_slow_input_parms()
    input_parms.update(
        {
            "dp3_steps": "[solve1,solve2]",
            "output_solve2_h5parm": [
                "medium1_phase_0.h5parm",
                "medium1_phase_1.h5parm",
            ],
            "collected_solve2_h5parm": "medium1_phases.h5parm",
            "combined_h5parms": "combined_solutions.h5",
            "solution_combine_mode": "p1p2a2_scalar",
            "solint_solve2_timestep": [9, 10],
            "solint_solve2_freqstep": [5, 6],
            "solve1_antennaconstraint": "[]",
            "solve2_mode": "scalarphase",
            "solve2_solutions_per_direction": [[1, 1], [1, 1]],
            "solve2_smoothness_dd_factors": [[2.0, 3.0], [2.5, 3.5]],
            "solve2_smoothnessconstraint": 2400000.0,
            "solve2_smoothnessreffrequency": [152000000.0, 153000000.0],
            "solve2_smoothnessrefdistance": 3500.0,
            "solve2_antennaconstraint": "[[CS001HBA0,CS002HBA0]]",
            "solve2_datause": "full",
            "solve2_initialsolutions_h5parm": None,
            "has_slow_gain_solve": True,
        }
    )
    return _add_explicit_solve_metadata(input_parms)


def _dd_fast_medium_input_parms():
    input_parms = _dd_fast_phase_input_parms()
    input_parms.update(
        {
            "dp3_steps": "[solve1,solve2]",
            "output_solve2_h5parm": [
                "medium1_phase_0.h5parm",
                "medium1_phase_1.h5parm",
            ],
            "collected_solve2_h5parm": "medium1_phases.h5parm",
            "combined_phase_1_2_h5parm": "combined_fast_medium1_phases.h5parm",
            "solint_solve2_timestep": [9, 10],
            "solint_solve2_freqstep": [5, 6],
            "solve2_mode": "scalarphase",
            "solve2_solutions_per_direction": [[1, 1], [1, 1]],
            "solve2_smoothness_dd_factors": [[2.0, 3.0], [2.5, 3.5]],
            "solve2_smoothnessconstraint": 2400000.0,
            "solve2_smoothnessreffrequency": [152000000.0, 153000000.0],
            "solve2_smoothnessrefdistance": 3500.0,
            "solve2_antennaconstraint": "[]",
            "solve2_datause": "full",
            "solve2_initialsolutions_h5parm": None,
            "has_slow_gain_solve": False,
        }
    )
    return _add_explicit_solve_metadata(input_parms)


def _dd_medium_fast_input_parms():
    input_parms = _dd_fast_medium_input_parms()
    input_parms.update(
        {
            "output_solve1_h5parm": [
                "medium1_phase_0.h5parm",
                "medium1_phase_1.h5parm",
            ],
            "output_solve2_h5parm": [
                "fast_phase_0.h5parm",
                "fast_phase_1.h5parm",
            ],
            "collected_solve1_h5parm": "medium1_phases.h5parm",
            "collected_solve2_h5parm": "fast_phases.h5parm",
            "solint_solve1_timestep": [9, 10],
            "solint_solve1_freqstep": [5, 6],
            "solint_solve2_timestep": [3, 4],
            "solint_solve2_freqstep": [1, 2],
            "solve1_solutions_per_direction": [[1, 1], [1, 1]],
            "solve2_solutions_per_direction": [[1, 1], [1, 1]],
            "solve1_smoothness_dd_factors": [[2.0, 3.0], [2.5, 3.5]],
            "solve2_smoothness_dd_factors": [[1.0, 2.0], [1.5, 2.5]],
            "solve1_smoothnessconstraint": 2400000.0,
            "solve2_smoothnessconstraint": 1200000.0,
            "solve1_smoothnessreffrequency": [152000000.0, 153000000.0],
            "solve2_smoothnessreffrequency": [150000000.0, 151000000.0],
            "solve1_smoothnessrefdistance": 3500.0,
            "solve2_smoothnessrefdistance": 2500.0,
            "solve1_antennaconstraint": "[]",
            "solve2_antennaconstraint": "[[CS001HBA0,CS002HBA0]]",
        }
    )
    return _add_explicit_solve_metadata(input_parms)


def _dd_preapply_input_parms():
    input_parms = _dd_fast_medium_input_parms()
    input_parms.update(
        {
            "dp3_steps": "[applycal,solve1,solve2]",
            "applycal_steps": "[fastphase,slowgain,fulljones,normalization]",
            "applycal_h5parm": file_record("/solutions/di_solutions.h5"),
            "fulljones_h5parm": file_record("/solutions/fulljones_solutions.h5"),
            "normalize_h5parm": file_record("/solutions/normalize_solutions.h5"),
        }
    )
    return _add_explicit_solve_metadata(input_parms)


def _dd_image_predict_input_parms():
    input_parms = _dd_fast_medium_input_parms()
    input_parms.update(
        {
            "dp3_steps": "[predict,applybeam,solve1,solve2]",
            "model_image_root": "calibration_model",
            "model_image_ra_dec": ["12:00:00.0", "+45.00.00.0"],
            "model_image_imsize": [1024, 1024],
            "model_image_cellsize": 0.001,
            "model_image_frequency_bandwidth": [150000000.0, 1000000.0],
            "num_spectral_terms": 2,
            "ra_mid": 123.0,
            "dec_mid": 45.0,
            "facet_region_width_ra": 2.0,
            "facet_region_width_dec": 2.5,
            "facet_region_file": "field_facets_ds9.reg",
        }
    )
    return _add_explicit_solve_metadata(input_parms)


def _dd_wsclean_predict_input_parms():
    input_parms = _dd_image_predict_input_parms()
    input_parms.update(
        {
            "dp3_steps": "[solve1,solve2]",
            "use_wsclean_predict": True,
            "predict_facet_region_file": "predict_field_facets_ds9.reg",
        }
    )
    return _add_explicit_solve_metadata(input_parms)


def _use_local_timechunk_dirs(input_parms, tmp_path):
    local_paths = []
    for index in range(len(input_parms["timechunk_filename"])):
        path = tmp_path / f"input_chunk_{index}.ms"
        path.mkdir()
        local_paths.append(directory_record(path))
    input_parms["timechunk_filename"] = local_paths
    return input_parms


def _dd_image_predict_preapply_input_parms(normalize_h5parm="/solutions/normalize_solutions.h5"):
    input_parms = _dd_image_predict_input_parms()
    input_parms.update(
        {
            "dp3_steps": "[predict,applybeam,applycal,solve1,solve2]",
            "applycal_steps": "[fastphase,normalization]",
            "applycal_h5parm": file_record("/solutions/di_solutions.h5"),
            "normalize_h5parm": file_record(normalize_h5parm),
        }
    )
    return _add_explicit_solve_metadata(input_parms)


def _dd_screen_input_parms(has_slow_gain_solve=False):
    input_parms = _dd_image_predict_input_parms()
    input_parms.update(
        {
            "generate_screens": True,
            "output_idgcal_h5parm": ["idgcal_0", "idgcal_1"],
            "combined_h5parms": "combined_solutions.h5",
            "idgcal_antennaconstraint": "[]",
            "has_slow_gain_solve": has_slow_gain_solve,
        }
    )
    if has_slow_gain_solve:
        input_parms["solint_slow_timestep"] = [11, 12]
    return _add_explicit_solve_metadata(input_parms)


def _dd_with_slow_input_parms():
    input_parms = _dd_fast_medium_input_parms()
    input_parms.update(
        {
            "dp3_steps": "[solve1,solve2,solve3,solve4]",
            "output_solve3_h5parm": [
                "slow_gain_0.h5parm",
                "slow_gain_1.h5parm",
            ],
            "output_solve4_h5parm": [
                "medium2_phase_0.h5parm",
                "medium2_phase_1.h5parm",
            ],
            "collected_solve3_h5parm": "slow_gains.h5parm",
            "collected_solve4_h5parm": "medium2_phases.h5parm",
            "combined_phase_1_2_3_h5parm": ("combined_fast_medium1_medium2_phases.h5parm"),
            "combined_h5parms": "combined_solutions.h5",
            "solution_combine_mode": "p1p2a2_scalar",
            "solint_solve3_timestep": [11, 12],
            "solint_solve3_freqstep": [7, 8],
            "solint_solve4_timestep": [13, 14],
            "solint_solve4_freqstep": [9, 10],
            "solve3_mode": "diagonal",
            "solve4_mode": "scalarphase",
            "solve3_solutions_per_direction": [[1, 1], [1, 1]],
            "solve4_solutions_per_direction": [[1, 1], [1, 1]],
            "solve3_smoothness_dd_factors": [[3.0, 4.0], [3.5, 4.5]],
            "solve4_smoothness_dd_factors": [[4.0, 5.0], [4.5, 5.5]],
            "solve3_smoothnessconstraint": 3600000.0,
            "solve4_smoothnessconstraint": 4800000.0,
            "solve3_smoothnessreffrequency": [0, 0],
            "solve4_smoothnessreffrequency": [154000000.0, 155000000.0],
            "solve3_smoothnessrefdistance": None,
            "solve4_smoothnessrefdistance": 4500.0,
            "solve3_antennaconstraint": "[]",
            "solve4_antennaconstraint": "[[CS001HBA0,CS002HBA0]]",
            "solve3_datause": "full",
            "solve4_datause": "full",
            "solve3_initialsolutions_h5parm": None,
            "solve4_initialsolutions_h5parm": None,
            "has_slow_gain_solve": True,
            "max_normalization_delta": 0.25,
            "scale_normalization_delta": "False",
            "phase_center_ra": 123.0,
            "phase_center_dec": 45.0,
        }
    )
    return _add_explicit_solve_metadata(input_parms)


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
            "initialsolutions_soltab": "[amplitude000,phase000]",
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


def _di_scalar_phase_solve_slots():
    return [
        {
            "slot": 1,
            "h5parm": "fast_phase_di_0.h5parm",
            "solint": 5,
            "mode": "scalarphase",
            "nchan": 2,
            "solutions_per_direction": None,
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
            "smoothness_dd_factors": None,
            "smoothnessconstraint": 0,
            "smoothnessreffrequency": 0,
            "smoothnessrefdistance": None,
            "antennaconstraint": "[]",
            "correctfreqsmearing": False,
            "correcttimesmearing": True,
            "keepmodel": "True",
        },
        {
            "slot": 2,
            "h5parm": "medium1_phase_di_0.h5parm",
            "solint": 7,
            "mode": "scalarphase",
            "nchan": 4,
            "solutions_per_direction": None,
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
            "smoothness_dd_factors": None,
            "smoothnessconstraint": 0,
            "smoothnessreffrequency": 0,
            "smoothnessrefdistance": None,
            "antennaconstraint": "[]",
            "modeldatacolumns": "[MODEL_DATA]",
        },
    ]


def _dd_fast_phase_solve_slots():
    return [
        {
            "slot": 1,
            "h5parm": "fast_phase_0.h5parm",
            "solint": 3,
            "mode": "scalarphase",
            "nchan": 1,
            "solutions_per_direction": [1, 1],
            "llssolver": "qr",
            "maxiter": 50,
            "propagatesolutions": True,
            "initialsolutions_soltab": "[phase000]",
            "solveralgorithm": "directionsolve",
            "solverlbfgs_dof": 200.0,
            "solverlbfgs_iter": 4,
            "solverlbfgs_minibatches": 1,
            "datause": "full",
            "stepsize": 0.2,
            "stepsigma": 0.0,
            "tolerance": 0.0001,
            "uvlambdamin": 80.0,
            "smoothness_dd_factors": [1.0, 2.0],
            "smoothnessconstraint": 1200000.0,
            "smoothnessreffrequency": 150000000.0,
            "smoothnessrefdistance": 2500.0,
            "antennaconstraint": "[[CS001HBA0,CS002HBA0]]",
            "correctfreqsmearing": False,
            "correcttimesmearing": True,
            "keepmodel": "True",
        }
    ]


def _dd_fast_medium_solve_slots():
    return [
        *_dd_fast_phase_solve_slots(),
        {
            "slot": 2,
            "h5parm": "medium1_phase_0.h5parm",
            "solint": 9,
            "mode": "scalarphase",
            "nchan": 5,
            "solutions_per_direction": [1, 1],
            "llssolver": "qr",
            "maxiter": 50,
            "propagatesolutions": True,
            "initialsolutions_soltab": "[phase000]",
            "solveralgorithm": "directionsolve",
            "solverlbfgs_dof": 200.0,
            "solverlbfgs_iter": 4,
            "solverlbfgs_minibatches": 1,
            "datause": "full",
            "stepsize": 0.2,
            "stepsigma": 0.0,
            "tolerance": 0.0001,
            "uvlambdamin": 80.0,
            "smoothness_dd_factors": [2.0, 3.0],
            "smoothnessconstraint": 2400000.0,
            "smoothnessreffrequency": 152000000.0,
            "smoothnessrefdistance": 3500.0,
            "antennaconstraint": "[]",
            "reusemodel": "[solve1.*]",
        },
    ]


def _dd_fast_medium_image_predict_solve_slots():
    slots = _dd_fast_medium_solve_slots()
    slots[0]["reusemodel"] = "[predict.*]"
    slots[1]["reusemodel"] = "[predict.*]"
    return slots


def _calibration_solve_options(**overrides) -> CalibrationSolveOptions:
    values = {
        "msin": "obs_0.ms",
        "data_colname": "DATA",
        "starttime": "50000.0",
        "ntimes": 10,
        "steps": "[solve1]",
        "solve_slots": _di_fulljones_solve_slots(),
        "num_threads": 4,
    }
    values.update(overrides)
    return CalibrationSolveOptions(**values)


def _dd_calibration_solve_options(**overrides) -> CalibrationSolveOptions:
    values = {
        "msin": "dd_obs_0.ms",
        "steps": "[solve1]",
        "solve_slots": _dd_fast_phase_solve_slots(),
        "timebase": 0.0,
        "maxinterval": 8.0,
        "frequencybase": 0.0,
        "minchannels": 1,
        "onebeamperpatch": True,
        "parallelbaselines": False,
        "sagecalpredict": False,
        "sourcedb": "calibration.skymodel",
        "directions": ["patch1", "patch2"],
    }
    values.update(overrides)
    return _calibration_solve_options(**values)


def _idgcal_screen_solve_options(**overrides) -> IdgcalScreenSolveOptions:
    values = {
        "msin": "dd_obs_0.ms",
        "starttime": "50000.0",
        "ntimes": 10,
        "h5parm": "idgcal_0",
        "solint_phase": 3,
        "model_images": [
            "calibration_model-term-0.fits",
            "calibration_model-term-1.fits",
        ],
        "maxiter": 4,
        "antennaconstraint": "[]",
        "num_threads": 4,
    }
    values.update(overrides)
    return IdgcalScreenSolveOptions(**values)


def _draw_model_options(**overrides) -> DrawModelOptions:
    values = {
        "skymodel": "calibration.skymodel",
        "num_terms": 2,
        "name": "calibration_model",
        "ra_dec": ["12:00:00.0", "+45.00.00.0"],
        "frequency_bandwidth": [150000000.0, 1000000.0],
        "cellsize_deg": 0.001,
        "imsize": [1024, 1024],
        "num_threads": 4,
    }
    values.update(overrides)
    return DrawModelOptions(**values)


def _wsclean_predict_options(**overrides) -> WscleanPredictOptions:
    values = {
        "msin": "dd_obs_0_wsclean_predict.ms",
        "region_file": "predict_field_facets_ds9.reg",
        "model_column": "patch1",
        "facet": "patch1",
        "model_root": "wsclean_predict_chunk_1_band_1",
        "channel_range": (0, 3),
        "model_storage_manager": "default",
        "num_threads": 4,
        "apply_time_frequency_smearing": True,
    }
    values.update(overrides)
    return WscleanPredictOptions(**values)


def test_calibrate_command_builders_match_reference_fixtures():
    commands = json.loads((FIXTURE_DIR / "command_reference.json").read_text())

    assert (
        normalize_command(
            build_calibration_solve_command(
                _calibration_solve_options(modeldatacolumn="[MODEL_DATA]")
            )
        )
        == commands["calibrate"]["calibration_di_fulljones"]
    )
    assert (
        normalize_command(
            build_collect_h5parms_command(
                ["fulljones_gain_0.h5parm", "fulljones_gain_1.h5parm"],
                "fulljones_solutions.h5",
            )
        )
        == commands["calibrate"]["collect_fulljones"]
    )
    assert (
        normalize_command(build_plot_solutions_command("fulljones_solutions.h5", "phase"))
        == commands["calibrate"]["plot_fulljones_phase"]
    )

    assert (
        normalize_command(
            build_calibration_solve_command(
                _calibration_solve_options(
                    steps="[solve1,solve2]",
                    solve_slots=_di_scalar_phase_solve_slots(),
                    modeldatacolumn="[MODEL_DATA]",
                )
            )
        )
        == commands["calibrate"]["calibration_di_scalar_phase"]
    )
    assert (
        normalize_command(build_draw_model_command(_draw_model_options()))
        == commands["calibrate"]["draw_model"]
    )
    assert (
        normalize_command(build_calibration_solve_command(_dd_calibration_solve_options()))
        == commands["calibrate"]["calibration_dd_fast_phase"]
    )
    assert (
        normalize_command(
            build_calibration_solve_command(
                _dd_calibration_solve_options(
                    steps="[solve1,solve2]",
                    solve_slots=_dd_fast_medium_solve_slots(),
                )
            )
        )
        == commands["calibrate"]["calibration_dd_fast_medium"]
    )

    bda_command = normalize_command(
        build_calibration_solve_command(
            _dd_calibration_solve_options(
                steps="[avg,solve1,solve2,null]",
                solve_slots=_dd_fast_medium_solve_slots(),
                timebase=20000.0,
                frequencybase=20000.0,
            )
        )
    )
    assert "steps=[avg,solve1,solve2,null]" in bda_command
    assert "null.type=null" in bda_command

    assert (
        normalize_command(
            build_calibration_solve_command(
                _dd_calibration_solve_options(
                    steps="[applycal,solve1,solve2]",
                    solve_slots=_dd_fast_medium_solve_slots(),
                    applycal_steps="[fastphase,slowgain,fulljones,normalization]",
                    applycal_h5parm="di_solutions.h5",
                    fulljones_h5parm="fulljones_solutions.h5",
                    normalize_h5parm="normalize_solutions.h5",
                )
            )
        )
        == commands["calibrate"]["calibration_dd_fast_medium_preapply"]
    )
    assert (
        normalize_command(
            build_calibration_solve_command(
                _dd_calibration_solve_options(
                    steps="[predict,applybeam,solve1,solve2]",
                    solve_slots=_dd_fast_medium_image_predict_solve_slots(),
                    sourcedb=None,
                    directions=None,
                    predict_regions="field_facets_ds9.reg",
                    predict_images=[
                        "calibration_model-term-0.fits",
                        "calibration_model-term-1.fits",
                    ],
                )
            )
        )
        == commands["calibrate"]["calibration_dd_fast_medium_image_predict"]
    )
    assert (
        normalize_command(
            build_calibration_solve_command(
                _dd_calibration_solve_options(
                    steps="[predict,applybeam,applycal,solve1,solve2]",
                    solve_slots=_dd_fast_medium_image_predict_solve_slots(),
                    applycal_steps="[fastphase,normalization]",
                    applycal_h5parm="di_solutions.h5",
                    normalize_h5parm="normalize_solutions.h5",
                    sourcedb=None,
                    directions=None,
                    predict_regions="field_facets_ds9.reg",
                    predict_images=[
                        "calibration_model-term-0.fits",
                        "calibration_model-term-1.fits",
                    ],
                )
            )
        )
        == commands["calibrate"]["calibration_dd_fast_medium_image_predict_preapply"]
    )
    assert (
        normalize_command(build_idgcal_solve_phase_command(_idgcal_screen_solve_options()))
        == commands["calibrate"]["idgcal_solve_phase"]
    )
    assert (
        normalize_command(
            build_idgcal_solve_phase_and_gain_command(
                _idgcal_screen_solve_options(solint_amplitude=11)
            )
        )
        == commands["calibrate"]["idgcal_solve_phase_and_gain"]
    )


@pytest.mark.parametrize(
    ("solve_type", "expected_soltab"),
    [
        ("fast_phase", "[phase000]"),
        ("medium_phase", "[phase000]"),
        ("slow_gains", "[phase000,amplitude000]"),
        ("full_jones", "[amplitude000,phase000]"),
    ],
)
def test_initial_solution_soltab_matches_solve_product_shape(solve_type, expected_soltab):
    """DP3 initial-solution soltabs must match the h5parm produced by each solve type."""
    assert initialsolutions_soltab(solve_type) == expected_soltab


def test_calibrate_command_builders_create_reference_tokens():
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
        *PLOT_SOLUTIONS_COMMAND_PREFIX,
        "fulljones_solutions.h5",
        "phase",
    ]
    assert build_plot_solutions_command("fulljones_solutions.h5", "phase", first_dir=True) == [
        *PLOT_SOLUTIONS_COMMAND_PREFIX,
        "fulljones_solutions.h5",
        "phase",
        "--first-dir",
    ]
    assert build_wsclean_predict_command(_wsclean_predict_options()) == [
        "wsclean",
        "-j",
        "4",
        "-predict",
        "-facet-regions",
        "predict_field_facets_ds9.reg",
        "-apply-time-frequency-smearing",
        "-model-column",
        "patch1",
        "-select-facets",
        "patch1",
        "-name",
        "wsclean_predict_chunk_1_band_1",
        "-channel-range",
        "0",
        "3",
        "-model-storage-manager",
        "default",
        "-no-reorder",
        "dd_obs_0_wsclean_predict.ms",
    ]
    assert build_idgcal_solve_phase_command(
        _idgcal_screen_solve_options(model_images=["calibration_model-term-0.fits"])
    )[:6] == [
        "DP3",
        "msin.datacolumn=DATA",
        "msout=",
        "steps=[solve]",
        "solve.type=python",
        "solve.python.module=idg.idgcaldpstep_phase_only_dirac",
    ]
    assert build_idgcal_solve_phase_and_gain_command(
        _idgcal_screen_solve_options(
            model_images=["calibration_model-term-0.fits"],
            solint_amplitude=11,
        )
    )[5:7] == [
        "solve.python.module=idg.idgcaldpstep_rapthor_dirac",
        "solve.python.class=IDGCalDPStepRapthorDirac",
    ]
    assert (
        build_calibration_solve_command(_calibration_solve_options(modeldatacolumn="[MODEL_DATA]"))[
            0
        ]
        == "DP3"
    )


def test_patch_names_from_region_uses_lsmtool_reader_signature(monkeypatch):
    def fake_read_ds9_region_file(region_file):
        assert region_file == "facets.reg"
        return [
            type("Facet", (), {"name": "patch1"})(),
            type("Facet", (), {"name": "patch2"})(),
        ]

    monkeypatch.setattr(
        "lsmtool.facet.read_ds9_region_file",
        fake_read_ds9_region_file,
    )

    assert calibrate_prediction._patch_names_from_region("facets.reg") == [
        "patch1",
        "patch2",
    ]


def test_calibrate_payload_from_inputs_builds_di_fulljones_payload(tmp_path):
    payload = calibrate_payload_from_inputs("di", _di_fulljones_input_parms(), tmp_path)

    assert payload["mode"] == "di"
    assert payload["calibration_kind"] == "di_fulljones"
    assert payload["pipeline_working_dir"] == str(tmp_path)
    assert payload["max_threads"] == 4
    assert payload["collected_h5parms"]["solve1"] == {
        "filename": "fulljones_solutions.h5",
        "path": str(tmp_path / "fulljones_solutions.h5"),
    }
    assert payload["chunks"] == [
        {
            "msin": "/data/obs_0.ms",
            "starttime": "50000.0",
            "ntimes": 10,
            "output_h5parm": "fulljones_gain_0.h5parm",
            "output_h5parm_path": str(tmp_path / "fulljones_gain_0.h5parm"),
            "solve1_solint": 5,
            "solve1_nchan": 2,
            "solve_slots": [
                {
                    "slot": 1,
                    "solve_type": "full_jones",
                    "solution_label": "fulljones",
                    "h5parm": "fulljones_gain_0.h5parm",
                    "h5parm_path": str(tmp_path / "fulljones_gain_0.h5parm"),
                    "solint": 5,
                    "mode": "fulljones",
                    "nchan": 2,
                    "solutions_per_direction": None,
                    "smoothness_dd_factors": None,
                    "smoothnessconstraint": None,
                    "smoothnessreffrequency": None,
                    "smoothnessrefdistance": None,
                    "antennaconstraint": None,
                    "keepmodel": "True",
                    "reusemodel": None,
                    "initialsolutions_h5parm": None,
                }
            ],
        },
        {
            "msin": "/data/obs_1.ms",
            "starttime": "50010.0",
            "ntimes": 12,
            "output_h5parm": "fulljones_gain_1.h5parm",
            "output_h5parm_path": str(tmp_path / "fulljones_gain_1.h5parm"),
            "solve1_solint": 6,
            "solve1_nchan": 3,
            "solve_slots": [
                {
                    "slot": 1,
                    "solve_type": "full_jones",
                    "solution_label": "fulljones",
                    "h5parm": "fulljones_gain_1.h5parm",
                    "h5parm_path": str(tmp_path / "fulljones_gain_1.h5parm"),
                    "solint": 6,
                    "mode": "fulljones",
                    "nchan": 3,
                    "solutions_per_direction": None,
                    "smoothness_dd_factors": None,
                    "smoothnessconstraint": None,
                    "smoothnessreffrequency": None,
                    "smoothnessrefdistance": None,
                    "antennaconstraint": None,
                    "keepmodel": "True",
                    "reusemodel": None,
                    "initialsolutions_h5parm": None,
                }
            ],
        },
    ]


def test_calibrate_payload_from_inputs_builds_di_scalar_phase_payload(tmp_path):
    payload = calibrate_payload_from_inputs("di", _di_scalar_phase_input_parms(), tmp_path)

    assert payload["calibration_kind"] == "di_scalar_phase"
    assert payload["collected_h5parms"] == {
        "solve1": {
            "filename": "fast_phases_di.h5parm",
            "path": str(tmp_path / "fast_phases_di.h5parm"),
        },
        "solve2": {
            "filename": "medium1_phases_di.h5parm",
            "path": str(tmp_path / "medium1_phases_di.h5parm"),
        },
    }
    assert payload["combined_h5parm"] == {
        "filename": "combined_solve1_solve2_di.h5parm",
        "path": str(tmp_path / "combined_solve1_solve2_di.h5parm"),
    }
    assert payload["chunks"][0]["solve_slots"] == [
        {
            "slot": 1,
            "solve_type": "fast_phase",
            "solution_label": "fast",
            "h5parm": "fast_phase_di_0.h5parm",
            "h5parm_path": str(tmp_path / "fast_phase_di_0.h5parm"),
            "solint": 5,
            "mode": "scalarphase",
            "nchan": 2,
            "solutions_per_direction": None,
            "smoothness_dd_factors": None,
            "smoothnessconstraint": 0,
            "smoothnessreffrequency": 0,
            "smoothnessrefdistance": None,
            "antennaconstraint": "[]",
            "keepmodel": "True",
            "reusemodel": None,
            "initialsolutions_h5parm": None,
        },
        {
            "slot": 2,
            "solve_type": "medium_phase",
            "solution_label": "medium1",
            "medium_index": 1,
            "h5parm": "medium1_phase_di_0.h5parm",
            "h5parm_path": str(tmp_path / "medium1_phase_di_0.h5parm"),
            "solint": 7,
            "mode": "scalarphase",
            "nchan": 4,
            "solutions_per_direction": None,
            "smoothness_dd_factors": None,
            "smoothnessconstraint": 0,
            "smoothnessreffrequency": 0,
            "smoothnessrefdistance": None,
            "antennaconstraint": "[]",
            "keepmodel": None,
            "reusemodel": "[solve1.*]",
            "initialsolutions_h5parm": None,
        },
    ]


def test_calibrate_payload_from_inputs_builds_di_fast_phase_payload(tmp_path):
    payload = calibrate_payload_from_inputs("di", _di_fast_phase_input_parms(), tmp_path)

    assert payload["calibration_kind"] == "di_fast_phase"
    assert payload["collected_h5parms"] == {
        "solve1": {
            "filename": "fast_phases_di.h5parm",
            "path": str(tmp_path / "fast_phases_di.h5parm"),
        }
    }
    assert payload["combined_h5parm"] is None
    assert [slot["slot"] for slot in payload["chunks"][0]["solve_slots"]] == [1]


def test_calibrate_payload_from_inputs_builds_di_slow_payload(tmp_path):
    payload = calibrate_payload_from_inputs("di", _di_slow_input_parms(), tmp_path)

    assert payload["calibration_kind"] == "di_slow"
    assert payload["collected_h5parms"] == {
        "solve1": {
            "filename": "slow_gains_di.h5parm",
            "path": str(tmp_path / "slow_gains_di.h5parm"),
        }
    }
    assert payload["chunks"][0]["solve_slots"][0] == {
        "slot": 1,
        "solve_type": "slow_gains",
        "solution_label": "slow",
        "h5parm": "slow_gains_di_0.h5parm",
        "h5parm_path": str(tmp_path / "slow_gains_di_0.h5parm"),
        "solint": 11,
        "mode": "diagonal",
        "nchan": 7,
        "solutions_per_direction": None,
        "smoothness_dd_factors": None,
        "smoothnessconstraint": 0,
        "smoothnessreffrequency": 0,
        "smoothnessrefdistance": None,
        "antennaconstraint": "[]",
        "keepmodel": "True",
        "reusemodel": None,
        "initialsolutions_h5parm": None,
    }


def test_calibrate_payload_from_inputs_builds_di_phase_slow_payload(tmp_path):
    payload = calibrate_payload_from_inputs("di", _di_phase_slow_input_parms(), tmp_path)

    assert payload["calibration_kind"] == "di_phase_slow"
    assert payload["combined_h5parms"] == {
        "phase_1_2": {
            "filename": "combined_solve1_solve2_di.h5parm",
            "path": str(tmp_path / "combined_solve1_solve2_di.h5parm"),
        },
        "final": {
            "filename": "combined_di_solutions.h5parm",
            "path": str(tmp_path / "combined_di_solutions.h5parm"),
        },
    }
    assert [slot["slot"] for slot in payload["chunks"][0]["solve_slots"]] == [1, 2, 3]
    assert payload["chunks"][0]["solve_slots"][2] == {
        "slot": 3,
        "solve_type": "slow_gains",
        "solution_label": "slow",
        "h5parm": "slow_gains_di_0.h5parm",
        "h5parm_path": str(tmp_path / "slow_gains_di_0.h5parm"),
        "solint": 11,
        "mode": "diagonal",
        "nchan": 7,
        "solutions_per_direction": None,
        "smoothness_dd_factors": None,
        "smoothnessconstraint": 0,
        "smoothnessreffrequency": 0,
        "smoothnessrefdistance": None,
        "antennaconstraint": "[]",
        "keepmodel": None,
        "reusemodel": None,
        "modeldatacolumns": "[MODEL_DATA]",
        "initialsolutions_h5parm": None,
    }


def test_calibrate_payload_from_inputs_builds_mixed_di_strategy_payload(tmp_path):
    payload = calibrate_payload_from_inputs("di", _di_scalar_slow_fulljones_input_parms(), tmp_path)

    assert payload["calibration_kind"] == "di_calibration"
    assert payload["combined_h5parms"] == {
        "phase_1_2": {
            "filename": "combined_solve1_solve2_di.h5parm",
            "path": str(tmp_path / "combined_solve1_solve2_di.h5parm"),
        },
        "final": {
            "filename": "combined_di_solutions.h5parm",
            "path": str(tmp_path / "combined_di_solutions.h5parm"),
        },
    }
    assert [
        (slot["slot"], slot["solve_type"], slot["h5parm"], slot["mode"])
        for slot in payload["chunks"][0]["solve_slots"]
    ] == [
        (1, "fast_phase", "fast_phase_di_0.h5parm", "scalarphase"),
        (2, "medium_phase", "medium1_phase_di_0.h5parm", "scalarphase"),
        (3, "slow_gains", "slow_gains_di_0.h5parm", "diagonal"),
        (4, "full_jones", "fulljones_gain_0.h5parm", "fulljones"),
    ]


def test_calibrate_payload_from_inputs_builds_dd_fast_phase_payload(tmp_path):
    payload = calibrate_payload_from_inputs("dd", _dd_fast_phase_input_parms(), tmp_path)

    assert payload["mode"] == "dd"
    assert payload["calibration_kind"] == "dd_fast_phase"
    assert payload["modeldatacolumn"] is None
    assert payload["sourcedb"] == "/data/calibration.skymodel"
    assert payload["directions"] == ["patch1", "patch2"]
    assert payload["bda_timebase"] == 0.0
    assert payload["bda_frequencybase"] == 0.0
    assert payload["onebeamperpatch"] is True
    assert payload["parallelbaselines"] is False
    assert payload["sagecalpredict"] is False
    assert payload["collected_h5parms"] == {
        "solve1": {
            "filename": "fast_phases.h5parm",
            "path": str(tmp_path / "fast_phases.h5parm"),
        }
    }
    assert payload["chunks"][0] == {
        "msin": "/data/dd_obs_0.ms",
        "starttime": "50000.0",
        "ntimes": 10,
        "output_h5parm": "fast_phase_0.h5parm",
        "output_h5parm_path": str(tmp_path / "fast_phase_0.h5parm"),
        "solve1_solint": 3,
        "solve1_nchan": 1,
        "solve_slots": [
            {
                "slot": 1,
                "solve_type": "fast_phase",
                "solution_label": "fast",
                "h5parm": "fast_phase_0.h5parm",
                "h5parm_path": str(tmp_path / "fast_phase_0.h5parm"),
                "solint": 3,
                "mode": "scalarphase",
                "nchan": 1,
                "solutions_per_direction": [1, 1],
                "datause": "full",
                "smoothness_dd_factors": [1.0, 2.0],
                "smoothnessconstraint": 1200000.0,
                "smoothnessreffrequency": 150000000.0,
                "smoothnessrefdistance": 2500.0,
                "antennaconstraint": "[[CS001HBA0,CS002HBA0]]",
                "keepmodel": "True",
                "reusemodel": None,
                "initialsolutions_h5parm": None,
            }
        ],
        "bda_maxinterval": 8.0,
        "bda_minchannels": 1,
    }


def test_calibrate_payload_from_inputs_builds_dd_slow_payload(tmp_path):
    payload = calibrate_payload_from_inputs("dd", _dd_slow_input_parms(), tmp_path)

    assert payload["calibration_kind"] == "dd_slow"
    assert payload["collected_h5parms"] == {
        "solve1": {
            "filename": "slow_gains.h5parm",
            "path": str(tmp_path / "slow_gains.h5parm"),
        }
    }
    assert payload["chunks"][0]["solve_slots"][0] == {
        "slot": 1,
        "solve_type": "slow_gains",
        "solution_label": "slow",
        "h5parm": "slow_gain_0.h5parm",
        "h5parm_path": str(tmp_path / "slow_gain_0.h5parm"),
        "solint": 11,
        "mode": "diagonal",
        "nchan": 7,
        "solutions_per_direction": [1, 1],
        "datause": "full",
        "smoothness_dd_factors": [1.0, 2.0],
        "smoothnessconstraint": 3600000.0,
        "smoothnessreffrequency": 0,
        "smoothnessrefdistance": None,
        "antennaconstraint": "[[CS001HBA0,CS002HBA0]]",
        "keepmodel": "True",
        "reusemodel": None,
        "initialsolutions_h5parm": None,
    }


def test_calibrate_payload_from_inputs_builds_dd_slow_then_medium_payload(tmp_path):
    payload = calibrate_payload_from_inputs("dd", _dd_slow_medium_input_parms(), tmp_path)

    assert payload["calibration_kind"] == "dd_calibration"
    assert payload["combined_h5parms"] == {
        "final": {
            "filename": "combined_solutions.h5",
            "path": str(tmp_path / "combined_solutions.h5"),
        },
    }
    assert [
        (
            slot["slot"],
            slot["solve_type"],
            slot["h5parm"],
            slot["antennaconstraint"],
            slot["reusemodel"],
        )
        for slot in payload["chunks"][0]["solve_slots"]
    ] == [
        (1, "slow_gains", "slow_gain_0.h5parm", "[]", None),
        (
            2,
            "medium_phase",
            "medium1_phase_0.h5parm",
            "[[CS001HBA0,CS002HBA0]]",
            "[solve1.*]",
        ),
    ]


def test_calibrate_payload_from_inputs_converts_array_like_scatter_values(tmp_path):
    input_parms = _dd_fast_medium_input_parms()
    input_parms["solve1_smoothness_dd_factors"] = [np.array([1.0]), np.array([1.5])]

    payload = calibrate_payload_from_inputs("dd", input_parms, tmp_path)

    assert payload["chunks"][0]["solve_slots"][0]["smoothness_dd_factors"] == [1.0]


def test_calibrate_payload_from_inputs_builds_dd_fast_medium_payload(tmp_path):
    payload = calibrate_payload_from_inputs("dd", _dd_fast_medium_input_parms(), tmp_path)

    assert payload["calibration_kind"] == "dd_phase"
    assert payload["collected_h5parms"]["solve2"] == {
        "filename": "medium1_phases.h5parm",
        "path": str(tmp_path / "medium1_phases.h5parm"),
    }
    assert payload["combined_h5parms"] == {
        "phase_1_2": {
            "filename": "combined_fast_medium1_phases.h5parm",
            "path": str(tmp_path / "combined_fast_medium1_phases.h5parm"),
        }
    }
    assert payload["chunks"][0]["solve_slots"][1] == {
        "slot": 2,
        "solve_type": "medium_phase",
        "solution_label": "medium1",
        "medium_index": 1,
        "h5parm": "medium1_phase_0.h5parm",
        "h5parm_path": str(tmp_path / "medium1_phase_0.h5parm"),
        "solint": 9,
        "mode": "scalarphase",
        "nchan": 5,
        "solutions_per_direction": [1, 1],
        "smoothness_dd_factors": [2.0, 3.0],
        "smoothnessconstraint": 2400000.0,
        "smoothnessreffrequency": 152000000.0,
        "smoothnessrefdistance": 3500.0,
        "antennaconstraint": "[]",
        "keepmodel": None,
        "reusemodel": "[solve1.*]",
        "datause": "full",
        "initialsolutions_h5parm": None,
    }


def test_calibrate_payload_from_inputs_preserves_custom_dd_solve_order(tmp_path):
    payload = calibrate_payload_from_inputs("dd", _dd_medium_fast_input_parms(), tmp_path)

    assert payload["calibration_kind"] == "dd_calibration"
    assert payload["combined_h5parms"] == {
        "phase_1_2": {
            "filename": "combined_fast_medium1_phases.h5parm",
            "path": str(tmp_path / "combined_fast_medium1_phases.h5parm"),
        }
    }
    assert [
        (
            slot["slot"],
            slot["solve_type"],
            slot["h5parm"],
            slot["mode"],
            slot["reusemodel"],
        )
        for slot in payload["chunks"][0]["solve_slots"]
    ] == [
        (1, "medium_phase", "medium1_phase_0.h5parm", "scalarphase", None),
        (2, "fast_phase", "fast_phase_0.h5parm", "scalarphase", "[solve1.*]"),
    ]


def test_calibrate_payload_from_inputs_preserves_solve_initial_solution_h5parm(tmp_path):
    input_parms = _dd_fast_phase_input_parms()
    previous_solution = tmp_path / "solutions" / "calibrate_1" / "field-solutions-fast-phase.h5"
    input_parms["solve1_initialsolutions_h5parm"] = file_record(previous_solution)

    payload = calibrate_payload_from_inputs("dd", input_parms, tmp_path / "pipeline")
    first_slot = payload["chunks"][0]["solve_slots"][0]
    command = build_calibrate_chunk_command(payload, payload["chunks"][0])

    assert first_slot["initialsolutions_h5parm"] == str(previous_solution)
    assert f"solve1.initialsolutions.h5parm={previous_solution}" in command


def test_calibrate_payload_from_inputs_builds_dd_preapply_payload(tmp_path):
    payload = calibrate_payload_from_inputs("dd", _dd_preapply_input_parms(), tmp_path)

    assert payload["calibration_kind"] == "dd_phase"
    assert payload["dp3_steps"] == "[applycal,solve1,solve2]"
    assert payload["applycal_steps"] == "[fastphase,slowgain,fulljones,normalization]"
    assert payload["applycal_h5parm"] == "/solutions/di_solutions.h5"
    assert payload["fulljones_h5parm"] == "/solutions/fulljones_solutions.h5"
    assert payload["normalize_h5parm"] == "/solutions/normalize_solutions.h5"


def test_calibrate_payload_from_inputs_builds_dd_image_predict_payload(tmp_path):
    payload = calibrate_payload_from_inputs("dd", _dd_image_predict_input_parms(), tmp_path)

    assert payload["calibration_kind"] == "dd_phase"
    assert payload["image_based_predict"] is True
    assert payload["sourcedb"] == "/data/calibration.skymodel"
    assert payload["directions"] == ["patch1", "patch2"]
    assert payload["image_predict"] == {
        "skymodel": "/data/calibration.skymodel",
        "model_image_root": "calibration_model",
        "model_image_ra_dec": ["12:00:00.0", "+45.00.00.0"],
        "model_image_imsize": [1024, 1024],
        "model_image_cellsize": 0.001,
        "model_image_frequency_bandwidth": [150000000.0, 1000000.0],
        "max_predict_bandwidth_hz": 2000000.0,
        "num_spectral_terms": 2,
        "model_images": [
            str(tmp_path / "calibration_model-term-0.fits"),
            str(tmp_path / "calibration_model-term-1.fits"),
        ],
        "ra_mid": 123.0,
        "dec_mid": 45.0,
        "facet_region_width_ra": 2.0,
        "facet_region_width_dec": 2.5,
        "facet_region_file": "field_facets_ds9.reg",
        "facet_region_path": str(tmp_path / "field_facets_ds9.reg"),
    }
    assert payload["chunks"][0]["solve_slots"][0]["reusemodel"] == "[predict.*]"
    assert payload["chunks"][0]["solve_slots"][1]["reusemodel"] == "[predict.*]"


def test_calibrate_payload_from_inputs_builds_dd_wsclean_predict_payload(tmp_path):
    payload = calibrate_payload_from_inputs("dd", _dd_wsclean_predict_input_parms(), tmp_path)

    assert payload["calibration_kind"] == "dd_phase"
    assert payload["image_based_predict"] is False
    assert payload["wsclean_predict"] is True
    assert payload["modeldatacolumn"] is None
    assert payload["image_predict"]["max_predict_bandwidth_hz"] == 2000000.0
    assert payload["image_predict"]["facet_region_file"] == "predict_field_facets_ds9.reg"
    assert payload["image_predict"]["facet_region_path"] == str(
        tmp_path / "predict_field_facets_ds9.reg"
    )
    assert payload["chunks"][0]["solve_slots"][0]["reusemodel"] is None
    assert payload["chunks"][0]["solve_slots"][1]["reusemodel"] is None
    assert "modeldatacolumns" not in payload["chunks"][0]["solve_slots"][1]


def test_calibrate_payload_from_inputs_uses_configured_wsclean_predict_bandwidth(tmp_path):
    input_parms = _dd_wsclean_predict_input_parms()
    input_parms["wsclean_predict_bw"] = 500000.0

    payload = calibrate_payload_from_inputs("dd", input_parms, tmp_path)

    assert payload["image_predict"]["max_predict_bandwidth_hz"] == 500000.0


def test_calibrate_payload_from_inputs_rejects_invalid_wsclean_predict_bandwidth(tmp_path):
    input_parms = _dd_wsclean_predict_input_parms()
    input_parms["wsclean_predict_bw"] = 0.0

    with pytest.raises(ValueError, match="wsclean_predict_bw"):
        calibrate_payload_from_inputs("dd", input_parms, tmp_path)


def test_calibrate_payload_from_inputs_builds_dd_with_slow_payload(tmp_path):
    payload = calibrate_payload_from_inputs("dd", _dd_with_slow_input_parms(), tmp_path)

    assert payload["calibration_kind"] == "dd_phase_slow"
    assert payload["has_slow_gain_solve"] is True
    assert payload["combined_h5parms"] == {
        "phase_1_2": {
            "filename": "combined_fast_medium1_phases.h5parm",
            "path": str(tmp_path / "combined_fast_medium1_phases.h5parm"),
        },
        "phase_1_2_3": {
            "filename": "combined_fast_medium1_medium2_phases.h5parm",
            "path": str(tmp_path / "combined_fast_medium1_medium2_phases.h5parm"),
        },
        "final": {
            "filename": "combined_solutions.h5",
            "path": str(tmp_path / "combined_solutions.h5"),
        },
    }
    assert [slot["slot"] for slot in payload["chunks"][0]["solve_slots"]] == [1, 2, 3, 4]
    assert payload["chunks"][0]["solve_slots"][2] == {
        "slot": 3,
        "solve_type": "slow_gains",
        "solution_label": "slow",
        "h5parm": "slow_gain_0.h5parm",
        "h5parm_path": str(tmp_path / "slow_gain_0.h5parm"),
        "solint": 11,
        "mode": "diagonal",
        "nchan": 7,
        "solutions_per_direction": [1, 1],
        "smoothness_dd_factors": [3.0, 4.0],
        "smoothnessconstraint": 3600000.0,
        "smoothnessreffrequency": 0,
        "smoothnessrefdistance": None,
        "antennaconstraint": "[]",
        "keepmodel": "true",
        "reusemodel": "[solve1.*]",
        "datause": "full",
        "initialsolutions_h5parm": None,
    }


def test_calibrate_payload_from_inputs_builds_dd_screen_payload(tmp_path):
    payload = calibrate_payload_from_inputs("dd", _dd_screen_input_parms(), tmp_path)

    assert payload["calibration_kind"] == "dd_screen"
    assert payload["image_based_predict"] is True
    assert payload["has_slow_gain_solve"] is False
    assert payload["idgcal_antennaconstraint"] == "[]"
    assert payload["combined_h5parm"] == {
        "filename": "combined_solutions.h5",
        "path": str(tmp_path / "combined_solutions.h5"),
    }
    assert payload["chunks"] == [
        {
            "msin": "/data/dd_obs_0.ms",
            "starttime": "50000.0",
            "ntimes": 10,
            "output_h5parm": "idgcal_0",
            "output_h5parm_path": str(tmp_path / "idgcal_0"),
            "solint_fast": 3,
        },
        {
            "msin": "/data/dd_obs_1.ms",
            "starttime": "50010.0",
            "ntimes": 12,
            "output_h5parm": "idgcal_1",
            "output_h5parm_path": str(tmp_path / "idgcal_1"),
            "solint_fast": 4,
        },
    ]


def test_calibrate_payload_from_inputs_builds_dd_screen_slow_payload(tmp_path):
    payload = calibrate_payload_from_inputs(
        "dd",
        _dd_screen_input_parms(has_slow_gain_solve=True),
        tmp_path,
    )

    assert payload["calibration_kind"] == "dd_screen"
    assert payload["has_slow_gain_solve"] is True
    assert payload["chunks"][0]["solint_fast"] == 3
    assert payload["chunks"][0]["solint_slow"] == 11


@pytest.mark.parametrize(
    "mode, input_factory, updates, match",
    [
        (
            "dd",
            _dd_fast_phase_input_parms,
            {"dp3_steps": "[predict,applybeam,solve1]"},
            "DD image-based prediction requires model_image_root",
        ),
        (
            "dd",
            _dd_image_predict_input_parms,
            {"dp3_steps": "[predict,solve1,solve2]"},
            "DD image-based prediction requires predict and applybeam steps",
        ),
        (
            "dd",
            _dd_image_predict_input_parms,
            {"model_image_imsize": [1024]},
            "model_image_imsize must contain exactly 2 entries",
        ),
        (
            "dd",
            _dd_fast_phase_input_parms,
            {"solve1_type": "unknown"},
            "Unsupported DD calibration solve slot",
        ),
        (
            "dd",
            _dd_fast_medium_input_parms,
            {"dp3_steps": "[applycal,solve1,solve2]", "applycal_steps": None},
            "DD pre-application requires applycal_steps",
        ),
        (
            "dd",
            _dd_fast_medium_input_parms,
            {"dp3_steps": "[applycal,solve1,solve2]", "applycal_steps": "[unknown]"},
            "Unsupported DD pre-apply step",
        ),
        (
            "dd",
            _dd_fast_medium_input_parms,
            {"dp3_steps": "[applycal,solve1,solve2]", "applycal_steps": "[mediumphase]"},
            "Unsupported DD pre-apply step",
        ),
        (
            "dd",
            _dd_fast_medium_input_parms,
            {
                "dp3_steps": "[applycal,solve1,solve2]",
                "applycal_steps": "[fulljones]",
            },
            "DD pre-application requires fulljones_h5parm",
        ),
        (
            "di",
            _di_fulljones_input_parms,
            {"max_normalization_delta": None},
            "Full-Jones gain processing requires max_normalization_delta",
        ),
        (
            "di",
            _di_fulljones_input_parms,
            {"solve1_mode": "slow"},
            "Unsupported DI calibration solve slot",
        ),
        (
            "di",
            _di_fulljones_input_parms,
            {"dp3_steps": "[solve1,solve2]"},
            "Unsupported DI calibration solve slot",
        ),
    ],
)
def test_calibrate_payload_from_inputs_rejects_unsupported_slice(
    tmp_path, mode, input_factory, updates, match
):
    input_parms = input_factory()
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

    assert output == {"solve1": file_record(tmp_path / "fulljones_gain_0.h5parm")}
    assert fake_calibrate_shell_operation_cls.instances[0].kwargs["working_dir"] == str(tmp_path)


def test_run_plot_solutions_returns_only_new_plots(tmp_path, fake_calibrate_shell_operation_cls):
    h5parm = tmp_path / "solutions.h5parm"
    h5parm.write_text("h5parm")
    (tmp_path / "phase_solutions.png").write_text("existing")

    plots = calibrate_collection.run_plot_solutions(
        file_record(h5parm),
        "phase",
        str(tmp_path),
        ExecutionConfig(task_runner="sync"),
        shell_operation_cls=fake_calibrate_shell_operation_cls,
    )

    assert plots == []


def test_run_plot_solutions_publishes_new_plot_artifacts(
    tmp_path, monkeypatch, fake_calibrate_shell_operation_cls
):
    h5parm = tmp_path / "solutions.h5parm"
    h5parm.write_text("h5parm")
    published = []

    def fake_publish(records, root_dir):
        published.append((records, root_dir))
        return []

    monkeypatch.setattr(calibrate_collection, "publish_plot_file_records", fake_publish)

    plots = calibrate_collection.run_plot_solutions(
        file_record(h5parm),
        "phase",
        str(tmp_path),
        ExecutionConfig(task_runner="sync"),
        first_dir=True,
        shell_operation_cls=fake_calibrate_shell_operation_cls,
    )

    assert plots == [file_record(tmp_path / "phase_solutions.png")]
    assert published == [(plots, str(tmp_path))]
    command = shlex.split(fake_calibrate_shell_operation_cls.instances[0].kwargs["commands"][0])
    assert "--first-dir" in command


def test_run_calibrate_flow_supports_di_fulljones(
    tmp_path, fake_calibrate_shell_operation_cls, fake_direct_calibrate_helpers
):
    payload = calibrate_payload_from_inputs("di", _di_fulljones_input_parms(), tmp_path)

    outputs = run_flow_for_test(
        calibrate_flow,
        payload,
        execution_config=ExecutionConfig(task_runner="sync"),
        shell_operation_cls=fake_calibrate_shell_operation_cls,
    )

    expected_solution = file_record(tmp_path / "fulljones_solutions.h5")
    assert outputs == {
        "combined_solutions": expected_solution,
        "fulljones_solutions": expected_solution,
        "fulljones_phase_plots": [file_record(tmp_path / "fulljones_phase_solutions.png")],
    }
    for value in outputs.values():
        validate_output_record(value)

    commands = [
        shlex.split(instance.kwargs["commands"][0])
        for instance in fake_calibrate_shell_operation_cls.instances
    ]
    assert _command_names(commands) == [
        "DP3",
        "DP3",
        "H5parm_collector.py",
        PLOT_SOLUTIONS_COMMAND_NAME,
    ]
    assert "solve1.mode=fulljones" in commands[0]
    assert "solve1.h5parm=fulljones_gain_0.h5parm" in commands[0]
    assert "numthreads=4" in commands[0]
    assert "--first-dir" in commands[3]
    assert (
        commands[2][2]
        == f"{tmp_path / 'fulljones_gain_0.h5parm'},{tmp_path / 'fulljones_gain_1.h5parm'}"
    )
    assert fake_direct_calibrate_helpers["process_gains"] == [
        {
            "h5parmfile": str(tmp_path / "fulljones_solutions.h5"),
            "normalize": True,
            "flag": False,
            "smooth": False,
            "max_station_delta": 0.3,
            "scale_delta_with_dist": False,
            "phase_center": (0.0, 0.0),
        }
    ]


def test_run_calibrate_flow_rejects_invalid_chunk_payload(
    tmp_path, fake_calibrate_shell_operation_cls
):
    payload = calibrate_payload_from_inputs("di", _di_fulljones_input_parms(), tmp_path)
    payload["chunks"] = ["not-a-chunk"]

    with pytest.raises(ValueError, match=r"chunks\[0\]"):
        run_flow_for_test(
            calibrate_flow,
            payload,
            execution_config=ExecutionConfig(task_runner="sync"),
            shell_operation_cls=fake_calibrate_shell_operation_cls,
        )

    assert fake_calibrate_shell_operation_cls.instances == []


def test_run_calibrate_flow_supports_di_scalar_phase(tmp_path, fake_calibrate_shell_operation_cls):
    payload = calibrate_payload_from_inputs("di", _di_scalar_phase_input_parms(), tmp_path)

    outputs = run_flow_for_test(
        calibrate_flow,
        payload,
        execution_config=ExecutionConfig(task_runner="sync"),
        shell_operation_cls=fake_calibrate_shell_operation_cls,
    )

    assert outputs == {
        "combined_solutions": file_record(tmp_path / "combined_solve1_solve2_di.h5parm"),
        "fast_phase_solutions": file_record(tmp_path / "fast_phases_di.h5parm"),
        "medium1_phase_solutions": file_record(tmp_path / "medium1_phases_di.h5parm"),
        "fast_phase_plots": [file_record(tmp_path / "phase_solutions.png")],
        "medium1_phase_plots": [file_record(tmp_path / "medium1_phase_solutions.png")],
    }
    for value in outputs.values():
        validate_output_record(value)

    commands = [
        shlex.split(instance.kwargs["commands"][0])
        for instance in fake_calibrate_shell_operation_cls.instances
    ]
    assert _command_names(commands) == _calibration_command_names_after_collect_split(1, 1)
    assert "steps=[solve1,solve2]" in commands[0]
    assert "solve1.mode=scalarphase" in commands[0]
    assert "solve2.mode=scalarphase" in commands[0]
    assert "solve2.modeldatacolumns=[MODEL_DATA]" not in commands[0]
    assert "solve2.reusemodel=[solve1.*]" in commands[0]
    assert "--first-dir" in commands[4]
    assert "--first-dir" in commands[5]


def test_run_calibrate_flow_supports_di_fast_phase(tmp_path, fake_calibrate_shell_operation_cls):
    payload = calibrate_payload_from_inputs("di", _di_fast_phase_input_parms(), tmp_path)

    outputs = run_flow_for_test(
        calibrate_flow,
        payload,
        execution_config=ExecutionConfig(task_runner="sync"),
        shell_operation_cls=fake_calibrate_shell_operation_cls,
    )

    expected_solution = file_record(tmp_path / "fast_phases_di.h5parm")
    assert outputs == {
        "combined_solutions": expected_solution,
        "fast_phase_solutions": expected_solution,
        "fast_phase_plots": [file_record(tmp_path / "phase_solutions.png")],
    }
    commands = _command_tokens(fake_calibrate_shell_operation_cls)
    assert _command_names(commands) == [
        "DP3",
        "DP3",
        "H5parm_collector.py",
        PLOT_SOLUTIONS_COMMAND_NAME,
    ]
    assert "steps=[solve1]" in commands[0]
    assert "solve1.h5parm=fast_phase_di_0.h5parm" in commands[0]
    assert "--first-dir" in commands[3]


def test_run_calibrate_flow_supports_di_slow(tmp_path, fake_calibrate_shell_operation_cls):
    payload = calibrate_payload_from_inputs("di", _di_slow_input_parms(), tmp_path)

    outputs = run_flow_for_test(
        calibrate_flow,
        payload,
        execution_config=ExecutionConfig(task_runner="sync"),
        shell_operation_cls=fake_calibrate_shell_operation_cls,
    )

    expected_solution = file_record(tmp_path / "slow_gains_di.h5parm")
    assert outputs == {
        "combined_solutions": expected_solution,
        "slow_gain_solutions": expected_solution,
        "slow_phase_plots": [file_record(tmp_path / "slow_phase_solutions.png")],
        "slow_amp_plots": [file_record(tmp_path / "slow_amplitude_solutions.png")],
    }
    commands = _command_tokens(fake_calibrate_shell_operation_cls)
    assert _command_names(commands) == [
        "DP3",
        "DP3",
        "H5parm_collector.py",
        PLOT_SOLUTIONS_COMMAND_NAME,
        PLOT_SOLUTIONS_COMMAND_NAME,
    ]
    assert "steps=[solve1]" in commands[0]
    assert "solve1.h5parm=slow_gains_di_0.h5parm" in commands[0]
    assert "solve1.mode=diagonal" in commands[0]
    assert "solve1.solint=11" in commands[0]
    assert "--first-dir" in commands[3]
    assert "--first-dir" in commands[4]


def test_run_calibrate_flow_supports_di_phase_slow(tmp_path, fake_calibrate_shell_operation_cls):
    payload = calibrate_payload_from_inputs("di", _di_phase_slow_input_parms(), tmp_path)

    outputs = run_flow_for_test(
        calibrate_flow,
        payload,
        execution_config=ExecutionConfig(task_runner="sync"),
        shell_operation_cls=fake_calibrate_shell_operation_cls,
    )

    assert outputs["combined_solutions"] == file_record(tmp_path / "combined_di_solutions.h5parm")
    assert outputs["fast_phase_solutions"] == file_record(tmp_path / "fast_phases_di.h5parm")
    assert outputs["medium1_phase_solutions"] == file_record(tmp_path / "medium1_phases_di.h5parm")
    assert outputs["slow_gain_solutions"] == file_record(tmp_path / "slow_gains_di.h5parm")
    assert outputs["slow_amp_plots"] == [file_record(tmp_path / "slow_amplitude_solutions.png")]

    commands = _command_tokens(fake_calibrate_shell_operation_cls)
    assert _command_names(commands) == _calibration_command_names_after_collect_split(1, 1, 2)
    assert "steps=[solve1,solve2,solve3]" in commands[0]
    assert "solve3.h5parm=slow_gains_di_0.h5parm" in commands[0]
    assert "solve2.modeldatacolumns=[MODEL_DATA]" in commands[0]
    assert "solve3.modeldatacolumns=[MODEL_DATA]" in commands[0]
    assert "solve2.reusemodel=[solve1.*]" not in commands[0]
    assert "solve3.reusemodel=[solve1.*]" not in commands[0]
    assert "--first-dir" in commands[5]
    assert "--first-dir" in commands[6]
    assert "--first-dir" in commands[7]
    assert "--first-dir" in commands[8]


def test_run_calibrate_flow_supports_mixed_di_strategy(
    tmp_path, fake_calibrate_shell_operation_cls, fake_direct_calibrate_helpers
):
    payload = calibrate_payload_from_inputs("di", _di_scalar_slow_fulljones_input_parms(), tmp_path)

    outputs = run_flow_for_test(
        calibrate_flow,
        payload,
        execution_config=ExecutionConfig(task_runner="sync"),
        shell_operation_cls=fake_calibrate_shell_operation_cls,
    )

    assert outputs == {
        "combined_solutions": file_record(tmp_path / "combined_di_solutions.h5parm"),
        "fast_phase_solutions": file_record(tmp_path / "fast_phases_di.h5parm"),
        "medium1_phase_solutions": file_record(tmp_path / "medium1_phases_di.h5parm"),
        "slow_gain_solutions": file_record(tmp_path / "slow_gains_di.h5parm"),
        "fulljones_solutions": file_record(tmp_path / "fulljones_solutions.h5"),
        "fast_phase_plots": [file_record(tmp_path / "phase_solutions.png")],
        "medium1_phase_plots": [file_record(tmp_path / "medium1_phase_solutions.png")],
        "slow_phase_plots": [file_record(tmp_path / "slow_phase_solutions.png")],
        "slow_amp_plots": [file_record(tmp_path / "slow_amplitude_solutions.png")],
        "fulljones_phase_plots": [file_record(tmp_path / "fulljones_phase_solutions.png")],
    }

    commands = _command_tokens(fake_calibrate_shell_operation_cls)
    assert _command_names(commands) == _calibration_command_names_after_collect_split(1, 1, 2, 1)
    assert "steps=[solve1,solve2,solve3,solve4]" in commands[0]
    assert "solve3.initialsolutions.soltab=[phase000,amplitude000]" in commands[0]
    assert "solve4.h5parm=fulljones_gain_0.h5parm" in commands[0]
    assert "solve4.mode=fulljones" in commands[0]
    assert fake_direct_calibrate_helpers["process_gains"] == [
        {
            "h5parmfile": str(tmp_path / "slow_gains_di.h5parm"),
            "normalize": True,
            "flag": True,
            "smooth": True,
            "max_station_delta": 0.25,
            "scale_delta_with_dist": "False",
            "phase_center": (123.0, 45.0),
        },
        {
            "h5parmfile": str(tmp_path / "fulljones_solutions.h5"),
            "normalize": True,
            "flag": False,
            "smooth": False,
            "max_station_delta": 0.25,
            "scale_delta_with_dist": False,
            "phase_center": (0.0, 0.0),
        },
    ]


def test_run_calibrate_flow_supports_dd_fast_phase(tmp_path, fake_calibrate_shell_operation_cls):
    payload = calibrate_payload_from_inputs("dd", _dd_fast_phase_input_parms(), tmp_path)

    outputs = run_flow_for_test(
        calibrate_flow,
        payload,
        execution_config=ExecutionConfig(task_runner="sync"),
        shell_operation_cls=fake_calibrate_shell_operation_cls,
    )

    expected_solution = file_record(tmp_path / "fast_phases.h5parm")
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
    assert _command_names(commands) == [
        "DP3",
        "DP3",
        "H5parm_collector.py",
        PLOT_SOLUTIONS_COMMAND_NAME,
    ]
    assert "solve1.mode=scalarphase" in commands[0]
    assert "solve1.h5parm=fast_phase_0.h5parm" in commands[0]
    assert "solve1.sourcedb=/data/calibration.skymodel" in commands[0]
    assert "solve1.directions=[patch1,patch2]" in commands[0]
    assert "solve1.smoothness_dd_factors=[1.0,2.0]" in commands[0]
    assert "avg.maxinterval=8.0" in commands[0]
    assert "--first-dir" not in commands[3]
    assert (
        commands[2][2] == f"{tmp_path / 'fast_phase_0.h5parm'},{tmp_path / 'fast_phase_1.h5parm'}"
    )


def test_run_calibrate_flow_supports_dd_slow(tmp_path, fake_calibrate_shell_operation_cls):
    payload = calibrate_payload_from_inputs("dd", _dd_slow_input_parms(), tmp_path)

    outputs = run_flow_for_test(
        calibrate_flow,
        payload,
        execution_config=ExecutionConfig(task_runner="sync"),
        shell_operation_cls=fake_calibrate_shell_operation_cls,
    )

    expected_solution = file_record(tmp_path / "slow_gains.h5parm")
    assert outputs == {
        "combined_solutions": expected_solution,
        "slow_gain_solutions": expected_solution,
        "slow_phase_plots": [file_record(tmp_path / "slow_phase_solutions.png")],
        "slow_amp_plots": [file_record(tmp_path / "slow_amplitude_solutions.png")],
    }
    for value in outputs.values():
        validate_output_record(value)

    commands = [
        shlex.split(instance.kwargs["commands"][0])
        for instance in fake_calibrate_shell_operation_cls.instances
    ]
    assert _command_names(commands) == [
        "DP3",
        "DP3",
        "H5parm_collector.py",
        PLOT_SOLUTIONS_COMMAND_NAME,
        PLOT_SOLUTIONS_COMMAND_NAME,
    ]
    assert "solve1.mode=diagonal" in commands[0]
    assert "solve1.h5parm=slow_gain_0.h5parm" in commands[0]
    assert "solve1.solint=11" in commands[0]
    assert "--first-dir" not in commands[3]
    assert "--first-dir" not in commands[4]
    assert commands[2][2] == f"{tmp_path / 'slow_gain_0.h5parm'},{tmp_path / 'slow_gain_1.h5parm'}"


def test_run_calibrate_flow_supports_dd_slow_then_medium(
    tmp_path, fake_calibrate_shell_operation_cls
):
    payload = calibrate_payload_from_inputs("dd", _dd_slow_medium_input_parms(), tmp_path)

    outputs = run_flow_for_test(
        calibrate_flow,
        payload,
        execution_config=ExecutionConfig(task_runner="sync"),
        shell_operation_cls=fake_calibrate_shell_operation_cls,
    )

    assert outputs == {
        "combined_solutions": file_record(tmp_path / "combined_solutions.h5"),
        "slow_gain_solutions": file_record(tmp_path / "slow_gains.h5parm"),
        "medium1_phase_solutions": file_record(tmp_path / "medium1_phases.h5parm"),
        "slow_phase_plots": [file_record(tmp_path / "slow_phase_solutions.png")],
        "slow_amp_plots": [file_record(tmp_path / "slow_amplitude_solutions.png")],
        "medium1_phase_plots": [file_record(tmp_path / "medium1_phase_solutions.png")],
    }
    for value in outputs.values():
        validate_output_record(value)

    commands = _command_tokens(fake_calibrate_shell_operation_cls)
    assert _command_names(commands) == _calibration_command_names_after_collect_split(2, 1)
    assert "steps=[solve1,solve2]" in commands[0]
    assert "solve1.mode=diagonal" in commands[0]
    assert "solve1.antennaconstraint=[]" in commands[0]
    assert "solve2.mode=scalarphase" in commands[0]
    assert "solve2.antennaconstraint=[[CS001HBA0,CS002HBA0]]" in commands[0]
    assert "solve2.reusemodel=[solve1.*]" in commands[0]


def test_run_calibrate_flow_supports_dd_fast_medium(tmp_path, fake_calibrate_shell_operation_cls):
    payload = calibrate_payload_from_inputs("dd", _dd_fast_medium_input_parms(), tmp_path)

    outputs = run_flow_for_test(
        calibrate_flow,
        payload,
        execution_config=ExecutionConfig(task_runner="sync"),
        shell_operation_cls=fake_calibrate_shell_operation_cls,
    )

    assert outputs == {
        "combined_solutions": file_record(tmp_path / "combined_fast_medium1_phases.h5parm"),
        "fast_phase_solutions": file_record(tmp_path / "fast_phases.h5parm"),
        "medium1_phase_solutions": file_record(tmp_path / "medium1_phases.h5parm"),
        "fast_phase_plots": [file_record(tmp_path / "phase_solutions.png")],
        "medium1_phase_plots": [file_record(tmp_path / "medium1_phase_solutions.png")],
    }
    for value in outputs.values():
        validate_output_record(value)

    commands = [
        shlex.split(instance.kwargs["commands"][0])
        for instance in fake_calibrate_shell_operation_cls.instances
    ]
    assert _command_names(commands) == _calibration_command_names_after_collect_split(1, 1)
    assert "steps=[solve1,solve2]" in commands[0]
    assert "solve2.h5parm=medium1_phase_0.h5parm" in commands[0]
    assert "solve2.reusemodel=[solve1.*]" in commands[0]


def test_run_calibrate_flow_supports_custom_dd_solve_order(
    tmp_path, fake_calibrate_shell_operation_cls
):
    payload = calibrate_payload_from_inputs("dd", _dd_medium_fast_input_parms(), tmp_path)

    outputs = run_flow_for_test(
        calibrate_flow,
        payload,
        execution_config=ExecutionConfig(task_runner="sync"),
        shell_operation_cls=fake_calibrate_shell_operation_cls,
    )

    assert outputs == {
        "combined_solutions": file_record(tmp_path / "combined_fast_medium1_phases.h5parm"),
        "medium1_phase_solutions": file_record(tmp_path / "medium1_phases.h5parm"),
        "fast_phase_solutions": file_record(tmp_path / "fast_phases.h5parm"),
        "medium1_phase_plots": [file_record(tmp_path / "medium1_phase_solutions.png")],
        "fast_phase_plots": [file_record(tmp_path / "phase_solutions.png")],
    }
    commands = _command_tokens(fake_calibrate_shell_operation_cls)
    assert _command_names(commands) == _calibration_command_names_after_collect_split(1, 1)
    assert "steps=[solve1,solve2]" in commands[0]
    assert "solve1.h5parm=medium1_phase_0.h5parm" in commands[0]
    assert "solve1.mode=scalarphase" in commands[0]
    assert "solve2.h5parm=fast_phase_0.h5parm" in commands[0]
    assert "solve2.mode=scalarphase" in commands[0]
    assert "solve2.reusemodel=[solve1.*]" in commands[0]


def test_run_calibrate_flow_supports_dd_preapply(tmp_path, fake_calibrate_shell_operation_cls):
    payload = calibrate_payload_from_inputs("dd", _dd_preapply_input_parms(), tmp_path)

    outputs = run_flow_for_test(
        calibrate_flow,
        payload,
        execution_config=ExecutionConfig(task_runner="sync"),
        shell_operation_cls=fake_calibrate_shell_operation_cls,
    )

    assert outputs["combined_solutions"] == file_record(
        tmp_path / "combined_fast_medium1_phases.h5parm"
    )

    commands = [
        shlex.split(instance.kwargs["commands"][0])
        for instance in fake_calibrate_shell_operation_cls.instances
    ]
    assert _command_names(commands) == _calibration_command_names_after_collect_split(1, 1)
    assert "steps=[applycal,solve1,solve2]" in commands[0]
    assert "applycal.steps=[fastphase,slowgain,fulljones,normalization]" in commands[0]
    assert "applycal.parmdb=/solutions/di_solutions.h5" in commands[0]
    assert "applycal.fulljones.parmdb=/solutions/fulljones_solutions.h5" in commands[0]
    assert "applycal.normalization.parmdb=/solutions/normalize_solutions.h5" in commands[0]
    assert "solve2.reusemodel=[solve1.*]" in commands[0]


def test_run_calibrate_flow_supports_dd_image_predict(tmp_path, fake_calibrate_shell_operation_cls):
    payload = calibrate_payload_from_inputs("dd", _dd_image_predict_input_parms(), tmp_path)

    outputs = run_flow_for_test(
        calibrate_flow,
        payload,
        execution_config=ExecutionConfig(task_runner="sync"),
        shell_operation_cls=fake_calibrate_shell_operation_cls,
    )

    assert outputs["combined_solutions"] == file_record(
        tmp_path / "combined_fast_medium1_phases.h5parm"
    )
    assert (tmp_path / "calibration_model-term-0.fits").is_file()
    assert (tmp_path / "calibration_model-term-1.fits").is_file()
    assert (tmp_path / "field_facets_ds9.reg").is_file()

    commands = [
        shlex.split(instance.kwargs["commands"][0])
        for instance in fake_calibrate_shell_operation_cls.instances
    ]
    assert _command_names(commands) == _calibration_command_names_after_collect_split(
        1, 1, prefix=["wsclean"]
    )
    assert commands[0][1:] == [
        "-j",
        "4",
        "-draw-model",
        "/data/calibration.skymodel",
        "-draw-spectral-terms",
        "2",
        "-name",
        "calibration_model",
        "-draw-centre",
        "12:00:00.0",
        "+45.00.00.0",
        "-draw-frequencies",
        "150000000.0",
        "1000000.0",
        "-size",
        "1024",
        "1024",
        "-scale",
        "0.001",
    ]
    assert "steps=[predict,applybeam,solve1,solve2]" in commands[1]
    assert f"predict.regions={tmp_path / 'field_facets_ds9.reg'}" in commands[1]
    assert (
        f"predict.images=[{tmp_path / 'calibration_model-term-0.fits'},"
        f"{tmp_path / 'calibration_model-term-1.fits'}]" in commands[1]
    )
    assert "solve1.reusemodel=[predict.*]" in commands[1]
    assert "solve2.reusemodel=[predict.*]" in commands[1]
    assert not any(token.startswith("solve1.sourcedb=") for token in commands[1])
    assert not any(token.startswith("solve1.directions=") for token in commands[1])


def test_run_calibrate_flow_supports_dd_wsclean_predict(
    tmp_path,
    fake_calibrate_shell_operation_cls,
    fake_direct_calibrate_helpers,
):
    input_parms = _use_local_timechunk_dirs(_dd_wsclean_predict_input_parms(), tmp_path)
    payload = calibrate_payload_from_inputs("dd", input_parms, tmp_path)

    outputs = run_flow_for_test(
        calibrate_flow,
        payload,
        execution_config=ExecutionConfig(task_runner="sync"),
        shell_operation_cls=fake_calibrate_shell_operation_cls,
    )

    assert outputs["combined_solutions"] == file_record(
        tmp_path / "combined_fast_medium1_phases.h5parm"
    )
    assert (tmp_path / "predict_field_facets_ds9.reg").is_file()
    assert (tmp_path / "wsclean_predict_chunk_1_band_1-term-0.fits").is_file()
    assert (tmp_path / "wsclean_predict_chunk_1_band_1-model.fits").is_file()
    assert (tmp_path / "input_chunk_0_wsclean_predict_1.ms").is_dir()
    assert (tmp_path / "input_chunk_1_wsclean_predict_2.ms").is_dir()
    assert fake_direct_calibrate_helpers["patch_names_from_region"] == [
        str(tmp_path / "predict_field_facets_ds9.reg")
    ]
    assert [
        call["max_bandwidth_hz"]
        for call in fake_direct_calibrate_helpers["frequency_chunks_for_ms"]
    ] == [2000000.0, 2000000.0]

    commands = _command_tokens(fake_calibrate_shell_operation_cls)
    assert _command_names(commands) == _calibration_command_names_after_collect_split(
        1,
        1,
        prefix=["wsclean", "wsclean", "wsclean", "wsclean", "wsclean", "wsclean"],
    )
    assert "-draw-model" in commands[0]
    assert commands[1][1:] == [
        "-j",
        "4",
        "-predict",
        "-facet-regions",
        str(tmp_path / "predict_field_facets_ds9.reg"),
        "-apply-time-frequency-smearing",
        "-model-column",
        "patch1",
        "-select-facets",
        "patch1",
        "-name",
        str(tmp_path / "wsclean_predict_chunk_1_band_1"),
        "-channel-range",
        "0",
        "3",
        "-model-storage-manager",
        "default",
        "-no-reorder",
        str(tmp_path / "input_chunk_0_wsclean_predict_1.ms"),
    ]

    solve_command = commands[6]
    assert f"msin={tmp_path / 'input_chunk_0_wsclean_predict_1.ms'}" in solve_command
    assert "steps=[solve1,solve2]" in solve_command
    assert "solve1.modeldatacolumns=[patch1,patch2]" in solve_command
    assert "solve2.modeldatacolumns=[patch1,patch2]" in solve_command
    assert f"predict.regions={tmp_path / 'predict_field_facets_ds9.reg'}" in solve_command
    assert not any(token.startswith("predict.images=") for token in solve_command)
    assert not any(token.startswith("solve1.sourcedb=") for token in solve_command)
    assert not any(token.startswith("solve1.directions=") for token in solve_command)
    assert "solve2.reusemodel=[solve1.*]" not in solve_command


def test_run_calibrate_flow_supports_dd_screen_generation(
    tmp_path, fake_calibrate_shell_operation_cls
):
    payload = calibrate_payload_from_inputs("dd", _dd_screen_input_parms(), tmp_path)

    outputs = run_flow_for_test(
        calibrate_flow,
        payload,
        execution_config=ExecutionConfig(task_runner="sync"),
        shell_operation_cls=fake_calibrate_shell_operation_cls,
    )

    assert outputs == {"combined_solutions": file_record(tmp_path / "combined_solutions.h5")}
    validate_output_record(outputs["combined_solutions"])
    assert (tmp_path / "calibration_model-term-0.fits").is_file()
    assert (tmp_path / "calibration_model-term-1.fits").is_file()
    assert (tmp_path / "field_facets_ds9.reg").is_file()

    commands = _command_tokens(fake_calibrate_shell_operation_cls)
    assert _command_names(commands) == [
        "wsclean",
        "DP3",
        "DP3",
    ]
    assert "solve.python.module=idg.idgcaldpstep_phase_only_dirac" in commands[1]
    assert "solve.python.class=IDGCalDPStepPhaseOnlyDirac" in commands[1]
    assert "solve.h5parm=idgcal_0" in commands[1]
    assert "solve.solintphase=3" in commands[1]
    assert f"solve.modelimage={tmp_path / 'calibration_model-term-0.fits'}" in commands[1]
    assert "solve.maxiter=4" in commands[1]
    assert "solve.antennaconstraint=[]" in commands[1]


def test_run_calibrate_flow_supports_dd_screen_generation_with_slow_gain(
    tmp_path, fake_calibrate_shell_operation_cls
):
    payload = calibrate_payload_from_inputs(
        "dd",
        _dd_screen_input_parms(has_slow_gain_solve=True),
        tmp_path,
    )

    outputs = run_flow_for_test(
        calibrate_flow,
        payload,
        execution_config=ExecutionConfig(task_runner="sync"),
        shell_operation_cls=fake_calibrate_shell_operation_cls,
    )

    assert outputs == {"combined_solutions": file_record(tmp_path / "combined_solutions.h5")}
    commands = _command_tokens(fake_calibrate_shell_operation_cls)
    assert _command_names(commands) == [
        "wsclean",
        "DP3",
        "DP3",
    ]
    assert "solve.python.module=idg.idgcaldpstep_rapthor_dirac" in commands[1]
    assert "solve.python.class=IDGCalDPStepRapthorDirac" in commands[1]
    assert "solve.polynomialdegamplitude=2" in commands[1]
    assert "solve.solintphase=3" in commands[1]
    assert "solve.solintamplitude=11" in commands[1]


def test_run_calibrate_flow_fails_when_image_predict_model_is_missing(
    tmp_path, fake_calibrate_shell_operation_cls
):
    class MissingModelShellOperation(fake_calibrate_shell_operation_cls):
        instances = []

        def run(self):
            tokens = shlex.split(self.kwargs["commands"][0])
            if tokens[0] == "wsclean":
                return "OK"
            return super().run()

    payload = calibrate_payload_from_inputs("dd", _dd_image_predict_input_parms(), tmp_path)

    with pytest.raises(FileNotFoundError, match="Calibration model image was not created"):
        run_flow_for_test(
            calibrate_flow,
            payload,
            execution_config=ExecutionConfig(task_runner="sync"),
            shell_operation_cls=MissingModelShellOperation,
        )


def test_run_calibrate_flow_fails_when_image_predict_region_is_missing(
    tmp_path, monkeypatch, fake_calibrate_shell_operation_cls
):
    def skip_region_file(*args, **kwargs):
        return None

    monkeypatch.setattr(
        calibrate_prediction,
        "make_ds9_region_from_skymodel",
        skip_region_file,
    )

    payload = calibrate_payload_from_inputs("dd", _dd_image_predict_input_parms(), tmp_path)

    with pytest.raises(FileNotFoundError, match="Calibration region file was not created"):
        run_flow_for_test(
            calibrate_flow,
            payload,
            execution_config=ExecutionConfig(task_runner="sync"),
            shell_operation_cls=fake_calibrate_shell_operation_cls,
        )


def test_run_calibrate_flow_supports_dd_image_predict_preapply(
    tmp_path, fake_calibrate_shell_operation_cls
):
    payload = calibrate_payload_from_inputs(
        "dd",
        _dd_image_predict_preapply_input_parms(tmp_path / "normalize_solutions.h5"),
        tmp_path,
    )

    outputs = run_flow_for_test(
        calibrate_flow,
        payload,
        execution_config=ExecutionConfig(task_runner="sync"),
        shell_operation_cls=fake_calibrate_shell_operation_cls,
    )

    assert outputs["combined_solutions"] == file_record(
        tmp_path / "combined_fast_medium1_phases.h5parm"
    )
    commands = [
        shlex.split(instance.kwargs["commands"][0])
        for instance in fake_calibrate_shell_operation_cls.instances
    ]
    assert [command[0] for command in commands][:2] == [
        "wsclean",
        "DP3",
    ]
    assert "steps=[predict,applybeam,applycal,solve1,solve2]" in commands[1]
    assert "applycal.steps=[fastphase,normalization]" in commands[1]
    assert "applycal.parmdb=/solutions/di_solutions.h5" in commands[1]
    assert f"applycal.normalization.parmdb={tmp_path / 'normalize_solutions.h5'}" in commands[1]
    assert "solve1.reusemodel=[predict.*]" in commands[1]
    assert "solve2.reusemodel=[predict.*]" in commands[1]


def test_run_calibrate_flow_skips_dd_source_adjustment_for_single_direction(
    tmp_path, fake_calibrate_shell_operation_cls, fake_direct_calibrate_helpers
):
    input_parms = _dd_fast_medium_input_parms()
    input_parms["calibrator_patch_names"] = ["patch1"]
    input_parms["calibrator_fluxes"] = [10.0]
    input_parms["solve_directions"] = ["patch1"]
    payload = calibrate_payload_from_inputs("dd", input_parms, tmp_path)

    outputs = run_flow_for_test(
        calibrate_flow,
        payload,
        execution_config=ExecutionConfig(task_runner="sync"),
        shell_operation_cls=fake_calibrate_shell_operation_cls,
    )

    assert outputs["combined_solutions"] == file_record(
        tmp_path / "combined_fast_medium1_phases.h5parm"
    )
    assert fake_direct_calibrate_helpers["adjust_h5parm_sources"] == []


def test_run_calibrate_flow_supports_dd_with_slow(tmp_path, fake_calibrate_shell_operation_cls):
    payload = calibrate_payload_from_inputs("dd", _dd_with_slow_input_parms(), tmp_path)

    outputs = run_flow_for_test(
        calibrate_flow,
        payload,
        execution_config=ExecutionConfig(task_runner="sync"),
        shell_operation_cls=fake_calibrate_shell_operation_cls,
    )

    assert outputs == {
        "combined_solutions": file_record(tmp_path / "combined_solutions.h5"),
        "fast_phase_solutions": file_record(tmp_path / "fast_phases.h5parm"),
        "medium1_phase_solutions": file_record(tmp_path / "medium1_phases.h5parm"),
        "slow_gain_solutions": file_record(tmp_path / "slow_gains.h5parm"),
        "medium2_phase_solutions": file_record(tmp_path / "medium2_phases.h5parm"),
        "fast_phase_plots": [file_record(tmp_path / "phase_solutions.png")],
        "medium1_phase_plots": [file_record(tmp_path / "medium1_phase_solutions.png")],
        "slow_phase_plots": [file_record(tmp_path / "slow_phase_solutions.png")],
        "slow_amp_plots": [file_record(tmp_path / "slow_amplitude_solutions.png")],
        "medium2_phase_plots": [file_record(tmp_path / "medium2_phase_solutions.png")],
    }
    for value in outputs.values():
        validate_output_record(value)

    commands = [
        shlex.split(instance.kwargs["commands"][0])
        for instance in fake_calibrate_shell_operation_cls.instances
    ]
    assert _command_names(commands) == _calibration_command_names_after_collect_split(1, 1, 2, 1)
    assert "steps=[solve1,solve2,solve3,solve4]" in commands[0]
    assert "solve3.initialsolutions.soltab=[phase000,amplitude000]" in commands[0]
    assert "solve3.keepmodel=true" in commands[0]


@pytest.mark.parametrize("solution_combine_mode", ["p1p2a2_scalar", "p1p2a2_diagonal"])
def test_run_calibrate_flow_supports_dd_with_slow_without_medium2(
    tmp_path, fake_calibrate_shell_operation_cls, solution_combine_mode
):
    input_parms = _dd_with_slow_input_parms()
    input_parms["dp3_steps"] = "[solve1,solve2,solve3]"
    input_parms["solution_combine_mode"] = solution_combine_mode
    payload = calibrate_payload_from_inputs("dd", input_parms, tmp_path)

    outputs = run_flow_for_test(
        calibrate_flow,
        payload,
        execution_config=ExecutionConfig(task_runner="sync"),
        shell_operation_cls=fake_calibrate_shell_operation_cls,
    )

    assert outputs == {
        "combined_solutions": file_record(tmp_path / "combined_solutions.h5"),
        "fast_phase_solutions": file_record(tmp_path / "fast_phases.h5parm"),
        "medium1_phase_solutions": file_record(tmp_path / "medium1_phases.h5parm"),
        "slow_gain_solutions": file_record(tmp_path / "slow_gains.h5parm"),
        "fast_phase_plots": [file_record(tmp_path / "phase_solutions.png")],
        "medium1_phase_plots": [file_record(tmp_path / "medium1_phase_solutions.png")],
        "slow_phase_plots": [file_record(tmp_path / "slow_phase_solutions.png")],
        "slow_amp_plots": [file_record(tmp_path / "slow_amplitude_solutions.png")],
    }

    commands = [
        shlex.split(instance.kwargs["commands"][0])
        for instance in fake_calibrate_shell_operation_cls.instances
    ]
    assert _command_names(commands) == _calibration_command_names_after_collect_split(1, 1, 2)
    assert "steps=[solve1,solve2,solve3]" in commands[0]
    assert all("solve4.h5parm" not in token for token in commands[0])


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


def test_calibrate_prefect_flow_entrypoint_runs_screen_generation(
    tmp_path, monkeypatch, fake_calibrate_shell_operation_cls
):
    monkeypatch.setattr(
        "rapthor.execution.shell._load_shell_operation_cls",
        lambda: fake_calibrate_shell_operation_cls,
    )

    with prefect_test_harness(server_startup_timeout=None):
        outputs = calibrate_flow(
            calibrate_payload_from_inputs("dd", _dd_screen_input_parms(), tmp_path),
            execution_config=ExecutionConfig(task_runner="sync"),
        )

    assert outputs["combined_solutions"] == file_record(tmp_path / "combined_solutions.h5")
    assert _command_names(_command_tokens(fake_calibrate_shell_operation_cls)) == [
        "wsclean",
        "DP3",
        "DP3",
    ]


def test_calibrate_di_operation_run_uses_prefect_flow(
    tmp_path, monkeypatch, fake_calibrate_shell_operation_cls
):
    monkeypatch.setattr(
        "rapthor.execution.shell._load_shell_operation_cls",
        lambda: fake_calibrate_shell_operation_cls,
    )
    monkeypatch.setattr(
        "rapthor.lib.miscellaneous.get_flagged_solution_fraction",
        lambda *args, **kwargs: 0.0,
    )

    field = CalibrateFieldStub(tmp_path)
    operation = Calibrate("di", field, index=1)

    with prefect_test_harness(server_startup_timeout=None):
        operation.run()

    expected_outputs = _expected_di_fulljones_operation_outputs(operation)
    solutions_dir = Path(field.parset["dir_working"]) / "solutions" / "calibrate_di_1"
    plots_dir = Path(field.parset["dir_working"]) / "plots" / "calibrate_di_1"

    assert operation.outputs == expected_outputs
    assert json.loads(Path(operation.outputs_file).read_text()) == expected_outputs
    assert Path(operation.done_file).is_file()
    assert Path(operation.pipeline_inputs_file).is_file()
    assert field.fulljones_h5parm_filename == str(solutions_dir / "fulljones-solutions.h5")
    assert (solutions_dir / "fulljones-solutions.h5").is_file()
    assert (plots_dir / "fulljones_phase_solutions.png").is_file()
    assert field.scan_h5parms_calls == 1
    assert len(fake_calibrate_shell_operation_cls.instances) == 4


def test_calibrate_di_scalar_operation_run_uses_prefect_flow(
    tmp_path, monkeypatch, fake_calibrate_shell_operation_cls
):
    monkeypatch.setattr(
        "rapthor.execution.shell._load_shell_operation_cls",
        lambda: fake_calibrate_shell_operation_cls,
    )
    monkeypatch.setattr(
        "rapthor.lib.miscellaneous.get_flagged_solution_fraction",
        lambda *args, **kwargs: 0.0,
    )

    field = CalibrateFieldStub(tmp_path)
    field.calibration_strategy = {"di": ["fast_phase", "medium_phase"]}
    operation = Calibrate("di", field, index=1)

    with prefect_test_harness(server_startup_timeout=None):
        operation.run()

    expected_outputs = _expected_di_scalar_phase_operation_outputs(operation)
    solutions_dir = Path(field.parset["dir_working"]) / "solutions" / "calibrate_di_1"
    plots_dir = Path(field.parset["dir_working"]) / "plots" / "calibrate_di_1"

    assert operation.outputs == expected_outputs
    assert json.loads(Path(operation.outputs_file).read_text()) == expected_outputs
    assert Path(operation.done_file).is_file()
    assert Path(operation.pipeline_inputs_file).is_file()
    assert field.h5parm_filename == str(solutions_dir / "di-solutions.h5")
    assert field.di_h5parm_filename == str(solutions_dir / "di-solutions.h5")
    assert field.di_fast_phases_h5parm_filename == str(solutions_dir / "di-solutions-fast-phase.h5")
    assert field.di_medium1_phases_h5parm_filename == str(
        solutions_dir / "di-solutions-medium1-phase.h5"
    )
    assert field.fulljones_h5parm_filename is None
    assert (solutions_dir / "di-solutions.h5").is_file()
    assert (solutions_dir / "di-solutions-fast-phase.h5").is_file()
    assert (solutions_dir / "di-solutions-medium1-phase.h5").is_file()
    assert (plots_dir / "phase_solutions.png").is_file()
    assert (plots_dir / "medium1_phase_solutions.png").is_file()
    assert field.scan_h5parms_calls == 1
    assert len(fake_calibrate_shell_operation_cls.instances) == 6


def test_calibrate_di_operation_run_reuses_prefect_outputs_when_done(
    tmp_path, monkeypatch, fake_calibrate_shell_operation_cls
):
    monkeypatch.setattr(
        "rapthor.execution.shell._load_shell_operation_cls",
        lambda: fake_calibrate_shell_operation_cls,
    )
    monkeypatch.setattr(
        "rapthor.lib.miscellaneous.get_flagged_solution_fraction",
        lambda *args, **kwargs: 0.0,
    )

    field = CalibrateFieldStub(tmp_path)
    operation = Calibrate("di", field, index=1)
    expected_outputs = _expected_di_fulljones_operation_outputs(operation)
    _materialize_calibrate_operation_outputs(expected_outputs)
    Path(operation.done_file).touch()
    Path(operation.outputs_file).write_text(json.dumps(expected_outputs))

    operation.run()

    solutions_dir = Path(field.parset["dir_working"]) / "solutions" / "calibrate_di_1"
    plots_dir = Path(field.parset["dir_working"]) / "plots" / "calibrate_di_1"

    assert operation.outputs == expected_outputs
    assert fake_calibrate_shell_operation_cls.instances == []
    assert field.fulljones_h5parm_filename == str(solutions_dir / "fulljones-solutions.h5")
    assert (solutions_dir / "fulljones-solutions.h5").is_file()
    assert (plots_dir / "fulljones_phase_solutions.png").is_file()
    assert field.scan_h5parms_calls == 1


@pytest.mark.parametrize(
    "shell_operation_cls, expected_message",
    [
        pytest.param(FailingShellOperation, "calibrate failed", id="shell-failure"),
        pytest.param(
            NoOutputShellOperation,
            "DI solve1 full-Jones h5parm",
            id="missing-fulljones-output",
        ),
    ],
)
def test_calibrate_di_operation_run_failure_does_not_mark_done(
    tmp_path, monkeypatch, shell_operation_cls, expected_message
):
    monkeypatch.setattr(
        "rapthor.execution.shell._load_shell_operation_cls",
        lambda: shell_operation_cls,
    )
    monkeypatch.setattr(
        "rapthor.lib.miscellaneous.get_flagged_solution_fraction",
        lambda *args, **kwargs: 0.0,
    )

    field = CalibrateFieldStub(tmp_path)
    operation = Calibrate("di", field, index=1)

    with (
        prefect_test_harness(server_startup_timeout=None),
        pytest.raises((FileNotFoundError, RuntimeError), match=expected_message),
    ):
        operation.run()

    assert Path(operation.pipeline_inputs_file).is_file()
    assert not Path(operation.done_file).exists()
    assert not Path(operation.outputs_file).exists()
    assert operation.outputs == {}
    assert field.fulljones_h5parm_filename is None
    assert field.h5parm_filename is None
    assert field.di_h5parm_filename is None
    assert field.scan_h5parms_calls == 0
    assert field.calibration_diagnostics == []


def test_calibrate_di_scalar_operation_run_reuses_prefect_outputs_when_done(
    tmp_path, monkeypatch, fake_calibrate_shell_operation_cls
):
    monkeypatch.setattr(
        "rapthor.execution.shell._load_shell_operation_cls",
        lambda: fake_calibrate_shell_operation_cls,
    )
    monkeypatch.setattr(
        "rapthor.lib.miscellaneous.get_flagged_solution_fraction",
        lambda *args, **kwargs: 0.0,
    )

    field = CalibrateFieldStub(tmp_path)
    field.calibration_strategy = {"di": ["fast_phase", "medium_phase"]}
    operation = Calibrate("di", field, index=1)
    expected_outputs = _expected_di_scalar_phase_operation_outputs(operation)
    _materialize_calibrate_operation_outputs(expected_outputs)
    Path(operation.done_file).touch()
    Path(operation.outputs_file).write_text(json.dumps(expected_outputs))

    operation.run()

    solutions_dir = Path(field.parset["dir_working"]) / "solutions" / "calibrate_di_1"
    plots_dir = Path(field.parset["dir_working"]) / "plots" / "calibrate_di_1"

    assert operation.outputs == expected_outputs
    assert fake_calibrate_shell_operation_cls.instances == []
    assert field.h5parm_filename == str(solutions_dir / "di-solutions.h5")
    assert field.di_h5parm_filename == str(solutions_dir / "di-solutions.h5")
    assert field.di_fast_phases_h5parm_filename == str(solutions_dir / "di-solutions-fast-phase.h5")
    assert field.di_medium1_phases_h5parm_filename == str(
        solutions_dir / "di-solutions-medium1-phase.h5"
    )
    assert (solutions_dir / "di-solutions.h5").is_file()
    assert (solutions_dir / "di-solutions-fast-phase.h5").is_file()
    assert (solutions_dir / "di-solutions-medium1-phase.h5").is_file()
    assert (plots_dir / "phase_solutions.png").is_file()
    assert (plots_dir / "medium1_phase_solutions.png").is_file()
    assert field.scan_h5parms_calls == 1


def test_calibrate_dd_fast_medium_operation_run_uses_prefect_flow(
    tmp_path, monkeypatch, fake_calibrate_shell_operation_cls
):
    monkeypatch.setattr(
        "rapthor.execution.shell._load_shell_operation_cls",
        lambda: fake_calibrate_shell_operation_cls,
    )
    monkeypatch.setattr(
        "rapthor.lib.miscellaneous.get_flagged_solution_fraction",
        lambda *args, **kwargs: 0.0,
    )
    _patch_dd_model_metadata(monkeypatch)

    field = CalibrateFieldStub(tmp_path)
    operation = Calibrate("dd", field, index=1)

    with prefect_test_harness(server_startup_timeout=None):
        operation.run()

    expected_outputs = _expected_dd_fast_medium_operation_outputs(operation)
    solutions_dir = Path(field.parset["dir_working"]) / "solutions" / "calibrate_1"
    plots_dir = Path(field.parset["dir_working"]) / "plots" / "calibrate_1"

    assert operation.outputs == expected_outputs
    assert json.loads(Path(operation.outputs_file).read_text()) == expected_outputs
    assert Path(operation.done_file).is_file()
    assert Path(operation.pipeline_inputs_file).is_file()
    assert field.h5parm_filename == str(solutions_dir / "field-solutions.h5")
    assert field.dd_h5parm_filename == str(solutions_dir / "field-solutions.h5")
    assert field.fast_phases_h5parm_filename == str(solutions_dir / "field-solutions-fast-phase.h5")
    assert (solutions_dir / "field-solutions.h5").read_text() == "collected"
    assert (solutions_dir / "field-solutions-fast-phase.h5").read_text() == "collected"
    assert (solutions_dir / "field-solutions-medium1-phase.h5").read_text() == "collected"
    assert (plots_dir / "phase_solutions.png").is_file()
    assert (plots_dir / "medium1_phase_solutions.png").is_file()
    assert field.calibration_diagnostics == [{"cycle_number": 1, "solution_flagged_fraction": 0.0}]
    assert field.scan_h5parms_calls == 1
    assert len(fake_calibrate_shell_operation_cls.instances) == 6


def test_calibrate_dd_operation_run_passes_only_current_cycle_initial_solutions_to_dp3(
    tmp_path, monkeypatch, fake_calibrate_shell_operation_cls
):
    monkeypatch.setattr(
        "rapthor.execution.shell._load_shell_operation_cls",
        lambda: fake_calibrate_shell_operation_cls,
    )
    monkeypatch.setattr(
        "rapthor.lib.miscellaneous.get_flagged_solution_fraction",
        lambda *args, **kwargs: 0.0,
    )
    _patch_dd_model_metadata(monkeypatch)

    field = CalibrateFieldStub(tmp_path)
    field.calibration_strategy = {"dd": ["fast_phase", "medium_phase", "slow_gains"]}
    field._calibration_strategy_defaulted = False

    current_solution_dir = Path(field.parset["dir_working"]) / "solutions" / "calibrate_1"
    future_solution_dir = Path(field.parset["dir_working"]) / "solutions" / "calibrate_2"
    current_solution_dir.mkdir(parents=True)
    future_solution_dir.mkdir(parents=True)

    current_fast = current_solution_dir / "field-solutions-fast-phase.h5"
    future_medium = future_solution_dir / "field-solutions-medium1-phase.h5"
    future_slow = future_solution_dir / "field-solutions-slow-gain.h5"
    for path in (current_fast, future_medium, future_slow):
        path.write_text("h5parm")

    field.fast_phases_h5parm_filename = str(current_fast)
    field.medium1_phases_h5parm_filename = str(future_medium)
    field.slow_gains_h5parm_filename = str(future_slow)

    operation = Calibrate("dd", field, index=1)

    with prefect_test_harness(server_startup_timeout=None):
        operation.run()

    dp3_command = next(
        command
        for command in _command_tokens(fake_calibrate_shell_operation_cls)
        if command[0] == "DP3"
    )
    dp3_arguments = _command_arguments(dp3_command)

    assert dp3_arguments["solve1.initialsolutions.h5parm"] == str(current_fast)
    assert "solve2.initialsolutions.h5parm" not in dp3_arguments
    assert "solve3.initialsolutions.h5parm" not in dp3_arguments
    assert all(str(future_medium) not in token for token in dp3_command)
    assert all(str(future_slow) not in token for token in dp3_command)


def test_calibrate_dd_fast_medium_operation_run_reuses_prefect_outputs_when_done(
    tmp_path, monkeypatch, fake_calibrate_shell_operation_cls
):
    monkeypatch.setattr(
        "rapthor.execution.shell._load_shell_operation_cls",
        lambda: fake_calibrate_shell_operation_cls,
    )
    monkeypatch.setattr(
        "rapthor.lib.miscellaneous.get_flagged_solution_fraction",
        lambda *args, **kwargs: 0.0,
    )
    _patch_dd_model_metadata(monkeypatch)

    field = CalibrateFieldStub(tmp_path)
    operation = Calibrate("dd", field, index=1)
    expected_outputs = _expected_dd_fast_medium_operation_outputs(operation)
    _materialize_calibrate_operation_outputs(expected_outputs)
    Path(operation.done_file).touch()
    Path(operation.outputs_file).write_text(json.dumps(expected_outputs))

    operation.run()

    solutions_dir = Path(field.parset["dir_working"]) / "solutions" / "calibrate_1"
    plots_dir = Path(field.parset["dir_working"]) / "plots" / "calibrate_1"

    assert operation.outputs == expected_outputs
    assert fake_calibrate_shell_operation_cls.instances == []
    assert field.h5parm_filename == str(solutions_dir / "field-solutions.h5")
    assert field.dd_h5parm_filename == str(solutions_dir / "field-solutions.h5")
    assert field.fast_phases_h5parm_filename == str(solutions_dir / "field-solutions-fast-phase.h5")
    assert (solutions_dir / "field-solutions.h5").is_file()
    assert (solutions_dir / "field-solutions-fast-phase.h5").is_file()
    assert (solutions_dir / "field-solutions-medium1-phase.h5").is_file()
    assert (plots_dir / "phase_solutions.png").is_file()
    assert (plots_dir / "medium1_phase_solutions.png").is_file()
    assert field.calibration_diagnostics == [{"cycle_number": 1, "solution_flagged_fraction": 0.0}]
    assert field.scan_h5parms_calls == 1


def test_calibrate_dd_preapply_operation_run_uses_prefect_flow(
    tmp_path, monkeypatch, fake_calibrate_shell_operation_cls
):
    monkeypatch.setattr(
        "rapthor.execution.shell._load_shell_operation_cls",
        lambda: fake_calibrate_shell_operation_cls,
    )
    monkeypatch.setattr(
        "rapthor.lib.miscellaneous.get_flagged_solution_fraction",
        lambda *args, **kwargs: 0.0,
    )
    _patch_dd_model_metadata(monkeypatch)

    field = CalibrateFieldStub(tmp_path)
    di_h5parm, fulljones_h5parm, normalize_h5parm = _configure_dd_preapply_products(field, tmp_path)
    operation = Calibrate("dd", field, index=1)

    with prefect_test_harness(server_startup_timeout=None):
        operation.run()

    expected_outputs = _expected_dd_fast_medium_operation_outputs(operation)
    solutions_dir = Path(field.parset["dir_working"]) / "solutions" / "calibrate_1"

    assert operation.outputs == expected_outputs
    assert json.loads(Path(operation.outputs_file).read_text()) == expected_outputs
    assert Path(operation.done_file).is_file()
    assert field.h5parm_filename == str(solutions_dir / "field-solutions.h5")
    assert field.fast_phases_h5parm_filename == str(solutions_dir / "field-solutions-fast-phase.h5")
    commands = _command_tokens(fake_calibrate_shell_operation_cls)
    assert _command_names(commands) == _calibration_command_names_after_collect_split(1, 1)
    assert "steps=[applycal,solve1,solve2]" in commands[0]
    assert "applycal.steps=[fastphase,slowgain,fulljones,normalization]" in commands[0]
    assert f"applycal.parmdb={di_h5parm}" in commands[0]
    assert f"applycal.fulljones.parmdb={fulljones_h5parm}" in commands[0]
    assert f"applycal.normalization.parmdb={normalize_h5parm}" in commands[0]
    assert "solve2.reusemodel=[solve1.*]" in commands[0]


def test_calibrate_dd_image_predict_operation_run_uses_prefect_flow(
    tmp_path, monkeypatch, fake_calibrate_shell_operation_cls
):
    monkeypatch.setattr(
        "rapthor.execution.shell._load_shell_operation_cls",
        lambda: fake_calibrate_shell_operation_cls,
    )
    monkeypatch.setattr(
        "rapthor.lib.miscellaneous.get_flagged_solution_fraction",
        lambda *args, **kwargs: 0.0,
    )
    _patch_dd_model_metadata(monkeypatch)

    field = CalibrateFieldStub(tmp_path)
    field.use_image_based_predict = True
    operation = Calibrate("dd", field, index=1)

    with prefect_test_harness(server_startup_timeout=None):
        operation.run()

    expected_outputs = _expected_dd_fast_medium_operation_outputs(operation)
    pipeline_dir = Path(operation.pipeline_working_dir)

    assert operation.outputs == expected_outputs
    assert (pipeline_dir / "calibration_model-term-0.fits").is_file()
    assert (pipeline_dir / "field_facets_ds9.reg").is_file()
    commands = _command_tokens(fake_calibrate_shell_operation_cls)
    assert [command[0] for command in commands][:2] == ["wsclean", "DP3"]
    assert commands[0][1:] == [
        "-j",
        "4",
        "-draw-model",
        field.calibration_skymodel_file,
        "-draw-spectral-terms",
        "1",
        "-name",
        "calibration_model",
        "-draw-centre",
        "12:00:00.0",
        "+45.00.00.0",
        "-draw-frequencies",
        "150000000.0",
        "1000000.0",
        "-size",
        "1024",
        "1024",
        "-scale",
        "0.001",
    ]
    assert "steps=[predict,applybeam,solve1,solve2]" in commands[1]
    assert f"predict.regions={pipeline_dir / 'field_facets_ds9.reg'}" in commands[1]
    assert f"predict.images=[{pipeline_dir / 'calibration_model-term-0.fits'}]" in commands[1]
    assert "solve1.reusemodel=[predict.*]" in commands[1]
    assert "solve2.reusemodel=[predict.*]" in commands[1]
    assert not any(token.startswith("solve1.sourcedb=") for token in commands[1])


def test_calibrate_dd_screen_operation_run_uses_prefect_flow(
    tmp_path, monkeypatch, fake_calibrate_shell_operation_cls
):
    monkeypatch.setattr(
        "rapthor.execution.shell._load_shell_operation_cls",
        lambda: fake_calibrate_shell_operation_cls,
    )
    flagged_calls = []

    def fake_flagged_fraction(path, **kwargs):
        flagged_calls.append((path, kwargs))
        return 0.0

    monkeypatch.setattr(
        "rapthor.lib.miscellaneous.get_flagged_solution_fraction",
        fake_flagged_fraction,
    )
    _patch_dd_model_metadata(monkeypatch)

    field = CalibrateFieldStub(tmp_path)
    field.generate_screens = True
    operation = Calibrate("dd", field, index=1)

    with prefect_test_harness(server_startup_timeout=None):
        operation.run()

    expected_outputs = _expected_dd_screen_operation_outputs(operation)
    solutions_dir = Path(field.parset["dir_working"]) / "solutions" / "calibrate_1"
    pipeline_dir = Path(operation.pipeline_working_dir)

    assert operation.outputs == expected_outputs
    assert json.loads(Path(operation.outputs_file).read_text()) == expected_outputs
    assert Path(operation.done_file).is_file()
    assert Path(operation.pipeline_inputs_file).is_file()
    assert field.h5parm_filename == str(solutions_dir / "field-solutions.h5")
    assert field.dd_h5parm_filename == str(solutions_dir / "field-solutions.h5")
    assert (solutions_dir / "field-solutions.h5").read_text() == "screens"
    assert not (solutions_dir / "field-solutions-fast-phase.h5").exists()
    assert (pipeline_dir / "calibration_model-term-0.fits").is_file()
    assert (pipeline_dir / "field_facets_ds9.reg").is_file()
    assert flagged_calls == [(field.h5parm_filename, {"solsetname": "coefficients000"})]
    assert field.calibration_diagnostics == [{"cycle_number": 1, "solution_flagged_fraction": 0.0}]
    assert field.scan_h5parms_calls == 1

    commands = _command_tokens(fake_calibrate_shell_operation_cls)
    assert _command_names(commands) == [
        "wsclean",
        "DP3",
        "DP3",
    ]
    assert "solve.python.module=idg.idgcaldpstep_phase_only_dirac" in commands[1]
    assert "solve.h5parm=idgcal_0" in commands[1]
    assert "solve.solintphase=5" in commands[1]
    assert f"solve.modelimage={pipeline_dir / 'calibration_model-term-0.fits'}" in commands[1]


def test_calibrate_dd_slow_source_adjusted_operation_run_uses_prefect_flow(
    tmp_path, monkeypatch, fake_calibrate_shell_operation_cls
):
    monkeypatch.setattr(
        "rapthor.execution.shell._load_shell_operation_cls",
        lambda: fake_calibrate_shell_operation_cls,
    )
    monkeypatch.setattr(
        "rapthor.lib.miscellaneous.get_flagged_solution_fraction",
        lambda *args, **kwargs: 0.0,
    )
    _patch_dd_model_metadata(monkeypatch)

    field = CalibrateFieldStub(tmp_path)
    field.calibration_strategy = {
        "dd": ["fast_phase", "medium_phase", "slow_gains", "medium_phase"]
    }
    field._calibration_strategy_defaulted = False
    _configure_dd_multidirection(field)
    operation = Calibrate("dd", field, index=1)

    with prefect_test_harness(server_startup_timeout=None):
        operation.run()

    expected_outputs = _expected_dd_slow_operation_outputs(operation)
    solutions_dir = Path(field.parset["dir_working"]) / "solutions" / "calibrate_1"
    plots_dir = Path(field.parset["dir_working"]) / "plots" / "calibrate_1"

    assert operation.outputs == expected_outputs
    assert json.loads(Path(operation.outputs_file).read_text()) == expected_outputs
    assert field.h5parm_filename == str(solutions_dir / "field-solutions.h5")
    assert field.dd_h5parm_filename == str(solutions_dir / "field-solutions.h5")
    assert field.fast_phases_h5parm_filename == str(solutions_dir / "field-solutions-fast-phase.h5")
    assert field.medium1_phases_h5parm_filename == str(
        solutions_dir / "field-solutions-medium1-phase.h5"
    )
    assert field.medium2_phases_h5parm_filename == str(
        solutions_dir / "field-solutions-medium2-phase.h5"
    )
    assert field.slow_gains_h5parm_filename == str(solutions_dir / "field-solutions-slow-gain.h5")
    assert (solutions_dir / "field-solutions.h5").read_text() == "adjusted"
    assert (solutions_dir / "field-solutions-fast-phase.h5").read_text() == "collected"
    assert (solutions_dir / "field-solutions-medium1-phase.h5").read_text() == "collected"
    assert (solutions_dir / "field-solutions-medium2-phase.h5").read_text() == "collected"
    assert (solutions_dir / "field-solutions-slow-gain.h5").read_text() == "processed"
    assert (plots_dir / "phase_solutions.png").is_file()
    assert (plots_dir / "medium1_phase_solutions.png").is_file()
    assert (plots_dir / "medium2_phase_solutions.png").is_file()
    assert (plots_dir / "slow_phase_solutions.png").is_file()
    assert (plots_dir / "slow_amplitude_solutions.png").is_file()
    assert field.calibration_diagnostics == [{"cycle_number": 1, "solution_flagged_fraction": 0.0}]
    commands = _command_tokens(fake_calibrate_shell_operation_cls)
    assert _command_names(commands) == _calibration_command_names_after_collect_split(1, 1, 2, 1)
    assert "steps=[solve1,solve2,solve3,solve4]" in commands[0]
    assert "solve3.h5parm=slow_gain_0.h5parm" in commands[0]
    assert field.scan_h5parms_calls == 1


def test_calibrate_prefect_tasks_submit_all_chunks_before_collect(monkeypatch, tmp_path):
    payload = calibrate_payload_from_inputs("di", _di_fulljones_input_parms(), tmp_path)
    events = []
    submitted_indexes = []

    class FakeFuture:
        def __init__(self, index):
            self.index = index

        def result(self):
            events.append(f"result-{self.index}")
            return {"solve1": file_record(tmp_path / f"fulljones_gain_{self.index}.h5parm")}

    def fake_submit(payload_arg, chunk, execution_config=None):
        assert payload_arg is payload
        assert execution_config == ExecutionConfig(task_runner="sync")
        future_index = len(submitted_indexes)
        submitted_indexes.append(future_index)
        events.append(f"submit-{chunk['output_h5parm']}")
        return FakeFuture(future_index)

    class FakeChunkTask:
        def __init__(self, task_run_name):
            self.task_run_name = task_run_name

        def submit(self, payload_arg, chunk, execution_config=None):
            events.append(f"task-name-{self.task_run_name}")
            return fake_submit(payload_arg, chunk, execution_config=execution_config)

    class FakeCollectFuture:
        def result(self):
            events.append("collect-result")
            return {
                "solve_key": "solve1",
                "solve_slot": payload["chunks"][0]["solve_slots"][0],
                "collected_record": file_record(tmp_path / "fulljones_solutions.h5"),
            }

    class FakeCollectTask:
        def __init__(self, task_run_name):
            self.task_run_name = task_run_name

        def submit(self, payload_arg, solve_records, solve_slot, execution_config=None):
            events.append(f"task-name-{self.task_run_name}")
            assert payload_arg is payload
            assert execution_config == ExecutionConfig(task_runner="sync")
            assert solve_records == [
                {"solve1": file_record(tmp_path / "fulljones_gain_0.h5parm")},
                {"solve1": file_record(tmp_path / "fulljones_gain_1.h5parm")},
            ]
            assert solve_slot is payload["chunks"][0]["solve_slots"][0]
            events.append("collect-submit")
            return FakeCollectFuture()

    class FakeProcessFuture:
        def result(self):
            events.append("process-result")
            return {
                "solve_key": "solve1",
                "solve_slot": payload["chunks"][0]["solve_slots"][0],
                "solution_record": file_record(tmp_path / "fulljones_solutions.h5"),
                "combine_record": file_record(tmp_path / "fulljones_solutions.h5"),
            }

    class FakeProcessTask:
        def __init__(self, task_run_name):
            self.task_run_name = task_run_name

        def submit(self, payload_arg, collected_product):
            events.append(f"task-name-{self.task_run_name}")
            assert payload_arg is payload
            assert collected_product.result() == {
                "solve_key": "solve1",
                "solve_slot": payload["chunks"][0]["solve_slots"][0],
                "collected_record": file_record(tmp_path / "fulljones_solutions.h5"),
            }
            events.append("process-submit")
            return FakeProcessFuture()

    class FakePlotFuture:
        def result(self):
            events.append("plot-result")
            return {
                "solve_key": "solve1",
                "plots": {"fulljones_phase_plots": []},
            }

    class FakePlotTask:
        def __init__(self, task_run_name):
            self.task_run_name = task_run_name

        def submit(self, payload_arg, processed_product, execution_config=None):
            events.append(f"task-name-{self.task_run_name}")
            assert payload_arg is payload
            assert execution_config == ExecutionConfig(task_runner="sync")
            assert processed_product.result() == {
                "solve_key": "solve1",
                "solve_slot": payload["chunks"][0]["solve_slots"][0],
                "solution_record": file_record(tmp_path / "fulljones_solutions.h5"),
                "combine_record": file_record(tmp_path / "fulljones_solutions.h5"),
            }
            events.append("plot-submit")
            return FakePlotFuture()

    class FakeFinalizeFuture:
        def result(self):
            events.append("finalize-result")
            return {"combined_solutions": file_record(tmp_path / "fulljones_solutions.h5")}

    class FakeFinalizeTask:
        def __init__(self, task_run_name):
            self.task_run_name = task_run_name

        def submit(self, payload_arg, processed_products, plot_products, active_solution):
            events.append(f"task-name-{self.task_run_name}")
            assert payload_arg is payload
            assert len(processed_products) == 1
            assert len(plot_products) == 1
            assert active_solution is processed_products[0]
            events.append("finalize-submit")
            return FakeFinalizeFuture()

    def fake_chunk_with_options(task_run_name, **_options):
        return FakeChunkTask(task_run_name)

    def fake_collect_with_options(task_run_name, **_options):
        return FakeCollectTask(task_run_name)

    def fake_process_with_options(task_run_name, **_options):
        return FakeProcessTask(task_run_name)

    def fake_plot_with_options(task_run_name, **_options):
        return FakePlotTask(task_run_name)

    def fake_finalize_with_options(task_run_name, **_options):
        return FakeFinalizeTask(task_run_name)

    monkeypatch.setattr(
        calibrate_module.calibrate_chunk_task,
        "with_options",
        fake_chunk_with_options,
    )
    monkeypatch.setattr(
        calibrate_module.collect_h5parms_task,
        "with_options",
        fake_collect_with_options,
    )
    monkeypatch.setattr(
        calibrate_module.process_solutions_task,
        "with_options",
        fake_process_with_options,
    )
    monkeypatch.setattr(
        calibrate_module.plot_solutions_task,
        "with_options",
        fake_plot_with_options,
    )
    monkeypatch.setattr(
        calibrate_module.finalize_solutions_task,
        "with_options",
        fake_finalize_with_options,
    )

    outputs = calibrate_module._run_calibrate_prefect_tasks(
        payload,
        execution_config=ExecutionConfig(task_runner="sync"),
    )

    assert outputs == {"combined_solutions": file_record(tmp_path / "fulljones_solutions.h5")}
    assert events == [
        "task-name-solve_chunk_1",
        "submit-fulljones_gain_0.h5parm",
        "task-name-solve_chunk_2",
        "submit-fulljones_gain_1.h5parm",
        "result-0",
        "result-1",
        "task-name-collect_full_jones",
        "collect-submit",
        "task-name-process_full_jones",
        "collect-result",
        "process-submit",
        "task-name-plot_full_jones",
        "process-result",
        "plot-submit",
        "task-name-finalize_solutions",
        "finalize-submit",
        "finalize-result",
    ]


def test_run_calibrate_flow_fails_when_expected_output_is_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="DI solve1 full-Jones h5parm"):
        run_flow_for_test(
            calibrate_flow,
            calibrate_payload_from_inputs("di", _di_fulljones_input_parms(), tmp_path),
            execution_config=ExecutionConfig(task_runner="sync"),
            shell_operation_cls=NoOutputShellOperation,
        )


def test_calibrate_reference_output_fixture_matches_output_contract():
    outputs = json.loads((FIXTURE_DIR / "output_reference.json").read_text())

    for value in outputs["calibrate_di_fulljones"].values():
        validate_output_record(value)
    for value in outputs["calibrate_di_scalar_phase"].values():
        validate_output_record(value)
    for value in outputs["calibrate_dd_fast_phase"].values():
        validate_output_record(value)
    for value in outputs["calibrate_dd_fast_medium"].values():
        validate_output_record(value)
    for value in outputs["calibrate_dd_with_slow"].values():
        validate_output_record(value)
