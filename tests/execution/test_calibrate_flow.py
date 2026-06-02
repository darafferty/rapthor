import json
import shlex
from pathlib import Path

import pytest
from prefect.testing.utilities import prefect_test_harness

import rapthor.execution.flows.calibrate as calibrate_module
from rapthor.execution.config import ExecutionConfig
from rapthor.execution.flows.calibrate import (
    build_adjust_h5parm_sources_command,
    build_combine_h5parms_command,
    build_collect_h5parms_command,
    build_ddecal_solve_command,
    build_plot_solutions_command,
    build_process_gains_command,
    calibrate_chunk_task,
    calibrate_flow,
    calibrate_payload_from_inputs,
    normalized_adjust_h5parm_sources_command,
    normalized_combine_h5parms_command,
    normalized_collect_h5parms_command,
    normalized_ddecal_solve_command,
    normalized_draw_model_command,
    normalized_make_region_file_command,
    normalized_plot_solutions_command,
    normalized_process_gains_command,
    run_calibrate_flow,
)
from rapthor.execution.outputs import directory_record, file_record, validate_output_record
from rapthor.operations.calibrate import Calibrate

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
            if tokens[0] == "wsclean":
                root = tokens[tokens.index("-name") + 1]
                numterms = int(tokens[tokens.index("-draw-spectral-terms") + 1])
                for index in range(numterms):
                    (cwd / f"{root}-term-{index}.fits").write_text("model")
                return "OK"
            if tokens[0] == "make_region_file.py":
                (cwd / tokens[6]).write_text("region")
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
            if tokens[0] == "combine_h5parms.py":
                output_path = cwd / tokens[3]
                output_path.write_text("combined")
                return "OK"
            if tokens[0] == "process_gains.py":
                h5parm = Path(tokens[2])
                output_path = h5parm if h5parm.is_absolute() else cwd / h5parm
                output_path.write_text("processed")
                return "OK"
            if tokens[0] == "adjust_h5parm_sources.py":
                h5parm = Path(tokens[2])
                output_path = h5parm if h5parm.is_absolute() else cwd / h5parm
                output_path.write_text("adjusted")
                return "OK"
            if tokens[0] == "plotrapthor":
                root = next(
                    (token.split("=", 1)[1] for token in tokens if token.startswith("--root=")),
                    f"{tokens[2]}_",
                )
                output_path = cwd / f"{root}solutions.png"
                output_path.write_text("plot")
                return "OK"
            raise AssertionError(f"Unexpected command: {tokens[0]}")

    return FakeCalibrateShellOperation


class NoOutputShellOperation:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def run(self):
        return "OK"


class CalibrateObservationStub:
    channels_are_regular = True


class CalibrateFieldStub:
    def __init__(self, tmp_path):
        self.parset = {
            "dir_working": str(tmp_path / "working"),
            "cluster_specific": {
                "cwl_runner": "toil",
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
        self.ra = 123.0
        self.dec = 45.0
        self.sector_bounds_deg = "[0,0,1,1]"
        self.sector_bounds_mid_deg = "[0.5,0.5]"
        self.smoothnessconstraint_fulljones = 1.5
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
        self.calibrate_bda_timebase = 0
        self.calibrate_bda_frequencybase = 0
        self.do_slowgain_solve = False
        self.calibration_strategy = {"di": ["full_jones"]}
        self.generate_screens = False
        self.use_image_based_predict = False
        self.apply_normalizations = False
        self.normalize_h5parm = None
        self.fulljones_h5parm_filename = None
        self.h5parm_filename = None
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
            "starttime": ["50000.0", "50010.0"],
            "ntimes": [10, 12],
            "bda_maxinterval": [8.0, 9.0],
            "bda_minchannels": [1, 1],
            "solint_fulljones_timestep": [5, 6],
            "solint_fulljones_freqstep": [2, 3],
            "solint_slow_timestep": [11, 12],
            "solint_slow_freqstep": [7, 8],
            "solint_medium_timestep": [9, 10],
            "solint_medium_freqstep": [5, 6],
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
        "fast_phase_solutions": solution,
        "fast_phase_plots": [file_record(pipeline_dir / "phase_solutions.png")],
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
            "combined_solve1_solve2_h5parm": "combined_solve1_solve2_di.h5parm",
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
    return input_parms


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
            "fast_initialsolutions_h5parm": None,
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
    return input_parms


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
            "combined_solve1_solve2_h5parm": "combined_fast_medium1_phases.h5parm",
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
            "medium1_initialsolutions_h5parm": None,
            "do_slowgain_solve": False,
        }
    )
    return input_parms


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
    return input_parms


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
    return input_parms


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
            "combined_solve1_solve2_solve4_h5parm": ("combined_fast_medium1_medium2_phases.h5parm"),
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
            "do_slowgain_solve": True,
            "max_normalization_delta": 0.25,
            "scale_normalization_delta": "False",
            "phase_center_ra": 123.0,
            "phase_center_dec": 45.0,
        }
    )
    return input_parms


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
            "reusemodel": "[solve1.*]",
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
    assert (
        normalized_ddecal_solve_command(
            msin="obs_0.ms",
            data_colname="DATA",
            starttime="50000.0",
            ntimes=10,
            steps="[solve1,solve2]",
            solve_slots=_di_scalar_phase_solve_slots(),
            numthreads=4,
            modeldatacolumn="[MODEL_DATA]",
        )
        == commands["calibrate"]["ddecal_di_scalar_phase"]
    )
    assert (
        normalized_combine_h5parms_command(
            "fast_phases_di.h5parm",
            "medium1_phases_di.h5parm",
            "combined_solve1_solve2_di.h5parm",
            "p1p2_scalar",
            reweight=False,
            calibrator_names=[],
            calibrator_fluxes=[],
        )
        == commands["calibrate"]["combine_fast_medium_di"]
    )
    assert (
        normalized_draw_model_command(
            skymodel="calibration.skymodel",
            numterms=2,
            name="calibration_model",
            ra_dec=["12:00:00.0", "+45.00.00.0"],
            frequency_bandwidth=[150000000.0, 1000000.0],
            cellsize_deg=0.001,
            imsize=[1024, 1024],
            numthreads=4,
        )
        == commands["calibrate"]["draw_model"]
    )
    assert (
        normalized_make_region_file_command(
            skymodel="calibration.skymodel",
            ra_mid=123.0,
            dec_mid=45.0,
            width_ra=2.0,
            width_dec=2.5,
            outfile="field_facets_ds9.reg",
            enclose_names=False,
        )
        == commands["calibrate"]["make_field_region_file"]
    )
    assert (
        normalized_ddecal_solve_command(
            msin="dd_obs_0.ms",
            data_colname="DATA",
            starttime="50000.0",
            ntimes=10,
            steps="[solve1]",
            solve_slots=_dd_fast_phase_solve_slots(),
            numthreads=4,
            timebase=0.0,
            maxinterval=8.0,
            frequencybase=0.0,
            minchannels=1,
            onebeamperpatch=True,
            parallelbaselines=False,
            sagecalpredict=False,
            sourcedb="calibration.skymodel",
            directions=["patch1", "patch2"],
        )
        == commands["calibrate"]["ddecal_dd_fast_phase"]
    )
    assert (
        normalized_ddecal_solve_command(
            msin="dd_obs_0.ms",
            data_colname="DATA",
            starttime="50000.0",
            ntimes=10,
            steps="[solve1,solve2]",
            solve_slots=_dd_fast_medium_solve_slots(),
            numthreads=4,
            timebase=0.0,
            maxinterval=8.0,
            frequencybase=0.0,
            minchannels=1,
            onebeamperpatch=True,
            parallelbaselines=False,
            sagecalpredict=False,
            sourcedb="calibration.skymodel",
            directions=["patch1", "patch2"],
        )
        == commands["calibrate"]["ddecal_dd_fast_medium"]
    )
    assert (
        normalized_ddecal_solve_command(
            msin="dd_obs_0.ms",
            data_colname="DATA",
            starttime="50000.0",
            ntimes=10,
            steps="[applycal,solve1,solve2]",
            solve_slots=_dd_fast_medium_solve_slots(),
            numthreads=4,
            applycal_steps="[fastphase,slowgain,fulljones,normalization]",
            applycal_h5parm="di_solutions.h5",
            fulljones_h5parm="fulljones_solutions.h5",
            normalize_h5parm="normalize_solutions.h5",
            timebase=0.0,
            maxinterval=8.0,
            frequencybase=0.0,
            minchannels=1,
            onebeamperpatch=True,
            parallelbaselines=False,
            sagecalpredict=False,
            sourcedb="calibration.skymodel",
            directions=["patch1", "patch2"],
        )
        == commands["calibrate"]["ddecal_dd_fast_medium_preapply"]
    )
    assert (
        normalized_ddecal_solve_command(
            msin="dd_obs_0.ms",
            data_colname="DATA",
            starttime="50000.0",
            ntimes=10,
            steps="[predict,applybeam,solve1,solve2]",
            solve_slots=_dd_fast_medium_image_predict_solve_slots(),
            numthreads=4,
            timebase=0.0,
            maxinterval=8.0,
            frequencybase=0.0,
            minchannels=1,
            onebeamperpatch=True,
            parallelbaselines=False,
            sagecalpredict=False,
            predict_regions="field_facets_ds9.reg",
            predict_images=[
                "calibration_model-term-0.fits",
                "calibration_model-term-1.fits",
            ],
        )
        == commands["calibrate"]["ddecal_dd_fast_medium_image_predict"]
    )
    assert (
        normalized_ddecal_solve_command(
            msin="dd_obs_0.ms",
            data_colname="DATA",
            starttime="50000.0",
            ntimes=10,
            steps="[predict,applybeam,applycal,solve1,solve2]",
            solve_slots=_dd_fast_medium_image_predict_solve_slots(),
            numthreads=4,
            applycal_steps="[fastphase,normalization]",
            applycal_h5parm="di_solutions.h5",
            normalize_h5parm="normalize_solutions.h5",
            timebase=0.0,
            maxinterval=8.0,
            frequencybase=0.0,
            minchannels=1,
            onebeamperpatch=True,
            parallelbaselines=False,
            sagecalpredict=False,
            predict_regions="field_facets_ds9.reg",
            predict_images=[
                "calibration_model-term-0.fits",
                "calibration_model-term-1.fits",
            ],
        )
        == commands["calibrate"]["ddecal_dd_fast_medium_image_predict_preapply"]
    )
    assert (
        normalized_process_gains_command(
            "slow_gains.h5parm",
            flag=True,
            smooth=True,
            max_station_delta=0.25,
            scale_station_delta="False",
            phase_center_ra=123.0,
            phase_center_dec=45.0,
        )
        == commands["calibrate"]["process_slow_gains"]
    )
    assert (
        normalized_adjust_h5parm_sources_command(
            "calibration.skymodel",
            "combined_solutions.h5",
        )
        == commands["calibrate"]["adjust_h5parm_sources"]
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
    assert build_combine_h5parms_command(
        "fast_phases_di.h5parm",
        "medium1_phases_di.h5parm",
        "combined_solve1_solve2_di.h5parm",
        "p1p2_scalar",
        reweight=False,
        calibrator_names=[],
        calibrator_fluxes=[],
    ) == [
        "combine_h5parms.py",
        "fast_phases_di.h5parm",
        "medium1_phases_di.h5parm",
        "combined_solve1_solve2_di.h5parm",
        "p1p2_scalar",
        "--reweight=False",
        "--cal_names=",
        "--cal_fluxes=",
    ]
    assert build_process_gains_command(
        "slow_gains.h5parm",
        flag=True,
        smooth=True,
        max_station_delta=0.25,
        scale_station_delta="False",
        phase_center_ra=123.0,
        phase_center_dec=45.0,
    ) == [
        "process_gains.py",
        "--normalize=True",
        "slow_gains.h5parm",
        "--smooth=True",
        "--flag=True",
        "--max_station_delta=0.25",
        "--scale_delta_with_dist=False",
        "--phase_center_ra=123.0",
        "--phase_center_dec=45.0",
    ]
    assert build_adjust_h5parm_sources_command(
        "calibration.skymodel",
        "combined_solutions.h5",
    ) == [
        "adjust_h5parm_sources.py",
        "calibration.skymodel",
        "combined_solutions.h5",
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
    assert payload["calibration_kind"] == "di_fulljones"
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
            "solve_slots": [
                {
                    "slot": 1,
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
        },
        {
            "slot": 2,
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
        },
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


def test_calibrate_payload_from_inputs_builds_dd_fast_medium_payload(tmp_path):
    payload = calibrate_payload_from_inputs("dd", _dd_fast_medium_input_parms(), tmp_path)

    assert payload["calibration_kind"] == "dd_phase"
    assert payload["collected_h5parms"]["solve2"] == {
        "filename": "medium1_phases.h5parm",
        "path": str(tmp_path / "medium1_phases.h5parm"),
    }
    assert payload["combined_h5parms"] == {
        "solve1_solve2": {
            "filename": "combined_fast_medium1_phases.h5parm",
            "path": str(tmp_path / "combined_fast_medium1_phases.h5parm"),
        }
    }
    assert payload["chunks"][0]["solve_slots"][1] == {
        "slot": 2,
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


def test_calibrate_payload_from_inputs_builds_dd_with_slow_payload(tmp_path):
    payload = calibrate_payload_from_inputs("dd", _dd_with_slow_input_parms(), tmp_path)

    assert payload["calibration_kind"] == "dd_phase_slow"
    assert payload["do_slowgain_solve"] is True
    assert payload["combined_h5parms"] == {
        "solve1_solve2": {
            "filename": "combined_fast_medium1_phases.h5parm",
            "path": str(tmp_path / "combined_fast_medium1_phases.h5parm"),
        },
        "solve1_solve2_solve4": {
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
            {"output_solve1_h5parm": ["slow_gain_0.h5parm", "slow_gain_1.h5parm"]},
            "Only DD fast/medium phase",
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
            {"solve1_mode": "slow"},
            "Only DI full-Jones and DI scalar phase",
        ),
        (
            "di",
            _di_fulljones_input_parms,
            {"dp3_steps": "[solve1,solve2]"},
            "Only DI full-Jones and DI scalar phase",
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


def test_run_calibrate_flow_supports_di_scalar_phase(tmp_path, fake_calibrate_shell_operation_cls):
    payload = calibrate_payload_from_inputs("di", _di_scalar_phase_input_parms(), tmp_path)

    outputs = run_calibrate_flow(
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
    assert [command[0] for command in commands] == [
        "DP3",
        "DP3",
        "H5parm_collector.py",
        "plotrapthor",
        "H5parm_collector.py",
        "plotrapthor",
        "combine_h5parms.py",
    ]
    assert "steps=[solve1,solve2]" in commands[0]
    assert "solve1.mode=scalarphase" in commands[0]
    assert "solve2.mode=scalarphase" in commands[0]
    assert "solve2.reusemodel=[solve1.*]" in commands[0]
    assert commands[-1][1:5] == [
        str(tmp_path / "fast_phases_di.h5parm"),
        str(tmp_path / "medium1_phases_di.h5parm"),
        "combined_solve1_solve2_di.h5parm",
        "p1p2_scalar",
    ]


def test_run_calibrate_flow_supports_dd_fast_phase(tmp_path, fake_calibrate_shell_operation_cls):
    payload = calibrate_payload_from_inputs("dd", _dd_fast_phase_input_parms(), tmp_path)

    outputs = run_calibrate_flow(
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
    assert [command[0] for command in commands] == [
        "DP3",
        "DP3",
        "H5parm_collector.py",
        "plotrapthor",
    ]
    assert "solve1.mode=scalarphase" in commands[0]
    assert "solve1.h5parm=fast_phase_0.h5parm" in commands[0]
    assert "solve1.sourcedb=/data/calibration.skymodel" in commands[0]
    assert "solve1.directions=[patch1,patch2]" in commands[0]
    assert "solve1.smoothness_dd_factors=[1.0,2.0]" in commands[0]
    assert "avg.maxinterval=8.0" in commands[0]
    assert (
        commands[2][2] == f"{tmp_path / 'fast_phase_0.h5parm'},{tmp_path / 'fast_phase_1.h5parm'}"
    )


def test_run_calibrate_flow_supports_dd_fast_medium(tmp_path, fake_calibrate_shell_operation_cls):
    payload = calibrate_payload_from_inputs("dd", _dd_fast_medium_input_parms(), tmp_path)

    outputs = run_calibrate_flow(
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
    assert [command[0] for command in commands] == [
        "DP3",
        "DP3",
        "H5parm_collector.py",
        "plotrapthor",
        "H5parm_collector.py",
        "plotrapthor",
        "combine_h5parms.py",
        "adjust_h5parm_sources.py",
    ]
    assert "steps=[solve1,solve2]" in commands[0]
    assert "solve2.h5parm=medium1_phase_0.h5parm" in commands[0]
    assert "solve2.reusemodel=[solve1.*]" in commands[0]
    assert commands[-2][1:5] == [
        str(tmp_path / "fast_phases.h5parm"),
        str(tmp_path / "medium1_phases.h5parm"),
        "combined_fast_medium1_phases.h5parm",
        "p1p2_scalar",
    ]
    assert commands[-1] == [
        "adjust_h5parm_sources.py",
        "/data/calibration.skymodel",
        str(tmp_path / "combined_fast_medium1_phases.h5parm"),
    ]


def test_run_calibrate_flow_supports_dd_preapply(tmp_path, fake_calibrate_shell_operation_cls):
    payload = calibrate_payload_from_inputs("dd", _dd_preapply_input_parms(), tmp_path)

    outputs = run_calibrate_flow(
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
    assert [command[0] for command in commands] == [
        "DP3",
        "DP3",
        "H5parm_collector.py",
        "plotrapthor",
        "H5parm_collector.py",
        "plotrapthor",
        "combine_h5parms.py",
        "adjust_h5parm_sources.py",
    ]
    assert "steps=[applycal,solve1,solve2]" in commands[0]
    assert "applycal.steps=[fastphase,slowgain,fulljones,normalization]" in commands[0]
    assert "applycal.parmdb=/solutions/di_solutions.h5" in commands[0]
    assert "applycal.fulljones.parmdb=/solutions/fulljones_solutions.h5" in commands[0]
    assert "applycal.normalization.parmdb=/solutions/normalize_solutions.h5" in commands[0]
    assert "solve2.reusemodel=[solve1.*]" in commands[0]


def test_run_calibrate_flow_supports_dd_image_predict(tmp_path, fake_calibrate_shell_operation_cls):
    payload = calibrate_payload_from_inputs("dd", _dd_image_predict_input_parms(), tmp_path)

    outputs = run_calibrate_flow(
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
    assert [command[0] for command in commands] == [
        "wsclean",
        "make_region_file.py",
        "DP3",
        "DP3",
        "H5parm_collector.py",
        "plotrapthor",
        "H5parm_collector.py",
        "plotrapthor",
        "combine_h5parms.py",
        "adjust_h5parm_sources.py",
    ]
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
    assert commands[1] == [
        "make_region_file.py",
        "/data/calibration.skymodel",
        "123.0",
        "45.0",
        "2.0",
        "2.5",
        "field_facets_ds9.reg",
        "--enclose_names=False",
    ]
    assert "steps=[predict,applybeam,solve1,solve2]" in commands[2]
    assert f"predict.regions={tmp_path / 'field_facets_ds9.reg'}" in commands[2]
    assert (
        f"predict.images=[{tmp_path / 'calibration_model-term-0.fits'},"
        f"{tmp_path / 'calibration_model-term-1.fits'}]" in commands[2]
    )
    assert "solve1.reusemodel=[predict.*]" in commands[2]
    assert "solve2.reusemodel=[predict.*]" in commands[2]
    assert not any(token.startswith("solve1.sourcedb=") for token in commands[2])
    assert not any(token.startswith("solve1.directions=") for token in commands[2])


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
        run_calibrate_flow(
            payload,
            execution_config=ExecutionConfig(task_runner="sync"),
            shell_operation_cls=MissingModelShellOperation,
        )


def test_run_calibrate_flow_fails_when_image_predict_region_is_missing(
    tmp_path, fake_calibrate_shell_operation_cls
):
    class MissingRegionShellOperation(fake_calibrate_shell_operation_cls):
        instances = []

        def run(self):
            tokens = shlex.split(self.kwargs["commands"][0])
            if tokens[0] == "make_region_file.py":
                return "OK"
            return super().run()

    payload = calibrate_payload_from_inputs("dd", _dd_image_predict_input_parms(), tmp_path)

    with pytest.raises(FileNotFoundError, match="Calibration region file was not created"):
        run_calibrate_flow(
            payload,
            execution_config=ExecutionConfig(task_runner="sync"),
            shell_operation_cls=MissingRegionShellOperation,
        )


def test_run_calibrate_flow_supports_dd_image_predict_preapply(
    tmp_path, fake_calibrate_shell_operation_cls
):
    payload = calibrate_payload_from_inputs(
        "dd",
        _dd_image_predict_preapply_input_parms(tmp_path / "normalize_solutions.h5"),
        tmp_path,
    )

    outputs = run_calibrate_flow(
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
    assert [command[0] for command in commands][:4] == [
        "wsclean",
        "make_region_file.py",
        "adjust_h5parm_sources.py",
        "DP3",
    ]
    assert commands[2] == [
        "adjust_h5parm_sources.py",
        "/data/calibration.skymodel",
        str(tmp_path / "normalize_solutions.h5"),
    ]
    assert "steps=[predict,applybeam,applycal,solve1,solve2]" in commands[3]
    assert "applycal.steps=[fastphase,normalization]" in commands[3]
    assert "applycal.parmdb=/solutions/di_solutions.h5" in commands[3]
    assert f"applycal.normalization.parmdb={tmp_path / 'normalize_solutions.h5'}" in commands[3]
    assert "solve1.reusemodel=[predict.*]" in commands[3]
    assert "solve2.reusemodel=[predict.*]" in commands[3]


def test_run_calibrate_flow_skips_dd_source_adjustment_for_single_direction(
    tmp_path, fake_calibrate_shell_operation_cls
):
    input_parms = _dd_fast_medium_input_parms()
    input_parms["calibrator_patch_names"] = ["patch1"]
    input_parms["calibrator_fluxes"] = [10.0]
    input_parms["solve_directions"] = ["patch1"]
    payload = calibrate_payload_from_inputs("dd", input_parms, tmp_path)

    outputs = run_calibrate_flow(
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
    assert "adjust_h5parm_sources.py" not in [command[0] for command in commands]


def test_run_calibrate_flow_supports_dd_with_slow(tmp_path, fake_calibrate_shell_operation_cls):
    payload = calibrate_payload_from_inputs("dd", _dd_with_slow_input_parms(), tmp_path)

    outputs = run_calibrate_flow(
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
    assert [command[0] for command in commands] == [
        "DP3",
        "DP3",
        "H5parm_collector.py",
        "plotrapthor",
        "H5parm_collector.py",
        "plotrapthor",
        "combine_h5parms.py",
        "H5parm_collector.py",
        "process_gains.py",
        "plotrapthor",
        "plotrapthor",
        "H5parm_collector.py",
        "plotrapthor",
        "combine_h5parms.py",
        "combine_h5parms.py",
        "adjust_h5parm_sources.py",
    ]
    assert "steps=[solve1,solve2,solve3,solve4]" in commands[0]
    assert "solve3.initialsolutions.soltab=[phase000,amplitude000]" in commands[0]
    assert "solve3.keepmodel=true" in commands[0]
    assert commands[8][1:3] == ["--normalize=True", str(tmp_path / "slow_gains.h5parm")]
    assert commands[-2][1:5] == [
        str(tmp_path / "combined_fast_medium1_medium2_phases.h5parm"),
        str(tmp_path / "slow_gains.h5parm"),
        "combined_solutions.h5",
        "p1p2a2_scalar",
    ]
    assert commands[-1] == [
        "adjust_h5parm_sources.py",
        "/data/calibration.skymodel",
        str(tmp_path / "combined_solutions.h5"),
    ]


def test_run_calibrate_flow_supports_dd_with_slow_without_medium2(
    tmp_path, fake_calibrate_shell_operation_cls
):
    input_parms = _dd_with_slow_input_parms()
    input_parms["dp3_steps"] = "[solve1,solve2,solve3]"
    payload = calibrate_payload_from_inputs("dd", input_parms, tmp_path)

    outputs = run_calibrate_flow(
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
    assert [command[0] for command in commands] == [
        "DP3",
        "DP3",
        "H5parm_collector.py",
        "plotrapthor",
        "H5parm_collector.py",
        "plotrapthor",
        "combine_h5parms.py",
        "H5parm_collector.py",
        "process_gains.py",
        "plotrapthor",
        "plotrapthor",
        "combine_h5parms.py",
        "adjust_h5parm_sources.py",
    ]
    assert "steps=[solve1,solve2,solve3]" in commands[0]
    assert all("solve4.h5parm" not in token for token in commands[0])
    assert commands[-2][1:5] == [
        str(tmp_path / "combined_fast_medium1_phases.h5parm"),
        str(tmp_path / "slow_gains.h5parm"),
        "combined_solutions.h5",
        "p1a2",
    ]
    assert commands[-1] == [
        "adjust_h5parm_sources.py",
        "/data/calibration.skymodel",
        str(tmp_path / "combined_solutions.h5"),
    ]


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
    assert not Path(operation.pipeline_parset_file).exists()
    assert field.fulljones_h5parm_filename == str(solutions_dir / "fulljones-solutions.h5")
    assert (solutions_dir / "fulljones-solutions.h5").is_file()
    assert (plots_dir / "phase_solutions.png").is_file()
    assert field.scan_h5parms_calls == 1
    assert len(fake_calibrate_shell_operation_cls.instances) == 4


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
    assert (plots_dir / "phase_solutions.png").is_file()
    assert field.scan_h5parms_calls == 1


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
    monkeypatch.setattr(calibrate_module, "_collect_plot_and_combine", fake_collect)

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
    for value in outputs["calibrate_di_scalar_phase"].values():
        validate_output_record(value)
    for value in outputs["calibrate_dd_fast_phase"].values():
        validate_output_record(value)
    for value in outputs["calibrate_dd_fast_medium"].values():
        validate_output_record(value)
    for value in outputs["calibrate_dd_with_slow"].values():
        validate_output_record(value)
