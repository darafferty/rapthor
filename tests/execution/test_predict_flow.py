import json
import shlex
from pathlib import Path

import pytest
from prefect.testing.utilities import prefect_test_harness

import rapthor.execution.predict.flow as predict_module
from rapthor.execution.commands import normalize_command
from rapthor.execution.config import ExecutionConfig
from rapthor.execution.predict.commands import (
    build_predict_model_data_command,
)
from rapthor.execution.predict.flow import (
    predict_flow,
    predict_model_data_task,
)
from rapthor.execution.predict.payloads import predict_payload_from_inputs, validate_predict_payload
from rapthor.lib.records import directory_record, file_record, validate_output_record
from rapthor.operations.predict import Predict
from tests.execution.conftest import run_flow_for_test

FIXTURE_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fake_predict_shell_operation_cls():
    class FakePredictShellOperation:
        instances = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.instances.append(self)

        def run(self):
            tokens = shlex.split(self.kwargs["commands"][0])
            cwd = Path(self.kwargs["working_dir"])
            output_paths = self._output_paths(tokens, cwd)
            for output_path in output_paths:
                output_path.mkdir(parents=True, exist_ok=True)
            return "OK"

        @staticmethod
        def _output_paths(tokens, cwd):
            if tokens[0] == "DP3":
                output_name = next(
                    token.split("=", 1)[1] for token in tokens if token.startswith("msout=")
                )
                return [cwd / output_name]
            raise AssertionError(f"Unexpected command: {tokens[0]}")

    return FakePredictShellOperation


def _matching_models(msobs, msmods):
    msobs_basename = Path(msobs).name
    matches = [model for model in msmods if Path(model).name.startswith(msobs_basename)]
    return matches or msmods[:1]


@pytest.fixture(autouse=True)
def fake_direct_predict_helpers(monkeypatch):
    calls = {"add_sector_models": [], "subtract_sector_models": []}

    def fake_add_sector_models(
        msin,
        msmod_list,
        msin_column="DATA",
        model_column="DATA",
        out_column="MODEL_DATA",
        use_compression=False,
        starttime=None,
        quiet=True,
        infix="",
        output_dir=None,
    ):
        calls["add_sector_models"].append(
            {
                "msin": msin,
                "msmod_list": list(msmod_list),
                "msin_column": msin_column,
                "model_column": model_column,
                "out_column": out_column,
                "use_compression": use_compression,
                "starttime": starttime,
                "quiet": quiet,
                "infix": infix,
                "output_dir": output_dir,
            }
        )
        output_root = Path(output_dir or ".")
        for model in _matching_models(msin, list(msmod_list)):
            output = output_root / f"{Path(model).name.removesuffix('_modeldata')}_di.ms"
            output.mkdir(parents=True, exist_ok=True)

    def fake_subtract_sector_models(
        msin,
        model_list,
        msin_column="DATA",
        model_column="DATA",
        out_column="DATA",
        nr_outliers=0,
        nr_bright=0,
        use_compression=False,
        peel_outliers=False,
        peel_bright=False,
        reweight=True,
        starttime=None,
        solint_sec=None,
        solint_hz=None,
        weights_colname="CAL_WEIGHT",
        gainfile="",
        uvcut_min=80.0,
        uvcut_max=1e6,
        phaseonly=True,
        dirname=None,
        quiet=True,
        infix="",
        output_dir=None,
    ):
        calls["subtract_sector_models"].append(
            {
                "msin": msin,
                "model_list": list(model_list),
                "msin_column": msin_column,
                "model_column": model_column,
                "out_column": out_column,
                "nr_outliers": nr_outliers,
                "nr_bright": nr_bright,
                "use_compression": use_compression,
                "peel_outliers": peel_outliers,
                "peel_bright": peel_bright,
                "reweight": reweight,
                "starttime": starttime,
                "solint_sec": solint_sec,
                "solint_hz": solint_hz,
                "weights_colname": weights_colname,
                "gainfile": gainfile,
                "uvcut_min": uvcut_min,
                "uvcut_max": uvcut_max,
                "phaseonly": phaseonly,
                "dirname": dirname,
                "quiet": quiet,
                "infix": infix,
                "output_dir": output_dir,
            }
        )
        output_root = Path(output_dir or ".")
        if (peel_outliers and nr_outliers > 0) or (peel_bright and nr_bright > 0):
            (output_root / f"{Path(msin).name}{infix}_field").mkdir(
                parents=True,
                exist_ok=True,
            )
        for model in _matching_models(msin, list(model_list)):
            output = output_root / Path(model).name.removesuffix("_modeldata")
            output.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(predict_module, "add_sector_models", fake_add_sector_models)
    monkeypatch.setattr(
        predict_module,
        "subtract_sector_models",
        fake_subtract_sector_models,
    )
    return calls


class FailingShellOperation:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def run(self):
        raise RuntimeError("predict failed")


class NoOutputShellOperation:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def run(self):
        return "OK"


class ObservationStub:
    def __init__(self, name, starttime, ms_filename):
        self.name = name
        self.starttime = starttime
        self.ms_filename = ms_filename
        self.infix = ".selfcal"
        self.ms_predict_di_filename = None
        self.parameters = {}
        self.timepersample = 5
        self.channelwidth = 7
        self.numsamples = 10
        self.goesto_endofms = False


class SectorStub:
    def __init__(
        self,
        observations,
        name="sector_1",
        patches=None,
        predict_skymodel_file="sector_1.skymodel",
        is_outlier=False,
    ):
        self.observations = observations
        self.name = name
        self.patches = patches or ["patch1", "patch2"]
        self.predict_skymodel_file = predict_skymodel_file
        self.is_outlier = is_outlier

    def set_prediction_parameters(self):
        for obs in self.observations:
            root_filename = Path(obs.ms_filename).name
            obs.parameters["ms_filename"] = obs.ms_filename
            obs.parameters["ms_model_filename"] = (
                f"{root_filename}{obs.infix}.{self.name}_modeldata"
            )
            obs.ms_subtracted_filename = f"{root_filename}{obs.infix}.{self.name}"
            obs.parameters["ms_subtracted_filename"] = obs.ms_subtracted_filename
            obs.ms_field = f"{root_filename}{obs.infix}_field"
            obs.ms_predict_di = f"{obs.ms_subtracted_filename}_di.ms"
            obs.parameters["patch_names"] = self.patches
            obs.parameters["predict_starttime"] = str(obs.starttime)
            obs.parameters["predict_ntimes"] = obs.numsamples

    def get_obs_parameters(self, name):
        if name == "ms_filename":
            return [obs.ms_filename for obs in self.observations]
        return [obs.parameters[name] for obs in self.observations]


class FieldStub:
    def __init__(self, tmp_path, field_observation, sector_observation):
        self.parset = _operation_parset(tmp_path)
        self.observations = [field_observation]
        self.predict_sectors = [SectorStub([sector_observation])]
        self.sectors = self.predict_sectors
        self.imaging_sectors = []
        self.bright_source_sectors = []
        self.outlier_sectors = []
        self.data_colname = "DATA"
        self.h5parm_filename = None
        self.dd_h5parm_filename = None
        self.di_h5parm_filename = None
        self.normalize_h5parm = None
        self.apply_amplitudes = False
        self.apply_normalizations = False
        self.onebeamperpatch = True
        self.sagecalpredict = False
        self.correct_smearing_in_calibration = False
        self.reweight = False
        self.peel_outliers = False
        self.peel_bright_sources = False
        self.imaged_sources_only = False


def _operation_parset(tmp_path):
    return {
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
            "max_threads": 1,
            "prefect_task_runner": "sync",
        },
        "imaging_specific": {
            "min_uv_lambda": 80.0,
            "max_uv_lambda": 1000000.0,
        },
    }


def _predict_input_parms():
    return {
        "sector_filename": [
            directory_record("/data/obs_0.ms"),
            directory_record("/data/obs_1.ms"),
        ],
        "data_colname": "DATA",
        "sector_starttime": ["50000.0", "50010.0"],
        "sector_ntimes": [10, 12],
        "sector_model_filename": [
            "obs_0.ms.sector_1_modeldata",
            "obs_1.ms.sector_1_modeldata",
        ],
        "sector_skymodel": [
            file_record("/data/sector_1.skymodel"),
            file_record("/data/sector_1.skymodel"),
        ],
        "sector_patches": [["patch1", "patch2"], ["patch1", "patch2"]],
        "h5parm": None,
        "normalize_h5parm": None,
        "dp3_applycal_steps": None,
        "onebeamperpatch": True,
        "sagecalpredict": False,
        "obs_filename": [
            directory_record("/data/obs_0.ms"),
            directory_record("/data/obs_1.ms"),
        ],
        "obs_starttime": ["50000.0", "50010.0"],
        "obs_infix": [".selfcal", ".selfcal"],
        "correctfreqsmearing": False,
        "correcttimesmearing": False,
        "max_threads": 4,
    }


def _single_observation_input_parms():
    input_parms = _predict_input_parms()
    for key in [
        "sector_filename",
        "sector_starttime",
        "sector_ntimes",
        "sector_model_filename",
        "sector_skymodel",
        "sector_patches",
        "obs_filename",
        "obs_starttime",
        "obs_infix",
    ]:
        input_parms[key] = input_parms[key][:1]
    return input_parms


def _dd_predict_input_parms():
    input_parms = _predict_input_parms()
    input_parms.update(
        {
            "obs_solint_sec": [20.0, 30.0],
            "obs_solint_hz": [1000.0, 2000.0],
            "min_uv_lambda": 80.0,
            "max_uv_lambda": 1000000.0,
            "nr_outliers": 1,
            "peel_outliers": True,
            "nr_bright": 0,
            "peel_bright": False,
            "reweight": True,
        }
    )
    return input_parms


def test_predict_command_builders_match_reference_fixtures():
    commands = json.loads((FIXTURE_DIR / "command_reference.json").read_text())

    assert (
        normalize_command(
            build_predict_model_data_command(
                msin="obs_0.ms",
                data_colname="DATA",
                msout="obs_0.ms.sector_1_modeldata",
                starttime="50000.0",
                ntimes=10,
                onebeamperpatch=True,
                correctfreqsmearing=False,
                correcttimesmearing=False,
                sagecalpredict=False,
                sourcedb="sector_1.skymodel",
                directions=["patch1", "patch2"],
                numthreads=4,
            )
        )
        == commands["predict"]["predict_model_data"]
    )


def test_build_predict_model_data_command_adds_h5parm_applycal_options():
    command = build_predict_model_data_command(
        "obs_0.ms",
        "CORRECTED_DATA",
        "obs_0.ms.sector_1_modeldata",
        "50000.0",
        10,
        onebeamperpatch=False,
        correctfreqsmearing=True,
        correcttimesmearing=True,
        sagecalpredict=False,
        sourcedb="sector_1.skymodel",
        directions=["patch1"],
        numthreads=8,
        h5parm="solutions.h5",
        applycal_steps="[fastphase,normalization]",
        normalize_h5parm="norm.h5",
    )

    assert "steps=[predict,applycal]" in command
    assert "predict.applycal.correction=phase000" in command
    assert "predict.type=h5parmpredict" in command
    assert "predict.applycal.steps=[fastphase]" in command
    assert "predict.applycal.parmdb=solutions.h5" in command
    assert "applycal.type=applycal" in command
    assert "applycal.steps=[normalization]" in command
    assert "applycal.normalization.parmdb=norm.h5" in command
    assert "applycal.normalization.usemodeldata=True" in command
    assert "applycal.normalization.invert=False" in command
    assert "predict.correctfreqsmearing=True" in command
    assert "predict.correcttimesmearing=True" in command


def test_build_predict_model_data_command_applies_normalization_to_model_data():
    command = build_predict_model_data_command(
        "obs_0.ms",
        "DATA",
        "obs_0.ms.sector_1_modeldata",
        "50000.0",
        10,
        onebeamperpatch=False,
        correctfreqsmearing=False,
        correcttimesmearing=False,
        sagecalpredict=False,
        sourcedb="sector_1.skymodel",
        directions=["patch1"],
        numthreads=8,
        applycal_steps="[normalization]",
        normalize_h5parm="norm.h5",
    )

    assert "steps=[predict,applycal]" in command
    assert "predict.type=predict" in command
    assert "predict.applycal.steps=[normalization]" not in command
    assert "predict.applycal.normalization.parmdb=norm.h5" not in command
    assert "applycal.steps=[normalization]" in command
    assert "applycal.normalization.parmdb=norm.h5" in command
    assert "applycal.normalization.usemodeldata=True" in command


def test_build_predict_model_data_command_omits_zero_ntimes():
    command = build_predict_model_data_command(
        "obs_0.ms",
        "DATA",
        "obs_0.ms.sector_1_modeldata",
        "50000.0",
        0,
        onebeamperpatch=True,
        correctfreqsmearing=False,
        correcttimesmearing=False,
        sagecalpredict=False,
        sourcedb="sector_1.skymodel",
        directions=["patch1"],
        numthreads=4,
    )

    assert "msin.ntimes=0" not in command
    assert command[command.index("msin.starttime=50000.0") + 1] == "predict.onebeamperpatch=True"


def test_build_predict_model_data_command_uses_sagecalpredict_when_requested():
    command = build_predict_model_data_command(
        "obs_0.ms",
        "DATA",
        "obs_0.ms.sector_1_modeldata",
        "50000.0",
        10,
        onebeamperpatch=True,
        correctfreqsmearing=False,
        correcttimesmearing=False,
        sagecalpredict=True,
        sourcedb="sector_1.skymodel",
        directions=["patch1"],
        numthreads=4,
        h5parm="solutions.h5",
    )

    assert "predict.type=sagecalpredict" in command
    assert "predict.applycal.parmdb=solutions.h5" in command


def test_predict_payload_from_inputs_builds_di_scatter_payload(tmp_path):
    payload = predict_payload_from_inputs("di", _predict_input_parms(), tmp_path)

    assert payload["mode"] == "di"
    assert payload["pipeline_working_dir"] == str(tmp_path)
    assert len(payload["predict_tasks"]) == 2
    assert len(payload["postprocess_tasks"]) == 2
    assert payload["predict_tasks"][0] == {
        "msin": "/data/obs_0.ms",
        "data_colname": "DATA",
        "msout": "obs_0.ms.sector_1_modeldata",
        "msout_path": str(tmp_path / "obs_0.ms.sector_1_modeldata"),
        "starttime": "50000.0",
        "ntimes": 10,
        "onebeamperpatch": True,
        "correctfreqsmearing": False,
        "correcttimesmearing": False,
        "sagecalpredict": False,
        "sourcedb": "/data/sector_1.skymodel",
        "directions": ["patch1", "patch2"],
        "numthreads": 4,
        "h5parm": None,
        "applycal_steps": None,
        "normalize_h5parm": None,
    }
    assert payload["postprocess_tasks"][0] == {
        "msobs": "/data/obs_0.ms",
        "data_colname": "DATA",
        "obs_starttime": "50000.0",
        "infix": ".selfcal",
    }


def test_predict_payload_from_inputs_builds_dd_specific_postprocess_payload(tmp_path):
    payload = predict_payload_from_inputs("dd", _dd_predict_input_parms(), tmp_path)

    assert payload["mode"] == "dd"
    assert payload["postprocess_tasks"][0] == {
        "msobs": "/data/obs_0.ms",
        "data_colname": "DATA",
        "obs_starttime": "50000.0",
        "infix": ".selfcal",
        "solint_sec": 20.0,
        "solint_hz": 1000.0,
        "min_uv_lambda": 80.0,
        "max_uv_lambda": 1000000.0,
        "nr_outliers": 1,
        "peel_outliers": True,
        "nr_bright": 0,
        "peel_bright": False,
        "reweight": True,
    }


def test_validate_predict_payload_normalizes_unset_optional_strings(tmp_path):
    payload = predict_payload_from_inputs("di", _single_observation_input_parms(), tmp_path)
    payload["predict_tasks"][0]["h5parm"] = "None"
    payload["predict_tasks"][0]["applycal_steps"] = ""
    payload["predict_tasks"][0]["normalize_h5parm"] = None

    validated = validate_predict_payload(payload)

    assert validated["predict_tasks"][0]["h5parm"] is None
    assert validated["predict_tasks"][0]["applycal_steps"] is None
    assert validated["predict_tasks"][0]["normalize_h5parm"] is None


def test_predict_payload_from_inputs_rejects_mismatched_scatter_inputs(tmp_path):
    input_parms = _predict_input_parms()
    input_parms["sector_ntimes"] = [10]

    with pytest.raises(ValueError, match="same length"):
        predict_payload_from_inputs("di", input_parms, tmp_path)


def test_predict_payload_from_inputs_rejects_non_basename_model_outputs(tmp_path):
    input_parms = _predict_input_parms()
    input_parms["sector_model_filename"][0] = "nested/modeldata.ms"

    with pytest.raises(ValueError, match="basename"):
        predict_payload_from_inputs("di", input_parms, tmp_path)


def test_run_predict_flow_executes_di_commands_and_returns_nested_records(
    tmp_path, fake_predict_shell_operation_cls, fake_direct_predict_helpers
):
    payload = predict_payload_from_inputs("di", _predict_input_parms(), tmp_path)

    outputs = run_flow_for_test(
        predict_flow,
        payload,
        execution_config=ExecutionConfig(task_runner="sync"),
        shell_operation_cls=fake_predict_shell_operation_cls,
    )

    assert outputs == {
        "msfiles_di_cal": [
            [directory_record(tmp_path / "obs_0.ms.sector_1_di.ms")],
            [directory_record(tmp_path / "obs_1.ms.sector_1_di.ms")],
        ]
    }
    validate_output_record(outputs["msfiles_di_cal"])
    command_tokens = [
        shlex.split(instance.kwargs["commands"][0])
        for instance in fake_predict_shell_operation_cls.instances
    ]
    assert [tokens[0] for tokens in command_tokens] == [
        "DP3",
        "DP3",
    ]
    assert fake_direct_predict_helpers["add_sector_models"] == [
        {
            "msin": "/data/obs_0.ms",
            "msmod_list": [
                str(tmp_path / "obs_0.ms.sector_1_modeldata"),
                str(tmp_path / "obs_1.ms.sector_1_modeldata"),
            ],
            "msin_column": "DATA",
            "model_column": "DATA",
            "out_column": "MODEL_DATA",
            "use_compression": False,
            "starttime": "50000.0",
            "quiet": True,
            "infix": ".selfcal",
            "output_dir": str(tmp_path),
        },
        {
            "msin": "/data/obs_1.ms",
            "msmod_list": [
                str(tmp_path / "obs_0.ms.sector_1_modeldata"),
                str(tmp_path / "obs_1.ms.sector_1_modeldata"),
            ],
            "msin_column": "DATA",
            "model_column": "DATA",
            "out_column": "MODEL_DATA",
            "use_compression": False,
            "starttime": "50010.0",
            "quiet": True,
            "infix": ".selfcal",
            "output_dir": str(tmp_path),
        },
    ]


def test_run_predict_flow_rejects_invalid_prediction_directions(
    tmp_path, fake_predict_shell_operation_cls
):
    payload = predict_payload_from_inputs("di", _single_observation_input_parms(), tmp_path)
    payload["predict_tasks"][0]["directions"] = ["patch1", 7]

    with pytest.raises(ValueError, match="directions"):
        run_flow_for_test(
            predict_flow,
            payload,
            execution_config=ExecutionConfig(task_runner="sync"),
            shell_operation_cls=fake_predict_shell_operation_cls,
        )

    assert fake_predict_shell_operation_cls.instances == []


def test_run_predict_flow_executes_dd_commands_and_returns_peeling_records(
    tmp_path, fake_predict_shell_operation_cls, fake_direct_predict_helpers
):
    payload = predict_payload_from_inputs("dd", _dd_predict_input_parms(), tmp_path)

    outputs = run_flow_for_test(
        predict_flow,
        payload,
        execution_config=ExecutionConfig(task_runner="sync"),
        shell_operation_cls=fake_predict_shell_operation_cls,
    )

    assert outputs == {
        "subtract_models": [
            [
                directory_record(tmp_path / "obs_0.ms.selfcal_field"),
                directory_record(tmp_path / "obs_0.ms.sector_1"),
            ],
            [
                directory_record(tmp_path / "obs_1.ms.selfcal_field"),
                directory_record(tmp_path / "obs_1.ms.sector_1"),
            ],
        ]
    }
    validate_output_record(outputs["subtract_models"])
    command_tokens = [
        shlex.split(instance.kwargs["commands"][0])
        for instance in fake_predict_shell_operation_cls.instances
    ]
    assert [tokens[0] for tokens in command_tokens] == ["DP3", "DP3"]
    assert fake_direct_predict_helpers["subtract_sector_models"][-1]["peel_outliers"] is True
    assert fake_direct_predict_helpers["subtract_sector_models"][-1]["reweight"] is True
    assert (
        fake_direct_predict_helpers["subtract_sector_models"][-1]["weights_colname"]
        == "WEIGHT_SPECTRUM"
    )


def test_predict_model_data_task_wraps_runner(tmp_path, fake_predict_shell_operation_cls):
    payload = predict_payload_from_inputs("di", _single_observation_input_parms(), tmp_path)

    task_fn = getattr(predict_model_data_task, "fn", predict_model_data_task)
    output = task_fn(
        payload["predict_tasks"][0],
        str(tmp_path),
        execution_config=ExecutionConfig(task_runner="sync"),
        shell_operation_cls=fake_predict_shell_operation_cls,
    )

    assert output == directory_record(tmp_path / "obs_0.ms.sector_1_modeldata")
    assert fake_predict_shell_operation_cls.instances[0].kwargs["working_dir"] == str(tmp_path)


def test_predict_prefect_flow_entrypoint_runs_with_mocked_shell(
    tmp_path, monkeypatch, fake_predict_shell_operation_cls
):
    monkeypatch.setattr(
        "rapthor.execution.shell._load_shell_operation_cls",
        lambda: fake_predict_shell_operation_cls,
    )
    payload = predict_payload_from_inputs("di", _single_observation_input_parms(), tmp_path)

    with prefect_test_harness(server_startup_timeout=None):
        outputs = predict_flow(
            payload,
            execution_config=ExecutionConfig(task_runner="sync"),
        )

    assert outputs == {"msfiles_di_cal": [[directory_record(tmp_path / "obs_0.ms.sector_1_di.ms")]]}
    assert len(fake_predict_shell_operation_cls.instances) == 1


def test_run_predict_flow_propagates_shell_failures(tmp_path):
    payload = predict_payload_from_inputs("di", _single_observation_input_parms(), tmp_path)

    with pytest.raises(RuntimeError, match="predict failed"):
        run_flow_for_test(
            predict_flow,
            payload,
            execution_config=ExecutionConfig(task_runner="sync"),
            shell_operation_cls=FailingShellOperation,
        )


def test_run_predict_flow_fails_when_predicted_model_is_missing(tmp_path):
    payload = predict_payload_from_inputs("di", _single_observation_input_parms(), tmp_path)

    with pytest.raises(FileNotFoundError, match="Predict output was not created"):
        run_flow_for_test(
            predict_flow,
            payload,
            execution_config=ExecutionConfig(task_runner="sync"),
            shell_operation_cls=NoOutputShellOperation,
        )


def test_run_predict_flow_fails_when_postprocess_output_is_missing(
    tmp_path, monkeypatch, fake_predict_shell_operation_cls
):
    def skip_add_sector_models(*args, **kwargs):
        return None

    monkeypatch.setattr(predict_module, "add_sector_models", skip_add_sector_models)
    payload = predict_payload_from_inputs("di", _single_observation_input_parms(), tmp_path)

    with pytest.raises(FileNotFoundError, match="post-processing outputs"):
        run_flow_for_test(
            predict_flow,
            payload,
            execution_config=ExecutionConfig(task_runner="sync"),
            shell_operation_cls=fake_predict_shell_operation_cls,
        )


def test_predict_reference_output_fixtures_match_output_contract():
    outputs = json.loads((FIXTURE_DIR / "output_reference.json").read_text())

    validate_output_record(outputs["predict_di"]["msfiles_di_cal"])
    validate_output_record(outputs["predict"]["subtract_models"])


def test_predict_finalizer_accepts_prefect_outputs(tmp_path, fake_predict_shell_operation_cls):
    field_observation = ObservationStub("obs_0", 59000.0, "obs_0.ms")
    sector_observation = ObservationStub("obs_0", 59000.0, "obs_0.ms")
    field = FieldStub(tmp_path, field_observation, sector_observation)
    operation = Predict("di", field, index=1)
    payload = predict_payload_from_inputs(
        "di",
        _single_observation_input_parms(),
        operation.pipeline_working_dir,
    )
    outputs = run_flow_for_test(
        predict_flow,
        payload,
        execution_config=ExecutionConfig(task_runner="sync"),
        shell_operation_cls=fake_predict_shell_operation_cls,
    )
    sector_observation.ms_predict_di = Path(outputs["msfiles_di_cal"][0][0]["path"]).name

    operation.outputs = outputs
    operation.finalize()

    assert field_observation.ms_predict_di_filename == str(
        Path(operation.pipeline_working_dir) / "obs_0.ms.sector_1_di.ms"
    )
    assert field_observation.infix == ".selfcal"
    assert Path(operation.done_file).is_file()


def test_predict_di_finalize_preserves_chunk_infixes_for_later_predict(tmp_path):
    field_observations = [
        ObservationStub("obs_0", 59000.0, "obs_0.ms"),
        ObservationStub("obs_0", 59010.0, "obs_0.ms"),
    ]
    field_observations[0].infix = ".mjd59000"
    field_observations[1].infix = ".mjd59010"
    sector_observations = [
        ObservationStub("obs_0", 59000.0, "obs_0.ms"),
        ObservationStub("obs_0", 59010.0, "obs_0.ms"),
    ]
    sector_observations[0].infix = ".mjd59000"
    sector_observations[1].infix = ".mjd59010"
    field = FieldStub(tmp_path, field_observations[0], sector_observations[0])
    field.observations = field_observations
    field.predict_sectors = [SectorStub(sector_observations, name="predict_1")]
    field.sectors = field.predict_sectors

    first_predict = Predict("di", field, index=1)
    field.predict_sectors[0].set_prediction_parameters()
    first_predict.finalize()

    assert [obs.infix for obs in field.observations] == [".mjd59000", ".mjd59010"]

    next_predict = Predict("di", field, index=3)
    next_predict.set_input_parameters()

    assert next_predict.input_parms["sector_model_filename"] == [
        "obs_0.ms.mjd59000.predict_1_modeldata",
        "obs_0.ms.mjd59010.predict_1_modeldata",
    ]


def test_predict_di_operation_run_uses_prefect_flow(
    tmp_path, monkeypatch, fake_predict_shell_operation_cls
):
    monkeypatch.setattr(
        "rapthor.execution.shell._load_shell_operation_cls",
        lambda: fake_predict_shell_operation_cls,
    )
    field_observation = ObservationStub("obs_0", 59000.0, "obs_0.ms")
    sector_observation = ObservationStub("obs_0", 59000.0, "obs_0.ms")
    field = FieldStub(tmp_path, field_observation, sector_observation)
    operation = Predict("di", field, index=1)

    with prefect_test_harness(server_startup_timeout=None):
        operation.run()

    expected_outputs = {
        "msfiles_di_cal": [
            [
                directory_record(
                    Path(operation.pipeline_working_dir) / "obs_0.ms.selfcal.sector_1_di.ms"
                )
            ]
        ]
    }
    assert operation.outputs == expected_outputs
    assert json.loads(Path(operation.outputs_file).read_text()) == expected_outputs
    assert Path(operation.done_file).is_file()
    assert Path(operation.pipeline_inputs_file).is_file()
    assert field_observation.ms_predict_di_filename == str(
        Path(operation.pipeline_working_dir) / "obs_0.ms.selfcal.sector_1_di.ms"
    )
    assert field_observation.infix == ".selfcal"
    assert len(fake_predict_shell_operation_cls.instances) == 1


def test_predict_di_operation_run_reuses_prefect_outputs_when_done(
    tmp_path, monkeypatch, fake_predict_shell_operation_cls
):
    monkeypatch.setattr(
        "rapthor.execution.shell._load_shell_operation_cls",
        lambda: fake_predict_shell_operation_cls,
    )
    field_observation = ObservationStub("obs_0", 59000.0, "obs_0.ms")
    sector_observation = ObservationStub("obs_0", 59000.0, "obs_0.ms")
    field = FieldStub(tmp_path, field_observation, sector_observation)
    operation = Predict("di", field, index=1)
    expected_outputs = {
        "msfiles_di_cal": [
            [
                directory_record(
                    Path(operation.pipeline_working_dir) / "obs_0.ms.selfcal.sector_1_di.ms"
                )
            ]
        ]
    }
    Path(operation.done_file).touch()
    Path(operation.outputs_file).write_text(json.dumps(expected_outputs))

    operation.run()

    assert operation.outputs == expected_outputs
    assert Path(operation.done_file).is_file()
    assert field_observation.ms_predict_di_filename == str(
        Path(operation.pipeline_working_dir) / "obs_0.ms.selfcal.sector_1_di.ms"
    )
    assert field_observation.infix == ".selfcal"
    assert fake_predict_shell_operation_cls.instances == []


@pytest.mark.parametrize(
    "shell_operation_cls, expected_message",
    [
        pytest.param(FailingShellOperation, "predict failed", id="shell-failure"),
        pytest.param(
            NoOutputShellOperation,
            "Predict output was not created",
            id="missing-predicted-output",
        ),
    ],
)
def test_predict_di_operation_run_failure_does_not_mark_done(
    tmp_path, monkeypatch, shell_operation_cls, expected_message
):
    monkeypatch.setattr(
        "rapthor.execution.shell._load_shell_operation_cls",
        lambda: shell_operation_cls,
    )
    field_observation = ObservationStub("obs_0", 59000.0, "obs_0.ms")
    sector_observation = ObservationStub("obs_0", 59000.0, "obs_0.ms")
    field = FieldStub(tmp_path, field_observation, sector_observation)
    operation = Predict("di", field, index=1)

    with (
        prefect_test_harness(server_startup_timeout=None),
        pytest.raises((FileNotFoundError, RuntimeError), match=expected_message),
    ):
        operation.run()

    assert Path(operation.pipeline_inputs_file).is_file()
    assert not Path(operation.done_file).exists()
    assert not Path(operation.outputs_file).exists()
    assert operation.outputs == {}
    assert field_observation.ms_predict_di_filename is None
    assert field_observation.infix == ".selfcal"


def test_predict_dd_operation_run_uses_prefect_flow(
    tmp_path, monkeypatch, fake_predict_shell_operation_cls
):
    monkeypatch.setattr(
        "rapthor.execution.shell._load_shell_operation_cls",
        lambda: fake_predict_shell_operation_cls,
    )
    field_observation = ObservationStub("obs_0", 59000.0, "obs_0.ms")
    sector_1_observation = ObservationStub("obs_0", 59000.0, "obs_0.ms")
    sector_2_observation = ObservationStub("obs_0", 59000.0, "obs_0.ms")
    field = FieldStub(tmp_path, field_observation, sector_1_observation)
    field.imaging_sectors = [
        SectorStub(
            [sector_1_observation],
            name="sector_1",
            patches=["patch1"],
            predict_skymodel_file="sector_1.skymodel",
        ),
        SectorStub(
            [sector_2_observation],
            name="sector_2",
            patches=["patch2"],
            predict_skymodel_file="sector_2.skymodel",
        ),
    ]
    field.sectors = field.imaging_sectors
    field.predict_sectors = []
    operation = Predict("dd", field, index=1)

    with prefect_test_harness(server_startup_timeout=None):
        operation.run()

    expected_outputs = {
        "subtract_models": [
            [
                directory_record(
                    Path(operation.pipeline_working_dir) / "obs_0.ms.selfcal.sector_1"
                ),
                directory_record(
                    Path(operation.pipeline_working_dir) / "obs_0.ms.selfcal.sector_2"
                ),
            ]
        ]
    }
    assert operation.outputs == expected_outputs
    assert json.loads(Path(operation.outputs_file).read_text()) == expected_outputs
    assert Path(operation.done_file).is_file()
    assert Path(operation.pipeline_inputs_file).is_file()
    assert field.data_colname == "DATA"
    assert sector_1_observation.ms_imaging_filename == str(
        Path(operation.pipeline_working_dir) / "obs_0.ms.selfcal.sector_1"
    )
    assert sector_2_observation.ms_imaging_filename == str(
        Path(operation.pipeline_working_dir) / "obs_0.ms.selfcal.sector_2"
    )
    assert len(fake_predict_shell_operation_cls.instances) == 2
