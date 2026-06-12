"""
Test cases for the `rapthor.operations.concatenate` module.
"""

import json
from pathlib import Path

from rapthor.lib.cwl import CWLDir
from rapthor.operations.concatenate import Concatenate


class ObservationStub:
    def __init__(self, ms_filename):
        self.ms_filename = ms_filename


class FieldStub:
    def __init__(self, parset, epoch_observations):
        self.parset = parset
        self.epoch_starttimes = [0, 1]
        self.epoch_observations = epoch_observations
        self.data_colname = "CORRECTED_DATA"
        self.ms_filenames = []
        self.scan_count = 0

    def scan_observations(self):
        self.scan_count += 1


def _operation_parset(tmp_path, batch_system="single_machine"):
    return {
        "dir_working": str(tmp_path / "working"),
        "cluster_specific": {
            "debug_workflow": False,
            "keep_temporary_files": False,
            "batch_system": batch_system,
            "max_cores": 4,
        },
    }


def _field(tmp_path, *, batch_system="single_machine"):
    return FieldStub(
        _operation_parset(tmp_path, batch_system=batch_system),
        [
            [ObservationStub("epoch_0_input_0.ms"), ObservationStub("epoch_0_input_1.ms")],
            [ObservationStub("epoch_1_single.ms")],
        ],
    )


def test_set_parset_parameters_records_runtime_inputs(tmp_path):
    operation = Concatenate(_field(tmp_path), index=1)

    operation.set_parset_parameters()

    assert operation.parset_parms == {
        "rapthor_pipeline_dir": operation.rapthor_pipeline_dir,
        "pipeline_working_dir": operation.pipeline_working_dir,
        "max_cores": 4,
    }


def test_set_parset_parameters_omits_max_cores_for_slurm(tmp_path):
    operation = Concatenate(_field(tmp_path, batch_system="slurm"), index=1)

    operation.set_parset_parameters()

    assert operation.parset_parms["max_cores"] is None


def test_set_input_parameters_builds_multi_epoch_inputs(tmp_path):
    field = _field(tmp_path)
    operation = Concatenate(field, index=1)

    operation.set_input_parameters()

    expected_output = Path(operation.pipeline_working_dir) / "epoch_0_concatenated.ms"
    assert operation.input_parms == {
        "input_filenames": [
            CWLDir(["epoch_0_input_0.ms", "epoch_0_input_1.ms"]).to_json(),
        ],
        "data_colname": "CORRECTED_DATA",
        "output_filenames": ["epoch_0_concatenated.ms"],
    }
    assert operation.final_filenames == [str(expected_output), "epoch_1_single.ms"]


def test_finalize_updates_field_state_and_marks_done(tmp_path):
    field = _field(tmp_path)
    operation = Concatenate(field, index=1)
    operation.set_input_parameters()

    operation.finalize()

    assert field.ms_filenames == [
        str(Path(operation.pipeline_working_dir) / "epoch_0_concatenated.ms"),
        "epoch_1_single.ms",
    ]
    assert field.scan_count == 1
    assert field.data_colname == "DATA"
    assert Path(operation.done_file).is_file()


def test_setup_writes_concatenate_flow_inputs(tmp_path):
    operation = Concatenate(_field(tmp_path), index=1)

    operation.setup()

    assert json.loads(Path(operation.pipeline_inputs_file).read_text()) == operation.input_parms
