import json
import shlex
from pathlib import Path

import pytest
from prefect.testing.utilities import prefect_test_harness

from rapthor.execution.config import ExecutionConfig
from rapthor.execution.flows.mosaic import (
    build_compress_mosaic_command,
    build_make_mosaic_command,
    build_make_mosaic_template_command,
    build_regrid_image_command,
    mosaic_flow,
    mosaic_payload_from_inputs,
    normalized_make_mosaic_command,
    normalized_make_mosaic_template_command,
    normalized_regrid_image_command,
    run_mosaic_flow,
)
from rapthor.execution.outputs import file_record, validate_output_record
from rapthor.operations.mosaic import Mosaic

FIXTURE_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fake_mosaic_shell_operation_cls():
    class FakeMosaicShellOperation:
        instances = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.instances.append(self)

        def run(self):
            tokens = shlex.split(self.kwargs["commands"][0])
            cwd = Path(self.kwargs["working_dir"])
            if tokens[0] == "make_mosaic_template.py":
                output_path = cwd / tokens[3]
            elif tokens[0] == "regrid_image.py":
                output_path = cwd / tokens[4]
            elif tokens[0] == "make_mosaic.py":
                output_path = cwd / tokens[3]
            elif tokens[0] == "fpack":
                output_path = cwd / f"{tokens[1]}.fz"
            else:
                raise AssertionError(f"Unexpected command: {tokens[0]}")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text("OK")
            return "OK"

    return FakeMosaicShellOperation


class NoOutputShellOperation:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def run(self):
        return "OK"


class FailingShellOperation:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def run(self):
        raise RuntimeError("mosaic failed")


class SectorStub:
    def __init__(self, root: Path, index: int):
        root.mkdir(parents=True, exist_ok=True)
        prefix = root / f"sector_{index}"
        self.vertices_file = str(prefix.with_suffix(".vertices"))
        self.I_image_file_true_sky = str(root / f"sector_{index}-I-image.fits")
        self.I_image_file_apparent_sky = str(root / f"sector_{index}-I-apparent.fits")
        Path(self.vertices_file).write_text("vertices")
        Path(self.I_image_file_true_sky).write_text("true")
        Path(self.I_image_file_apparent_sky).write_text("apparent")


class FieldStub:
    def __init__(self, tmp_path, sectors):
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
                "prefect_task_runner": "sync",
            },
            "imaging_specific": {"save_filtered_model_image": False},
        }
        self.imaging_sectors = sectors
        self.compress_images = False
        self.image_pol = ["I"]
        self.disable_clean = True
        self.save_supplementary_images = False
        self.field_image_filename = None
        self.field_image_filename_prev = None


def _mosaic_payload(tmp_path, compress_images=False):
    return {
        "pipeline_working_dir": str(tmp_path),
        "compress_images": compress_images,
        "skip_processing": False,
        "image_types": [
            {
                "sector_image_filenames": ["sector_1-I-image.fits", "sector_2-I-image.fits"],
                "sector_vertices_filenames": ["sector_1.vertices", "sector_2.vertices"],
                "template_image_filename": "mosaic_1_template.fits",
                "template_image_path": str(tmp_path / "mosaic_1_template.fits"),
                "regridded_image_filenames": [
                    "sector_1-I-image.fits.regridded",
                    "sector_2-I-image.fits.regridded",
                ],
                "mosaic_filename": "mosaic_1-I-image.fits",
                "mosaic_path": str(tmp_path / "mosaic_1-I-image.fits"),
            }
        ],
    }


def test_mosaic_command_builders_match_reference_fixtures():
    commands = json.loads((FIXTURE_DIR / "cwl_reference_commands.json").read_text())

    assert (
        normalized_make_mosaic_template_command(
            ["sector_1-I-image.fits", "sector_2-I-image.fits"],
            ["sector_1.vertices", "sector_2.vertices"],
            "mosaic_1_template.fits",
        )
        == commands["mosaic"]["make_mosaic_template"]
    )
    assert (
        normalized_regrid_image_command(
            "sector_1-I-image.fits",
            "mosaic_1_template.fits",
            "sector_1.vertices",
            "sector_1-I-image.fits.regridded",
        )
        == commands["mosaic"]["regrid_image"]
    )
    assert (
        normalized_make_mosaic_command(
            ["sector_1-I-image.fits.regridded", "sector_2-I-image.fits.regridded"],
            "mosaic_1_template.fits",
            "mosaic_1-I-image.fits",
        )
        == commands["mosaic"]["make_mosaic"]
    )
    assert (
        build_compress_mosaic_command("mosaic_1-I-image.fits")
        == commands["mosaic"]["compress_mosaic_image"]
    )


def test_mosaic_command_builders_create_cwl_equivalent_tokens():
    assert build_make_mosaic_template_command(
        ["sector_1-I-image.fits", "sector_2-I-image.fits"],
        ["sector_1.vertices", "sector_2.vertices"],
        "mosaic_1_template.fits",
    ) == [
        "make_mosaic_template.py",
        "sector_1-I-image.fits,sector_2-I-image.fits",
        "sector_1.vertices,sector_2.vertices",
        "mosaic_1_template.fits",
        "--skip=False",
    ]
    assert build_regrid_image_command(
        "sector_1-I-image.fits",
        "mosaic_1_template.fits",
        "sector_1.vertices",
        "sector_1-I-image.fits.regridded",
    ) == [
        "regrid_image.py",
        "sector_1-I-image.fits",
        "mosaic_1_template.fits",
        "sector_1.vertices",
        "sector_1-I-image.fits.regridded",
        "--skip=False",
    ]
    assert build_make_mosaic_command(
        ["sector_1-I-image.fits.regridded", "sector_2-I-image.fits.regridded"],
        "mosaic_1_template.fits",
        "mosaic_1-I-image.fits",
    ) == [
        "make_mosaic.py",
        "sector_1-I-image.fits.regridded,sector_2-I-image.fits.regridded",
        "mosaic_1_template.fits",
        "mosaic_1-I-image.fits",
        "--skip=False",
    ]


def test_mosaic_payload_from_inputs_is_serializable(tmp_path):
    payload = mosaic_payload_from_inputs(
        {
            "skip_processing": False,
            "sector_image_filename": [
                [
                    file_record("/data/sector_1-I-image.fits"),
                    file_record("/data/sector_2-I-image.fits"),
                ]
            ],
            "sector_vertices_filename": [
                [file_record("/data/sector_1.vertices"), file_record("/data/sector_2.vertices")]
            ],
            "template_image_filename": ["mosaic_1_template.fits"],
            "regridded_image_filename": [
                ["sector_1-I-image.fits.regridded", "sector_2-I-image.fits.regridded"]
            ],
            "mosaic_filename": ["mosaic_1-I-image.fits"],
        },
        tmp_path,
        compress_images=True,
    )

    assert payload == {
        "pipeline_working_dir": str(tmp_path),
        "compress_images": True,
        "skip_processing": False,
        "image_types": [
            {
                "sector_image_filenames": [
                    "/data/sector_1-I-image.fits",
                    "/data/sector_2-I-image.fits",
                ],
                "sector_vertices_filenames": [
                    "/data/sector_1.vertices",
                    "/data/sector_2.vertices",
                ],
                "template_image_filename": "mosaic_1_template.fits",
                "template_image_path": str(tmp_path / "mosaic_1_template.fits"),
                "regridded_image_filenames": [
                    "sector_1-I-image.fits.regridded",
                    "sector_2-I-image.fits.regridded",
                ],
                "mosaic_filename": "mosaic_1-I-image.fits",
                "mosaic_path": str(tmp_path / "mosaic_1-I-image.fits"),
            }
        ],
    }


def test_mosaic_payload_handles_skip_processing(tmp_path):
    payload = mosaic_payload_from_inputs(
        {"skip_processing": True},
        tmp_path,
        compress_images=False,
    )

    assert payload == {
        "pipeline_working_dir": str(tmp_path),
        "compress_images": False,
        "skip_processing": True,
        "image_types": [],
    }


def test_run_mosaic_flow_executes_commands_and_returns_records(
    tmp_path, fake_mosaic_shell_operation_cls
):
    outputs = run_mosaic_flow(
        _mosaic_payload(tmp_path),
        execution_config=ExecutionConfig(task_runner="sync"),
        shell_operation_cls=fake_mosaic_shell_operation_cls,
    )

    assert outputs == {"mosaic_image": [file_record(tmp_path / "mosaic_1-I-image.fits")]}
    validate_output_record(outputs["mosaic_image"])
    assert [
        instance.kwargs["commands"][0] for instance in fake_mosaic_shell_operation_cls.instances
    ] == [
        "make_mosaic_template.py sector_1-I-image.fits,sector_2-I-image.fits "
        "sector_1.vertices,sector_2.vertices mosaic_1_template.fits --skip=False",
        "regrid_image.py sector_1-I-image.fits mosaic_1_template.fits sector_1.vertices "
        "sector_1-I-image.fits.regridded --skip=False",
        "regrid_image.py sector_2-I-image.fits mosaic_1_template.fits sector_2.vertices "
        "sector_2-I-image.fits.regridded --skip=False",
        "make_mosaic.py sector_1-I-image.fits.regridded,sector_2-I-image.fits.regridded "
        "mosaic_1_template.fits mosaic_1-I-image.fits --skip=False",
    ]


def test_run_mosaic_flow_returns_compressed_records(tmp_path, fake_mosaic_shell_operation_cls):
    outputs = run_mosaic_flow(
        _mosaic_payload(tmp_path, compress_images=True),
        execution_config=ExecutionConfig(task_runner="sync"),
        shell_operation_cls=fake_mosaic_shell_operation_cls,
    )

    assert outputs == {"mosaic_image": [file_record(tmp_path / "mosaic_1-I-image.fits.fz")]}
    assert fake_mosaic_shell_operation_cls.instances[-1].kwargs["commands"] == [
        "fpack mosaic_1-I-image.fits"
    ]


def test_run_mosaic_flow_handles_skip_processing_without_commands(
    tmp_path, fake_mosaic_shell_operation_cls
):
    outputs = run_mosaic_flow(
        {
            "pipeline_working_dir": str(tmp_path),
            "compress_images": False,
            "skip_processing": True,
            "image_types": [],
        },
        execution_config=ExecutionConfig(task_runner="sync"),
        shell_operation_cls=fake_mosaic_shell_operation_cls,
    )

    assert outputs == {}
    assert fake_mosaic_shell_operation_cls.instances == []


def test_mosaic_prefect_flow_entrypoint_runs_with_mocked_shell(
    tmp_path, monkeypatch, fake_mosaic_shell_operation_cls
):
    monkeypatch.setattr(
        "rapthor.execution.shell._load_shell_operation_cls",
        lambda: fake_mosaic_shell_operation_cls,
    )

    with prefect_test_harness(server_startup_timeout=None):
        outputs = mosaic_flow(
            _mosaic_payload(tmp_path),
            execution_config=ExecutionConfig(task_runner="sync"),
        )

    assert outputs == {"mosaic_image": [file_record(tmp_path / "mosaic_1-I-image.fits")]}
    assert len(fake_mosaic_shell_operation_cls.instances) == 4


def test_run_mosaic_flow_fails_when_expected_output_is_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="Mosaic output was not created"):
        run_mosaic_flow(
            _mosaic_payload(tmp_path),
            execution_config=ExecutionConfig(task_runner="sync"),
            shell_operation_cls=NoOutputShellOperation,
        )


def test_mosaic_finalizer_accepts_prefect_outputs(tmp_path, fake_mosaic_shell_operation_cls):
    field = FieldStub(
        tmp_path,
        [SectorStub(tmp_path / "inputs", 1), SectorStub(tmp_path / "inputs", 2)],
    )
    operation = Mosaic(field, index=1)
    operation.set_input_parameters()
    payload = mosaic_payload_from_inputs(
        operation.input_parms,
        operation.pipeline_working_dir,
        compress_images=field.compress_images,
    )
    outputs = run_mosaic_flow(
        payload,
        execution_config=ExecutionConfig(task_runner="sync"),
        shell_operation_cls=fake_mosaic_shell_operation_cls,
    )

    operation.outputs = outputs
    operation.finalize()

    expected_field_image = (
        Path(field.parset["dir_working"]) / "images" / "image_1" / "field-I-image.fits"
    )
    assert field.field_image_filename == str(expected_field_image)
    assert field.field_image_filename_prev is None
    assert expected_field_image.is_file()
    assert Path(operation.done_file).is_file()


def test_mosaic_operation_run_uses_prefect_flow(
    tmp_path, monkeypatch, fake_mosaic_shell_operation_cls
):
    monkeypatch.setattr(
        "rapthor.execution.shell._load_shell_operation_cls",
        lambda: fake_mosaic_shell_operation_cls,
    )
    field = FieldStub(
        tmp_path,
        [SectorStub(tmp_path / "inputs", 1), SectorStub(tmp_path / "inputs", 2)],
    )
    operation = Mosaic(field, index=1)

    with prefect_test_harness(server_startup_timeout=None):
        operation.run()

    expected_outputs = {
        "mosaic_image": [
            file_record(Path(operation.pipeline_working_dir) / "mosaic_1-I-image.fits"),
            file_record(Path(operation.pipeline_working_dir) / "mosaic_1-I-apparent.fits"),
        ]
    }
    expected_field_image = (
        Path(field.parset["dir_working"]) / "images" / "image_1" / "field-I-image.fits"
    )
    assert operation.outputs == expected_outputs
    assert json.loads(Path(operation.outputs_file).read_text()) == expected_outputs
    assert Path(operation.done_file).is_file()
    assert Path(operation.pipeline_inputs_file).is_file()
    assert not Path(operation.pipeline_parset_file).exists()
    assert field.field_image_filename == str(expected_field_image)
    assert field.field_image_filename_prev is None
    assert expected_field_image.is_file()
    assert len(fake_mosaic_shell_operation_cls.instances) == 8


def test_mosaic_operation_run_reuses_prefect_outputs_when_done(
    tmp_path, monkeypatch, fake_mosaic_shell_operation_cls
):
    monkeypatch.setattr(
        "rapthor.execution.shell._load_shell_operation_cls",
        lambda: fake_mosaic_shell_operation_cls,
    )
    field = FieldStub(
        tmp_path,
        [SectorStub(tmp_path / "inputs", 1), SectorStub(tmp_path / "inputs", 2)],
    )
    operation = Mosaic(field, index=1)
    expected_outputs = {
        "mosaic_image": [
            file_record(Path(operation.pipeline_working_dir) / "mosaic_1-I-image.fits"),
            file_record(Path(operation.pipeline_working_dir) / "mosaic_1-I-apparent.fits"),
        ]
    }
    for output in expected_outputs["mosaic_image"]:
        Path(output["path"]).write_text("mosaic")
    Path(operation.done_file).touch()
    Path(operation.outputs_file).write_text(json.dumps(expected_outputs))

    operation.run()

    expected_field_image = (
        Path(field.parset["dir_working"]) / "images" / "image_1" / "field-I-image.fits"
    )
    assert operation.outputs == expected_outputs
    assert Path(operation.done_file).is_file()
    assert field.field_image_filename == str(expected_field_image)
    assert expected_field_image.is_file()
    assert fake_mosaic_shell_operation_cls.instances == []


@pytest.mark.parametrize(
    "shell_operation_cls, expected_message",
    [
        pytest.param(FailingShellOperation, "mosaic failed", id="shell-failure"),
        pytest.param(
            NoOutputShellOperation,
            "Mosaic output was not created",
            id="missing-mosaic-output",
        ),
    ],
)
def test_mosaic_operation_run_failure_does_not_mark_done(
    tmp_path, monkeypatch, shell_operation_cls, expected_message
):
    monkeypatch.setattr(
        "rapthor.execution.shell._load_shell_operation_cls",
        lambda: shell_operation_cls,
    )
    field = FieldStub(
        tmp_path,
        [SectorStub(tmp_path / "inputs", 1), SectorStub(tmp_path / "inputs", 2)],
    )
    operation = Mosaic(field, index=1)

    with (
        prefect_test_harness(server_startup_timeout=None),
        pytest.raises((FileNotFoundError, RuntimeError), match=expected_message),
    ):
        operation.run()

    assert Path(operation.pipeline_inputs_file).is_file()
    assert not Path(operation.done_file).exists()
    assert not Path(operation.outputs_file).exists()
    assert operation.outputs == {}
    assert field.field_image_filename is None
    assert field.field_image_filename_prev is None
