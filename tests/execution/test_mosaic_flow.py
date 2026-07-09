import json
import shlex
from pathlib import Path

import pytest
from prefect.testing.utilities import prefect_test_harness

import rapthor.execution.mosaic.flow as mosaic_module
from rapthor.execution.config import ExecutionConfig
from rapthor.execution.mosaic.commands import build_compress_mosaic_command
from rapthor.execution.mosaic.flow import (
    mosaic_flow,
)
from rapthor.execution.mosaic.payloads import mosaic_payload_from_inputs, validate_mosaic_payload
from rapthor.lib.records import file_record, validate_output_record
from rapthor.operations.mosaic import Mosaic
from tests.execution.conftest import run_flow_for_test

FIXTURE_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def fake_direct_mosaic_helpers(monkeypatch):
    calls = {
        "make_mosaic_template": [],
        "regrid_image": [],
        "regrid_sparse_model_image": [],
        "make_mosaic": [],
    }

    def fake_make_mosaic_template(input_image_filenames, vertices_filenames, output_image):
        calls["make_mosaic_template"].append(
            {
                "input_image_filenames": list(input_image_filenames),
                "vertices_filenames": list(vertices_filenames),
                "output_image": output_image,
            }
        )
        output_path = Path(output_image)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("template")

    def fake_regrid_image(input_image, template_image, vertices_file, output_image):
        calls["regrid_image"].append(
            {
                "input_image": input_image,
                "template_image": template_image,
                "vertices_file": vertices_file,
                "output_image": output_image,
            }
        )
        output_path = Path(output_image)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("regridded")

    def fake_regrid_sparse_model_image(input_image, template_image, vertices_file, output_image):
        calls["regrid_sparse_model_image"].append(
            {
                "input_image": input_image,
                "template_image": template_image,
                "vertices_file": vertices_file,
                "output_image": output_image,
            }
        )
        output_path = Path(output_image)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("regridded model")

    def fake_make_mosaic(input_image_filenames, template_image, output_image):
        calls["make_mosaic"].append(
            {
                "input_image_filenames": list(input_image_filenames),
                "template_image": template_image,
                "output_image": output_image,
            }
        )
        output_path = Path(output_image)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("mosaic")

    monkeypatch.setattr(mosaic_module, "make_mosaic_template", fake_make_mosaic_template)
    monkeypatch.setattr(mosaic_module, "regrid_image", fake_regrid_image)
    monkeypatch.setattr(mosaic_module, "regrid_sparse_model_image", fake_regrid_sparse_model_image)
    monkeypatch.setattr(mosaic_module, "make_mosaic", fake_make_mosaic)
    return calls


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
            if tokens[0] == "fpack":
                output_path = cwd / f"{tokens[1]}.fz"
            else:
                raise AssertionError(f"Unexpected command: {tokens[0]}")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text("OK")
            return "OK"

    return FakeMosaicShellOperation


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


def _mosaic_payload(tmp_path, compress_images=False, mosaic_product_count=1):
    mosaic_products = [
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
    ]
    if mosaic_product_count > 1:
        mosaic_products.append(
            {
                "sector_image_filenames": [
                    "sector_1-I-apparent.fits",
                    "sector_2-I-apparent.fits",
                ],
                "sector_vertices_filenames": ["sector_1.vertices", "sector_2.vertices"],
                "template_image_filename": "mosaic_1_template.fits",
                "template_image_path": str(tmp_path / "mosaic_1_template.fits"),
                "regridded_image_filenames": [
                    "sector_1-I-apparent.fits.regridded",
                    "sector_2-I-apparent.fits.regridded",
                ],
                "mosaic_filename": "mosaic_1-I-apparent.fits",
                "mosaic_path": str(tmp_path / "mosaic_1-I-apparent.fits"),
            }
        )
    return {
        "pipeline_working_dir": str(tmp_path),
        "compress_images": compress_images,
        "skip_processing": False,
        "mosaic_products": mosaic_products,
    }


def test_mosaic_command_builders_match_reference_fixtures():
    commands = json.loads((FIXTURE_DIR / "command_reference.json").read_text())

    assert (
        build_compress_mosaic_command("mosaic_1-I-image.fits")
        == commands["mosaic"]["compress_mosaic_image"]
    )


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
        "mosaic_products": [
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
        "mosaic_products": [],
    }


def test_validate_mosaic_payload_validates_mosaic_product_contract(tmp_path):
    payload = _mosaic_payload(tmp_path)

    validated = validate_mosaic_payload(payload)

    assert validated["mosaic_products"][0]["mosaic_filename"] == "mosaic_1-I-image.fits"
    assert validated["mosaic_products"][0]["regridded_image_filenames"] == [
        "sector_1-I-image.fits.regridded",
        "sector_2-I-image.fits.regridded",
    ]


def test_validate_mosaic_payload_rejects_duplicate_outputs(tmp_path):
    payload = _mosaic_payload(tmp_path)
    duplicate = dict(payload["mosaic_products"][0])
    duplicate["template_image_filename"] = "mosaic_2_template.fits"
    duplicate["template_image_path"] = str(tmp_path / "mosaic_2_template.fits")
    payload["mosaic_products"].append(duplicate)

    with pytest.raises(ValueError, match="mosaic paths must be unique"):
        validate_mosaic_payload(payload)


def test_validate_mosaic_payload_rejects_multiple_template_paths(tmp_path):
    payload = _mosaic_payload(tmp_path, mosaic_product_count=2)
    payload["mosaic_products"][1]["template_image_filename"] = "mosaic_2_template.fits"
    payload["mosaic_products"][1]["template_image_path"] = str(tmp_path / "mosaic_2_template.fits")

    with pytest.raises(ValueError, match="must share one template path"):
        validate_mosaic_payload(payload)


@pytest.mark.parametrize(
    ("mosaic_filename", "expected_run_name"),
    [
        ("mosaic_1-I-image.fits", "mosaic_I_image"),
        ("mosaic_1-MFS-image-pb.fits.fz", "mosaic_MFS_image_pb"),
        ("mosaic_1-MFS-image-pb-ast.fits.fz", "mosaic_MFS_image_pb_ast"),
        ("mosaic_1-MFS-model-pb.fits.fz", "mosaic_MFS_model_pb"),
        ("mosaic_1-MFS-residual.fits.fz", "mosaic_MFS_residual"),
        ("mosaic_1-MFS-dirty.fits.fz", "mosaic_MFS_dirty"),
    ],
)
def test_mosaic_task_run_name_uses_output_product_label(mosaic_filename, expected_run_name):
    mosaic_product = {"mosaic_filename": mosaic_filename}

    assert mosaic_module._mosaic_task_run_name(mosaic_product, index=0) == expected_run_name


def test_run_mosaic_flow_executes_python_helpers_and_returns_records(
    tmp_path, monkeypatch, fake_mosaic_shell_operation_cls, fake_direct_mosaic_helpers
):
    published = []

    def fake_publish_fits_image_artifacts(records, root_dir, *, clip_percentile):
        published.append(
            ([Path(record["path"]).name for record in records], root_dir, clip_percentile)
        )
        return []

    monkeypatch.setattr(
        mosaic_module,
        "publish_fits_image_artifacts",
        fake_publish_fits_image_artifacts,
    )

    outputs = run_flow_for_test(
        mosaic_flow,
        _mosaic_payload(tmp_path),
        execution_config=ExecutionConfig(
            task_runner="sync",
            publish_fits_previews=True,
            fits_preview_clip_percentile=99.7,
        ),
        shell_operation_cls=fake_mosaic_shell_operation_cls,
    )

    assert outputs == {"mosaic_image": [file_record(tmp_path / "mosaic_1-I-image.fits")]}
    assert published == [(["mosaic_1-I-image.fits"], str(tmp_path), 99.7)]
    validate_output_record(outputs["mosaic_image"])
    assert fake_mosaic_shell_operation_cls.instances == []
    assert fake_direct_mosaic_helpers["make_mosaic_template"] == [
        {
            "input_image_filenames": [
                str(tmp_path / "sector_1-I-image.fits"),
                str(tmp_path / "sector_2-I-image.fits"),
            ],
            "vertices_filenames": [
                str(tmp_path / "sector_1.vertices"),
                str(tmp_path / "sector_2.vertices"),
            ],
            "output_image": str(tmp_path / "mosaic_1_template.fits"),
        }
    ]
    assert fake_direct_mosaic_helpers["regrid_image"] == [
        {
            "input_image": str(tmp_path / "sector_1-I-image.fits"),
            "template_image": str(tmp_path / "mosaic_1_template.fits"),
            "vertices_file": str(tmp_path / "sector_1.vertices"),
            "output_image": str(tmp_path / "sector_1-I-image.fits.regridded"),
        },
        {
            "input_image": str(tmp_path / "sector_2-I-image.fits"),
            "template_image": str(tmp_path / "mosaic_1_template.fits"),
            "vertices_file": str(tmp_path / "sector_2.vertices"),
            "output_image": str(tmp_path / "sector_2-I-image.fits.regridded"),
        },
    ]
    assert fake_direct_mosaic_helpers["make_mosaic"] == [
        {
            "input_image_filenames": [
                str(tmp_path / "sector_1-I-image.fits.regridded"),
                str(tmp_path / "sector_2-I-image.fits.regridded"),
            ],
            "template_image": str(tmp_path / "mosaic_1_template.fits"),
            "output_image": str(tmp_path / "mosaic_1-I-image.fits"),
        }
    ]


def test_run_mosaic_flow_builds_shared_template_once_for_multiple_mosaic_products(
    tmp_path, fake_mosaic_shell_operation_cls, fake_direct_mosaic_helpers
):
    outputs = run_flow_for_test(
        mosaic_flow,
        _mosaic_payload(tmp_path, mosaic_product_count=2),
        execution_config=ExecutionConfig(task_runner="sync"),
        shell_operation_cls=fake_mosaic_shell_operation_cls,
    )

    assert outputs == {
        "mosaic_image": [
            file_record(tmp_path / "mosaic_1-I-image.fits"),
            file_record(tmp_path / "mosaic_1-I-apparent.fits"),
        ]
    }
    assert len(fake_direct_mosaic_helpers["make_mosaic_template"]) == 1
    assert len(fake_direct_mosaic_helpers["regrid_image"]) == 4
    assert len(fake_direct_mosaic_helpers["make_mosaic"]) == 2
    assert {call["template_image"] for call in fake_direct_mosaic_helpers["regrid_image"]} == {
        str(tmp_path / "mosaic_1_template.fits")
    }
    assert fake_direct_mosaic_helpers["regrid_sparse_model_image"] == []


def test_run_mosaic_flow_uses_sparse_regrid_for_model_products(
    tmp_path, fake_mosaic_shell_operation_cls, fake_direct_mosaic_helpers
):
    payload = _mosaic_payload(tmp_path)
    payload["mosaic_products"][0].update(
        {
            "sector_image_filenames": [
                "sector_1-MFS-model-pb.fits",
                "sector_2-MFS-model-pb.fits",
            ],
            "regridded_image_filenames": [
                "sector_1-MFS-model-pb.fits.regridded",
                "sector_2-MFS-model-pb.fits.regridded",
            ],
            "mosaic_filename": "mosaic_1-MFS-model-pb.fits",
            "mosaic_path": str(tmp_path / "mosaic_1-MFS-model-pb.fits"),
        }
    )

    outputs = run_flow_for_test(
        mosaic_flow,
        payload,
        execution_config=ExecutionConfig(task_runner="sync"),
        shell_operation_cls=fake_mosaic_shell_operation_cls,
    )

    assert outputs == {"mosaic_image": [file_record(tmp_path / "mosaic_1-MFS-model-pb.fits")]}
    assert fake_direct_mosaic_helpers["regrid_image"] == []
    assert fake_direct_mosaic_helpers["regrid_sparse_model_image"] == [
        {
            "input_image": str(tmp_path / "sector_1-MFS-model-pb.fits"),
            "template_image": str(tmp_path / "mosaic_1_template.fits"),
            "vertices_file": str(tmp_path / "sector_1.vertices"),
            "output_image": str(tmp_path / "sector_1-MFS-model-pb.fits.regridded"),
        },
        {
            "input_image": str(tmp_path / "sector_2-MFS-model-pb.fits"),
            "template_image": str(tmp_path / "mosaic_1_template.fits"),
            "vertices_file": str(tmp_path / "sector_2.vertices"),
            "output_image": str(tmp_path / "sector_2-MFS-model-pb.fits.regridded"),
        },
    ]
    assert fake_direct_mosaic_helpers["make_mosaic"] == [
        {
            "input_image_filenames": [
                str(tmp_path / "sector_1-MFS-model-pb.fits.regridded"),
                str(tmp_path / "sector_2-MFS-model-pb.fits.regridded"),
            ],
            "template_image": str(tmp_path / "mosaic_1_template.fits"),
            "output_image": str(tmp_path / "mosaic_1-MFS-model-pb.fits"),
        }
    ]


def test_run_mosaic_flow_can_skip_fits_preview_artifacts(
    tmp_path, monkeypatch, fake_mosaic_shell_operation_cls
):
    published = []

    def fake_publish_fits_image_artifacts(records, root_dir, *, clip_percentile):
        published.append((records, root_dir, clip_percentile))
        return []

    monkeypatch.setattr(
        mosaic_module,
        "publish_fits_image_artifacts",
        fake_publish_fits_image_artifacts,
    )

    outputs = run_flow_for_test(
        mosaic_flow,
        _mosaic_payload(tmp_path),
        execution_config=ExecutionConfig(task_runner="sync", publish_fits_previews=False),
        shell_operation_cls=fake_mosaic_shell_operation_cls,
    )

    assert outputs == {"mosaic_image": [file_record(tmp_path / "mosaic_1-I-image.fits")]}
    assert published == []


def test_run_mosaic_flow_rejects_invalid_mosaic_product_lists(
    tmp_path, fake_mosaic_shell_operation_cls
):
    payload = _mosaic_payload(tmp_path)
    payload["mosaic_products"][0]["sector_image_filenames"] = ["sector_1-I-image.fits", 7]

    with pytest.raises(ValueError, match="sector_image_filenames"):
        run_flow_for_test(
            mosaic_flow,
            payload,
            execution_config=ExecutionConfig(task_runner="sync"),
            shell_operation_cls=fake_mosaic_shell_operation_cls,
        )

    assert fake_mosaic_shell_operation_cls.instances == []


def test_run_mosaic_flow_returns_compressed_records(tmp_path, fake_mosaic_shell_operation_cls):
    outputs = run_flow_for_test(
        mosaic_flow,
        _mosaic_payload(tmp_path, compress_images=True),
        execution_config=ExecutionConfig(task_runner="sync"),
        shell_operation_cls=fake_mosaic_shell_operation_cls,
    )

    assert outputs == {"mosaic_image": [file_record(tmp_path / "mosaic_1-I-image.fits.fz")]}
    assert [
        instance.kwargs["commands"][0] for instance in fake_mosaic_shell_operation_cls.instances
    ] == [
        "fpack mosaic_1-I-image.fits",
    ]


def test_run_mosaic_flow_handles_skip_processing_without_commands(
    tmp_path, fake_mosaic_shell_operation_cls, fake_direct_mosaic_helpers
):
    outputs = run_flow_for_test(
        mosaic_flow,
        {
            "pipeline_working_dir": str(tmp_path),
            "compress_images": False,
            "skip_processing": True,
            "mosaic_products": [],
        },
        execution_config=ExecutionConfig(task_runner="sync"),
        shell_operation_cls=fake_mosaic_shell_operation_cls,
    )

    assert outputs == {}
    assert fake_mosaic_shell_operation_cls.instances == []
    assert fake_direct_mosaic_helpers == {
        "make_mosaic_template": [],
        "regrid_image": [],
        "regrid_sparse_model_image": [],
        "make_mosaic": [],
    }


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
    assert fake_mosaic_shell_operation_cls.instances == []


def test_run_mosaic_flow_fails_when_expected_output_is_missing(tmp_path, monkeypatch):
    def fake_missing_mosaic(*args, **kwargs):
        return None

    monkeypatch.setattr(mosaic_module, "make_mosaic", fake_missing_mosaic)

    with pytest.raises(FileNotFoundError, match="Mosaic output was not created"):
        run_flow_for_test(
            mosaic_flow,
            _mosaic_payload(tmp_path),
            execution_config=ExecutionConfig(task_runner="sync"),
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
    outputs = run_flow_for_test(
        mosaic_flow,
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
    tmp_path, monkeypatch, fake_mosaic_shell_operation_cls, fake_direct_mosaic_helpers
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
    assert field.field_image_filename == str(expected_field_image)
    assert field.field_image_filename_prev is None
    assert expected_field_image.is_file()
    assert fake_mosaic_shell_operation_cls.instances == []
    assert len(fake_direct_mosaic_helpers["make_mosaic_template"]) == 1
    assert len(fake_direct_mosaic_helpers["regrid_image"]) == 4
    assert fake_direct_mosaic_helpers["regrid_sparse_model_image"] == []
    assert len(fake_direct_mosaic_helpers["make_mosaic"]) == 2


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
    "failure_mode, expected_message",
    [
        pytest.param("raise", "mosaic failed", id="helper-failure"),
        pytest.param("missing-output", "Mosaic output was not created", id="missing-output"),
    ],
)
def test_mosaic_operation_run_failure_does_not_mark_done(
    tmp_path, monkeypatch, failure_mode, expected_message
):
    def fail_mosaic(*args, **kwargs):
        raise RuntimeError("mosaic failed")

    def missing_mosaic(*args, **kwargs):
        return None

    if failure_mode == "raise":
        monkeypatch.setattr(mosaic_module, "make_mosaic", fail_mosaic)
    else:
        monkeypatch.setattr(mosaic_module, "make_mosaic", missing_mosaic)
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
