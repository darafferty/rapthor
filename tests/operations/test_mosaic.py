"""
Test cases for the `rapthor.operations.mosaic` module.
"""

from pathlib import Path

import pytest

from rapthor.lib.cwl import CWLFile
from rapthor.operations.mosaic import Mosaic


class SectorStub:
    def __init__(self, root: Path, index: int, pols=("I", "Q")):
        root.mkdir(parents=True, exist_ok=True)
        prefix = root / f"sector_{index}"
        self.vertices_file = str(prefix.with_suffix(".vertices"))
        Path(self.vertices_file).write_text("vertices")

        for pol in pols:
            polup = pol.upper()
            self._write_attr(
                f"{polup}_image_file_true_sky",
                root / f"sector_{index}-{polup}-image.fits",
                "true sky",
            )
            self._write_attr(
                f"{polup}_image_file_apparent_sky",
                root / f"sector_{index}-{polup}-apparent.fits",
                "apparent sky",
            )
            self._write_attr(
                f"{polup}_model_file_true_sky",
                root / f"sector_{index}-{polup}-model.fits",
                "model",
            )
            self._write_attr(
                f"{polup}_residual_file_apparent_sky",
                root / f"sector_{index}-{polup}-residual.fits",
                "residual",
            )
            self._write_attr(
                f"{polup}_dirty_file_apparent_sky",
                root / f"sector_{index}-{polup}-dirty.fits",
                "dirty",
            )

        self._write_attr(
            "filtering_mask_file",
            root / f"sector_{index}-filtering-mask.fits",
            "mask",
        )
        self._write_attr(
            "filtered_model_file_apparent_sky",
            root / f"sector_{index}-filtered-model.fits",
            "filtered model",
        )

    def _write_attr(self, name, path, content):
        Path(path).write_text(content)
        setattr(self, name, str(path))


class FieldStub:
    def __init__(
        self,
        tmp_path,
        sectors,
        *,
        batch_system="single_machine",
        compress_images=False,
        image_pol=("I",),
        disable_clean=True,
        save_supplementary_images=False,
        save_filtered_model_image=False,
    ):
        self.parset = {
            "dir_working": str(tmp_path / "working"),
            "cluster_specific": {
                "cwl_runner": "toil",
                "debug_workflow": False,
                "keep_temporary_files": False,
                "max_nodes": 1,
                "batch_system": batch_system,
                "cpus_per_task": 1,
                "mem_per_node_gb": 0,
                "dir_local": None,
                "local_scratch_dir": None,
                "global_scratch_dir": None,
                "use_container": False,
                "container_type": "docker",
                "max_cores": 7,
                "prefect_task_runner": "sync",
            },
            "imaging_specific": {
                "save_filtered_model_image": save_filtered_model_image,
            },
        }
        self.imaging_sectors = sectors
        self.compress_images = compress_images
        self.image_pol = list(image_pol)
        self.disable_clean = disable_clean
        self.save_supplementary_images = save_supplementary_images
        self.field_image_filename = None
        self.field_image_filename_prev = None


def _field(tmp_path, sector_count=2, **kwargs):
    sectors = [SectorStub(tmp_path / "inputs", index) for index in range(1, sector_count + 1)]
    return FieldStub(tmp_path, sectors, **kwargs)


@pytest.mark.parametrize(
    "batch_system, expected_max_cores",
    [
        ("single_machine", 7),
        ("slurm", None),
    ],
)
def test_set_parset_parameters_records_runtime_inputs(tmp_path, batch_system, expected_max_cores):
    field = _field(tmp_path, batch_system=batch_system, compress_images=True)
    operation = Mosaic(field, index=1)

    operation.set_parset_parameters()

    assert operation.parset_parms == {
        "rapthor_pipeline_dir": operation.rapthor_pipeline_dir,
        "pipeline_working_dir": operation.pipeline_working_dir,
        "max_cores": expected_max_cores,
        "skip_processing": False,
        "compress_images": True,
    }


def test_set_input_parameters_builds_two_sector_mosaic_inputs(tmp_path):
    field = _field(tmp_path, sector_count=2, image_pol=("I",), disable_clean=True)
    operation = Mosaic(field, index=1)

    operation.set_input_parameters()

    true_sky_files = [
        field.imaging_sectors[0].I_image_file_true_sky,
        field.imaging_sectors[1].I_image_file_true_sky,
    ]
    apparent_sky_files = [
        field.imaging_sectors[0].I_image_file_apparent_sky,
        field.imaging_sectors[1].I_image_file_apparent_sky,
    ]
    vertices_files = [
        field.imaging_sectors[0].vertices_file,
        field.imaging_sectors[1].vertices_file,
    ]

    assert operation.image_names == [
        "I_image_file_true_sky",
        "I_image_file_apparent_sky",
    ]
    assert operation.input_parms == {
        "skip_processing": False,
        "sector_image_filename": [
            CWLFile(true_sky_files).to_json(),
            CWLFile(apparent_sky_files).to_json(),
        ],
        "sector_vertices_filename": [
            CWLFile(vertices_files).to_json(),
            CWLFile(vertices_files).to_json(),
        ],
        "template_image_filename": ["mosaic_1_template.fits", "mosaic_1_template.fits"],
        "regridded_image_filename": [
            ["sector_1-I-image.fits.regridded", "sector_2-I-image.fits.regridded"],
            ["sector_1-I-apparent.fits.regridded", "sector_2-I-apparent.fits.regridded"],
        ],
        "mosaic_filename": ["mosaic_1-I-image.fits", "mosaic_1-I-apparent.fits"],
    }


def test_set_input_parameters_includes_clean_and_supplementary_products(tmp_path):
    field = _field(
        tmp_path,
        sector_count=2,
        image_pol=("I", "Q"),
        disable_clean=False,
        save_supplementary_images=True,
        save_filtered_model_image=True,
    )
    operation = Mosaic(field, index=1)

    operation.set_input_parameters()

    assert operation.image_names == [
        "I_image_file_true_sky",
        "I_image_file_apparent_sky",
        "I_model_file_true_sky",
        "I_residual_file_apparent_sky",
        "I_dirty_file_apparent_sky",
        "Q_image_file_true_sky",
        "Q_image_file_apparent_sky",
        "Q_model_file_true_sky",
        "Q_residual_file_apparent_sky",
        "Q_dirty_file_apparent_sky",
        "filtering_mask_file",
        "filtered_model_file_apparent_sky",
    ]
    assert operation.input_parms["mosaic_filename"] == [
        "mosaic_1-I-image.fits",
        "mosaic_1-I-apparent.fits",
        "mosaic_1-I-model.fits",
        "mosaic_1-I-residual.fits",
        "mosaic_1-I-dirty.fits",
        "mosaic_1-Q-image.fits",
        "mosaic_1-Q-apparent.fits",
        "mosaic_1-Q-model.fits",
        "mosaic_1-Q-residual.fits",
        "mosaic_1-Q-dirty.fits",
        "mosaic_1-filtering-mask.fits",
        "mosaic_1-filtered-model.fits",
    ]


def test_set_input_parameters_reuses_single_sector_products_when_processing_is_skipped(tmp_path):
    field = _field(tmp_path, sector_count=1, image_pol=("I",), disable_clean=True)
    operation = Mosaic(field, index=1)

    operation.set_input_parameters()

    assert operation.skip_processing is True
    assert operation.input_parms["skip_processing"] is True
    assert operation.input_parms["mosaic_filename"] == [
        field.imaging_sectors[0].I_image_file_true_sky,
        field.imaging_sectors[0].I_image_file_apparent_sky,
    ]


def test_finalize_copies_single_sector_products_and_updates_field_state(tmp_path):
    field = _field(tmp_path, sector_count=1, image_pol=("I",), disable_clean=True)
    field.field_image_filename = "previous-field-image.fits"
    operation = Mosaic(field, index=1)
    operation.set_input_parameters()

    operation.finalize()

    expected_field_image = (
        Path(field.parset["dir_working"]) / "images" / "image_1" / "field-I-image.fits"
    )
    expected_apparent_image = (
        Path(field.parset["dir_working"]) / "images" / "image_1" / "field-I-apparent.fits"
    )
    assert field.field_image_filename_prev == "previous-field-image.fits"
    assert field.field_image_filename == str(expected_field_image)
    assert expected_field_image.read_text() == "true sky"
    assert expected_apparent_image.read_text() == "apparent sky"
    assert Path(operation.done_file).is_file()


def test_finalize_copies_compressed_processed_mosaic_products(tmp_path):
    field = _field(
        tmp_path,
        sector_count=2,
        compress_images=True,
        image_pol=("I",),
        disable_clean=True,
    )
    operation = Mosaic(field, index=1)
    operation.set_input_parameters()
    for mosaic_filename in operation.mosaic_filename:
        (Path(operation.pipeline_working_dir) / f"{mosaic_filename}.fz").write_text("compressed")

    operation.finalize()

    expected_field_image = (
        Path(field.parset["dir_working"]) / "images" / "image_1" / "field-I-image.fits.fz"
    )
    expected_apparent_image = (
        Path(field.parset["dir_working"]) / "images" / "image_1" / "field-I-apparent.fits.fz"
    )
    assert field.field_image_filename == str(expected_field_image)
    assert expected_field_image.read_text() == "compressed"
    assert expected_apparent_image.read_text() == "compressed"
