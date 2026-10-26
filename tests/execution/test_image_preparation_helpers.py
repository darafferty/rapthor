import shlex
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

import rapthor.execution.image.preparation as preparation_module
import rapthor.execution.outputs as outputs_module
from rapthor.execution.config import ExecutionConfig
from rapthor.execution.image.beam import ensure_image_beam
from rapthor.execution.image.masking import blank_image
from rapthor.execution.image.preparation import (
    concatenate_prepared_visibilities,
    prepare_visibility_ms,
)
from rapthor.execution.regions import make_ds9_region_from_skymodel
from rapthor.lib.records import directory_record
from tests.conftest import file_tree_state

RESOURCE_DIR = Path(__file__).resolve().parents[1] / "resources"


def _image_data_and_header(path):
    with fits.open(path) as hdul:
        return hdul[0].data.copy(), hdul[0].header.copy()


def _write_image(path, *, bmaj=None, bmin=None, bpa=None):
    header = fits.Header()
    if bmaj is not None:
        header["BMAJ"] = bmaj
    if bmin is not None:
        header["BMIN"] = bmin
    if bpa is not None:
        header["BPA"] = bpa
    fits.PrimaryHDU(data=np.zeros((2, 2)), header=header).writeto(path)


def _beam_header(path):
    with fits.open(path) as hdul:
        header = hdul[0].header
        return header["BMAJ"], header["BMIN"], header["BPA"]


def test_blank_image_creates_template_mask(tmp_path):
    """The helper creates a one-filled FITS template when no input image is given."""
    output_image = tmp_path / "test_blank_image.fits"

    blank_image(
        str(output_image),
        reference_ra_deg=10.684,
        reference_dec_deg=41.269,
        cellsize_deg=0.001,
        imsize="6,5",
    )

    data, header = _image_data_and_header(output_image)
    assert data.shape == (1, 1, 5, 6)
    assert np.all(data == 1.0)
    assert header["CRVAL1"] == 10.684
    assert header["CRVAL2"] == 41.269
    assert header["NAXIS1"] == 6
    assert header["NAXIS2"] == 5


def test_ensure_image_beam_adds_missing_beam(tmp_path):
    image = tmp_path / "image.fits"
    _write_image(image)

    ensure_image_beam(str(image), 12.0)

    bmaj, bmin, bpa = _beam_header(image)
    assert bmaj == pytest.approx(12.0 / 3600)
    assert bmin == pytest.approx(12.0 / 3600)
    assert bpa == 0.0


def test_ensure_image_beam_uses_default_for_invalid_fallback_size(tmp_path):
    image = tmp_path / "image.fits"
    _write_image(image, bmaj=0.0, bmin=-1.0)

    ensure_image_beam(str(image), 0.0)

    bmaj, bmin, _ = _beam_header(image)
    assert bmaj == pytest.approx(10.0 / 3600)
    assert bmin == pytest.approx(10.0 / 3600)


def test_ensure_image_beam_preserves_valid_values(tmp_path):
    image = tmp_path / "image.fits"
    _write_image(image, bmaj=0.03, bmin=0.02, bpa=45.0)

    ensure_image_beam(str(image), 12.0)

    assert _beam_header(image) == (0.03, 0.02, 45.0)


def test_make_ds9_region_from_skymodel_writes_regions(tmp_path):
    skymodel = RESOURCE_DIR / "test_true_sky.txt"
    region_file = tmp_path / "region.reg"

    make_ds9_region_from_skymodel(
        str(skymodel),
        258.0,
        57.5,
        3.0,
        2.0,
        str(region_file),
        enclose_names=False,
    )

    region_text = region_file.read_text()
    assert region_text.startswith("# Region file format: DS9 version 4.0")
    assert "polygon(" in region_text
    assert "point(" in region_text
    assert "Patch_patch_1" in region_text


class _MeasurementSetCommands:
    """Fake DP3 and TaQL commands that write an MS and can fail part-way."""

    def __init__(self):
        self.commands = []
        self.fail = set()

    def shell_operation_cls(self, **kwargs):
        commands = self

        class Operation:
            def run(self):
                tokens = shlex.split(kwargs["commands"][-1])
                commands.commands.append(tokens[0])
                if tokens[0] == "DP3":
                    output = next(token[6:] for token in tokens if token.startswith("msout="))
                    source = next(token[5:] for token in tokens if token.startswith("msin="))
                    assert Path(source, "table.dat").read_text() == "original"
                elif tokens[0] == "cp":
                    output = tokens[-1]
                else:
                    output = tokens[tokens.index("giving") + 1].strip('"')
                output = Path(kwargs["working_dir"], output)
                output.mkdir()
                (output / "table.dat").write_text(tokens[0])
                if tokens[0] in commands.fail:
                    raise RuntimeError(f"{tokens[0]} failed")
                return "OK"

        return Operation()


@pytest.fixture
def imaging_restart(tmp_path, monkeypatch):
    """Prepare and concatenate two observations as the image flow does."""
    monkeypatch.setattr(
        preparation_module,
        "select_concatenation_command",
        lambda msfiles, output, **kwargs: ["taql", "select", "from", "giving", f'"{output}"'],
    )
    workdir = tmp_path / "image_1"
    workdir.mkdir()
    inputs = []
    for index in range(2):
        original = tmp_path / f"obs_{index}.ms"
        original.mkdir()
        (original / "table.dat").write_text("original")
        inputs.append(original)
    original_state = [file_tree_state(path) for path in inputs]
    sector = {
        "data_colname": "DATA",
        "phasecenter": "[0deg, 0deg]",
        "max_threads": 1,
        "prepare_data_steps": "[shift]",
        "concat_path": str(workdir / "sector_1_concat.ms"),
    }
    tasks = [
        {
            "msin": str(original),
            "msout": f"sector_1_obs_{index}_prep.ms",
            "msout_path": str(workdir / f"sector_1_obs_{index}_prep.ms"),
            "starttime": "50000.0",
            "ntimes": 10,
            "freqstep": 1,
            "timestep": 1,
            "minchannels": 1,
        }
        for index, original in enumerate(inputs)
    ]
    commands = _MeasurementSetCommands()

    def run():
        config = ExecutionConfig(task_runner="sync")
        prepared = [
            prepare_visibility_ms(
                sector, task, str(workdir), config, shell_operation_cls=commands.shell_operation_cls
            )
            for task in tasks
        ]
        concat = concatenate_prepared_visibilities(
            sector, prepared, str(workdir), config, shell_operation_cls=commands.shell_operation_cls
        )
        return prepared, concat

    yield workdir, commands, run
    assert [file_tree_state(path) for path in inputs] == original_state


def _expected_records(workdir):
    return [
        directory_record(workdir / "sector_1_obs_0_prep.ms"),
        directory_record(workdir / "sector_1_obs_1_prep.ms"),
    ], directory_record(workdir / "sector_1_concat.ms")


def test_imaging_preparation_restarts_after_a_failure_during_creation(imaging_restart):
    workdir, commands, run = imaging_restart
    commands.fail = {"taql"}
    with pytest.raises(RuntimeError, match="taql failed"):
        run()
    # The partial concat MS is kept for inspection but is not promoted.
    assert (workdir / ".sector_1_concat.ms.partial").is_dir()
    assert not (workdir / "sector_1_concat.ms").exists()

    commands.fail = set()
    commands.commands.clear()
    assert run() == _expected_records(workdir)
    # The prepared MSs were complete and are reused.
    assert commands.commands == ["taql"]
    assert not (workdir / ".sector_1_concat.ms.partial").exists()
    assert (workdir / "sector_1_concat.ms" / "table.dat").read_text() == "taql"


def test_imaging_preparation_restarts_after_a_failure_before_promotion(
    imaging_restart, monkeypatch
):
    workdir, commands, run = imaging_restart
    # Stop after the second DP3 command succeeded, before its MS gets its final name.
    promotions = []

    def stop_second_promotion(source, destination):
        promotions.append(destination)
        if len(promotions) == 2:
            raise KeyboardInterrupt
        return original_replace(source, destination)

    original_replace = outputs_module.os.replace
    with monkeypatch.context() as patch:
        patch.setattr(outputs_module.os, "replace", stop_second_promotion)
        with pytest.raises(KeyboardInterrupt):
            run()
    assert (workdir / "sector_1_obs_0_prep.ms").is_dir()
    assert not (workdir / "sector_1_obs_1_prep.ms").exists()

    commands.commands.clear()
    assert run() == _expected_records(workdir)
    assert commands.commands == ["DP3", "taql"]


def test_imaging_preparation_second_restart_reuses_complete_outputs(imaging_restart):
    workdir, commands, run = imaging_restart
    commands.fail = {"taql"}
    with pytest.raises(RuntimeError, match="taql failed"):
        run()
    commands.fail = set()
    first_records = run()
    commands.commands.clear()

    assert run() == first_records == _expected_records(workdir)
    assert commands.commands == []
    assert sorted(path.name for path in workdir.iterdir()) == [
        "sector_1_concat.ms",
        "sector_1_obs_0_prep.ms",
        "sector_1_obs_1_prep.ms",
    ]


def test_concatenation_replaces_a_partial_ms_of_a_stopped_attempt(tmp_path):
    """The command selection must not refuse the partial MS that a stopped attempt left."""
    prepared = tmp_path / "sector_1_obs_0_prep.ms"
    prepared.mkdir()
    (tmp_path / ".sector_1_concat.ms.partial").mkdir()
    sector = {"data_colname": "DATA", "concat_path": str(tmp_path / "sector_1_concat.ms")}
    commands = _MeasurementSetCommands()

    record = concatenate_prepared_visibilities(
        sector,
        [directory_record(prepared)],
        str(tmp_path),
        ExecutionConfig(task_runner="sync"),
        shell_operation_cls=commands.shell_operation_cls,
    )

    assert record == directory_record(tmp_path / "sector_1_concat.ms")
    assert commands.commands == ["cp"]
    assert not (tmp_path / ".sector_1_concat.ms.partial").exists()
