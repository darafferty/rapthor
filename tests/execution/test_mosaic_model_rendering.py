from pathlib import Path

import numpy as np
from astropy.io import fits as pyfits
from astropy.wcs import WCS

import rapthor.execution.mosaic.model_rendering as model_rendering
from rapthor.execution.commands import normalize_command
from rapthor.execution.config import ExecutionConfig
from rapthor.execution.mosaic.model_rendering import render_model_mosaic_with_wsclean
from rapthor.lib.records import file_record


def test_render_model_mosaic_with_wsclean_uses_template_geometry(tmp_path, monkeypatch):
    template = tmp_path / "mosaic_template.fits"
    output = tmp_path / "mosaic_1-MFS-model-pb.fits"
    combined_skymodel = tmp_path / "mosaic_1-MFS-model-pb.skymodel"
    calls = {"combine": [], "commands": []}

    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---SIN", "DEC--SIN"]
    wcs.wcs.cdelt = [-0.01, 0.01]
    wcs.wcs.crval = [180.0, 45.0]
    wcs.wcs.crpix = [4.0, 3.0]
    header = wcs.to_header()
    header["FREQ"] = 150000000.0
    pyfits.PrimaryHDU(data=np.zeros((5, 7)), header=header).writeto(template)

    def fake_combine_sector_skymodels(sector_skymodels, output_skymodel):
        calls["combine"].append((list(sector_skymodels), output_skymodel))
        Path(output_skymodel).write_text("combined sky model")
        return file_record(output_skymodel)

    def fake_run_external_command(command, pipeline_working_dir, execution_config, **kwargs):
        calls["commands"].append(
            (normalize_command(command), pipeline_working_dir, execution_config, kwargs)
        )
        root = Path(command[command.index("-name") + 1])
        root.with_name(f"{root.name}-term-0.fits").write_text("drawn model")

    monkeypatch.setattr(
        model_rendering,
        "combine_sector_skymodels",
        fake_combine_sector_skymodels,
    )
    monkeypatch.setattr(model_rendering, "run_external_command", fake_run_external_command)

    result = render_model_mosaic_with_wsclean(
        ["sector_1.true_sky.txt", "sector_2.true_sky.txt"],
        str(template),
        str(output),
        str(tmp_path),
        ExecutionConfig(task_runner="sync", cpus_per_task=6),
    )

    assert result == file_record(output)
    assert output.read_text() == "drawn model"
    assert calls["combine"] == [
        (
            ["sector_1.true_sky.txt", "sector_2.true_sky.txt"],
            str(combined_skymodel),
        )
    ]
    command = calls["commands"][0][0]
    assert command[:5] == [
        "wsclean",
        "-j",
        "6",
        "-draw-model",
        str(combined_skymodel),
    ]
    assert command[command.index("-name") + 1] == str(output.with_suffix(""))
    assert command[
        command.index("-draw-frequencies") + 1 : command.index("-draw-frequencies") + 3
    ] == [
        "150000000.0",
        "1000000.0",
    ]
    assert command[command.index("-size") + 1 : command.index("-size") + 3] == ["7", "5"]
    assert command[command.index("-scale") + 1] == "0.01"
