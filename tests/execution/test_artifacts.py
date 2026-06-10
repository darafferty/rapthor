from pathlib import Path

from rapthor.execution.artifacts import (
    ArtifactWriters,
    publish_fits_image_artifacts,
    publish_plot_artifacts,
    publish_plot_file_records,
    render_fits_png,
)


class RecordingArtifactWriters:
    def __init__(self):
        self.calls = []

    @property
    def writers(self):
        return ArtifactWriters(
            image=self.image,
            link=self.link,
            markdown=self.markdown,
        )

    def image(self, **kwargs):
        self.calls.append(("image", kwargs))
        return f"image-{len(self.calls)}"

    def link(self, **kwargs):
        self.calls.append(("link", kwargs))
        return f"link-{len(self.calls)}"

    def markdown(self, **kwargs):
        self.calls.append(("markdown", kwargs))
        return f"markdown-{len(self.calls)}"


def test_publish_plot_artifacts_embeds_images_and_links_other_files(tmp_path):
    plots_dir = tmp_path / "plots"
    calibrate_dir = plots_dir / "calibrate_1"
    image_dir = plots_dir / "image_1"
    calibrate_dir.mkdir(parents=True)
    image_dir.mkdir(parents=True)
    (calibrate_dir / "phase_solutions.png").write_bytes(b"png-data")
    (image_dir / "sector_1.image_diagnostics.json").write_text(
        '{"dynamic_range": 42, "rms": 0.001}'
    )
    (image_dir / "sector_1.photometry.pdf").write_bytes(b"pdf-data")

    recorder = RecordingArtifactWriters()
    records = publish_plot_artifacts(
        plots_dir,
        artifact_writers=recorder.writers,
        in_run_context=lambda: True,
    )

    assert [record["relative_path"] for record in records] == [
        "calibrate_1/phase_solutions.png",
        "image_1/sector_1.image_diagnostics.json",
        "image_1/sector_1.photometry.pdf",
    ]
    assert [record["artifact_type"] for record in records] == ["image", "markdown", "link"]
    assert [call[0] for call in recorder.calls] == ["image", "markdown", "link", "markdown"]

    image_call = recorder.calls[0][1]
    assert image_call["image_url"].startswith("data:image/png;base64,")
    assert image_call["key"].startswith("rapthor-plot-calibrate-1-phase-solutions-png-")
    assert image_call["description"] == "Rapthor plot output: calibrate_1/phase_solutions.png"

    json_call = recorder.calls[1][1]
    assert json_call["key"].startswith("rapthor-plot-image-1-sector-1-image-diagnostics-json-")
    assert json_call["description"] == (
        "Rapthor plot output: image_1/sector_1.image_diagnostics.json"
    )
    assert "```json" in json_call["markdown"]
    assert '"dynamic_range": 42' in json_call["markdown"]
    assert '"rms": 0.001' in json_call["markdown"]

    link_call = recorder.calls[2][1]
    assert link_call["link"].startswith("data:application/pdf;base64,")
    assert link_call["link_text"] == "image_1/sector_1.photometry.pdf"

    markdown_call = recorder.calls[3][1]
    assert markdown_call["key"] == "rapthor-plot-index"
    assert "calibrate_1/phase_solutions.png" in markdown_call["markdown"]
    assert "image_1/sector_1.image_diagnostics.json" in markdown_call["markdown"]
    assert "image_1/sector_1.photometry.pdf" in markdown_call["markdown"]


def test_publish_plot_artifacts_is_noop_without_prefect_context(tmp_path):
    plots_dir = tmp_path / "plots"
    plots_dir.mkdir()
    (plots_dir / "plot.png").write_bytes(b"png-data")
    recorder = RecordingArtifactWriters()

    records = publish_plot_artifacts(
        plots_dir,
        artifact_writers=recorder.writers,
        in_run_context=lambda: False,
    )

    assert records == []
    assert recorder.calls == []


def test_publish_plot_file_records_uses_operation_directory_in_artifact_key(tmp_path):
    pipeline_dir = tmp_path / "pipelines" / "calibrate_1"
    pipeline_dir.mkdir(parents=True)
    plot_file = pipeline_dir / "phase_solutions.png"
    plot_file.write_bytes(b"png-data")
    recorder = RecordingArtifactWriters()

    records = publish_plot_file_records(
        [{"class": "File", "path": str(plot_file)}],
        pipeline_dir,
        artifact_writers=recorder.writers,
        in_run_context=lambda: True,
    )

    assert [record["relative_path"] for record in records] == ["calibrate_1/phase_solutions.png"]
    assert records[0]["artifact_key"].startswith("rapthor-plot-calibrate-1-phase-solutions-png-")
    assert [call[0] for call in recorder.calls] == ["image"]


def test_publish_plot_file_records_renders_json_as_markdown(tmp_path):
    pipeline_dir = tmp_path / "pipelines" / "image_1"
    pipeline_dir.mkdir(parents=True)
    diagnostics = pipeline_dir / "sector_1.image_diagnostics.json"
    diagnostics.write_text('{"noise": 0.2, "sources": 12}')
    recorder = RecordingArtifactWriters()

    records = publish_plot_file_records(
        [{"class": "File", "path": str(diagnostics)}],
        pipeline_dir,
        artifact_writers=recorder.writers,
        in_run_context=lambda: True,
    )

    assert [record["relative_path"] for record in records] == [
        "image_1/sector_1.image_diagnostics.json"
    ]
    assert records[0]["artifact_type"] == "markdown"
    assert [call[0] for call in recorder.calls] == ["markdown"]
    markdown_call = recorder.calls[0][1]
    assert markdown_call["key"].startswith("rapthor-plot-image-1-sector-1-image-diagnostics-json-")
    assert "# `image_1/sector_1.image_diagnostics.json`" in markdown_call["markdown"]
    assert '"noise": 0.2' in markdown_call["markdown"]
    assert '"sources": 12' in markdown_call["markdown"]


def test_render_fits_png_handles_wsclean_like_axes(tmp_path):
    import numpy as np
    from astropy.io import fits

    fits_path = tmp_path / "sector_1-MFS-I-image.fits"
    fits.writeto(fits_path, np.arange(25, dtype=float).reshape(1, 1, 5, 5), overwrite=True)

    png_path = render_fits_png(
        fits_path,
        tmp_path / "previews",
        root_dir=tmp_path,
    )

    assert png_path is not None
    assert png_path.is_file()
    assert png_path.name == "sector-1-mfs-i-image-fits.png"
    assert png_path.read_bytes().startswith(b"\x89PNG")


def test_publish_fits_image_artifacts_renders_png_preview(tmp_path):
    import numpy as np
    from astropy.io import fits

    pipeline_dir = tmp_path / "pipelines" / "image_1"
    pipeline_dir.mkdir(parents=True)
    fits_path = pipeline_dir / "sector_1-MFS-I-image.fits"
    fits.writeto(fits_path, np.arange(16, dtype=float).reshape(4, 4), overwrite=True)
    recorder = RecordingArtifactWriters()

    records = publish_fits_image_artifacts(
        [{"class": "File", "path": str(fits_path)}],
        pipeline_dir,
        artifact_writers=recorder.writers,
        in_run_context=lambda: True,
    )

    assert [record["relative_path"] for record in records] == ["image_1/sector_1-MFS-I-image.fits"]
    assert records[0]["artifact_type"] == "fits-preview"
    assert Path(records[0]["path"]).read_bytes().startswith(b"\x89PNG")
    assert [call[0] for call in recorder.calls] == ["image"]
    image_call = recorder.calls[0][1]
    assert image_call["image_url"].startswith("data:image/png;base64,")
    assert image_call["key"].startswith("rapthor-fits-preview-image-1-sector-1-mfs-i-image-fits-")
    assert image_call["description"] == "Rapthor FITS preview: image_1/sector_1-MFS-I-image.fits"


def test_publish_fits_image_artifacts_skips_fits_tables(tmp_path):
    import numpy as np
    from astropy.io import fits

    pipeline_dir = tmp_path / "pipelines" / "image_1"
    pipeline_dir.mkdir(parents=True)
    catalog_path = pipeline_dir / "sector_1.source_catalog.fits"
    table = fits.BinTableHDU.from_columns(
        [fits.Column(name="flux", array=np.array([1.0, 2.0]), format="E")]
    )
    table.writeto(catalog_path)
    recorder = RecordingArtifactWriters()

    records = publish_fits_image_artifacts(
        [{"class": "File", "path": str(catalog_path)}],
        pipeline_dir,
        artifact_writers=recorder.writers,
        in_run_context=lambda: True,
    )

    assert records == []
    assert recorder.calls == []


def test_publish_plot_artifacts_is_noop_without_plots_directory(tmp_path):
    recorder = RecordingArtifactWriters()

    records = publish_plot_artifacts(
        Path(tmp_path / "missing"),
        artifact_writers=recorder.writers,
        in_run_context=lambda: True,
    )

    assert records == []
    assert recorder.calls == []
