from pathlib import Path

from rapthor.execution.artifacts import (
    ArtifactWriters,
    publish_command_metrics_artifact,
    publish_fits_image_artifacts,
    publish_plot_artifacts,
    publish_plot_file_records,
    render_command_profile_chart,
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


def test_publish_plot_index_keeps_square_brackets_in_file_urls(tmp_path, monkeypatch):
    container_workspace = tmp_path / "container" / "app"
    host_workspace = tmp_path / "host" / "rapthor"
    monkeypatch.setenv("RAPTHOR_CONTAINER_WORKSPACE", str(container_workspace))
    monkeypatch.setenv("RAPTHOR_HOST_WORKSPACE", str(host_workspace))
    plots_dir = container_workspace / "runs" / "demo" / "rapthor-work" / "plots"
    calibrate_dir = plots_dir / "calibrate_4"
    calibrate_dir.mkdir(parents=True)
    plot_file = calibrate_dir / "medium1_phase_dir[Patch_0].png"
    plot_file.write_bytes(b"png-data")
    recorder = RecordingArtifactWriters()

    records = publish_plot_artifacts(
        plots_dir,
        artifact_writers=recorder.writers,
        in_run_context=lambda: True,
    )

    expected_url = (
        (host_workspace / "runs/demo/rapthor-work/plots/calibrate_4/medium1_phase_dir[Patch_0].png")
        .resolve()
        .as_uri()
    )
    assert records[0]["file_url"] == expected_url.replace("%5B", "[").replace("%5D", "]")
    markdown_call = recorder.calls[-1][1]
    assert "medium1_phase_dir[Patch_0].png" in markdown_call["markdown"]
    assert "%5BPatch_0%5D" not in markdown_call["markdown"]


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


def test_publish_command_metrics_artifact_renders_timing_table(tmp_path):
    working_dir = tmp_path / "work"
    log_dir = working_dir / "logs"
    log_dir.mkdir(parents=True)
    (log_dir / "commands.jsonl").write_text(
        "\n".join(
            [
                (
                    '{"operation": "calibrate_1", "name": "solve", '
                    '"status": "completed", "duration_seconds": 12.5, '
                    '"profile": {"resource_metrics": {"cpu_percent": 150.0, '
                    '"max_rss_kb": 1048576, "file_system_inputs": 8, '
                    '"file_system_outputs": 16}}, '
                    '"command_string": "DP3 msin=input.ms"}'
                ),
                (
                    '{"operation": "image_1", '
                    '"status": "failed", "duration_seconds": 61.0, '
                    '"profile": {"resource_metrics": {"cpu_percent": 95.0, '
                    '"max_rss_kb": 2097152, "file_system_inputs": 128, '
                    '"file_system_outputs": 256}}, '
                    '"command": ["wsclean", "-name", "sector"]}'
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    recorder = RecordingArtifactWriters()

    artifact_id = publish_command_metrics_artifact(
        working_dir,
        artifact_writers=recorder.writers,
        in_run_context=lambda: True,
    )

    assert artifact_id == "markdown-1"
    assert [call[0] for call in recorder.calls] == ["markdown", "image"]
    markdown_call = recorder.calls[0][1]
    assert markdown_call["key"] == "rapthor-command-metrics"
    assert "# Rapthor command timings" in markdown_call["markdown"]
    assert "Total recorded external-command time: `1.23 min`" in markdown_call["markdown"]
    assert "## Bottleneck summary" in markdown_call["markdown"]
    assert "- Highest peak memory: `image_1/wsclean` at `2.00 GB`" in markdown_call["markdown"]
    assert (
        "| calibrate_1 | solve | completed | 12.50 s | 150% | 1.00 GB | 8 | 16 | "
        "`DP3 msin=input.ms` |" in markdown_call["markdown"]
    )
    assert (
        "| image_1 | wsclean | failed | 1.02 min | 95% | 2.00 GB | 128 | 256 | "
        "`wsclean -name sector` |" in markdown_call["markdown"]
    )
    image_call = recorder.calls[1][1]
    assert image_call["key"] == "rapthor-command-profile-summary"
    assert image_call["image_url"].startswith("data:image/png;base64,")


def test_publish_command_metrics_artifact_publishes_perf_flamegraphs(tmp_path):
    working_dir = tmp_path / "work"
    log_dir = working_dir / "logs"
    profile_dir = log_dir / "profiles" / "calibrate-1-solve"
    profile_dir.mkdir(parents=True)
    flamegraph_path = profile_dir / "perf.flamegraph.svg"
    flamegraph_path.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg"><text>solve</text></svg>',
        encoding="utf-8",
    )
    (log_dir / "commands.jsonl").write_text(
        (
            '{"operation": "calibrate_1", "name": "solve", '
            '"status": "completed", "duration_seconds": 12.5, '
            '"profile": {"resource_metrics": {"cpu_percent": 150.0}, '
            f'"artifacts": {{"perf_flamegraph": "{flamegraph_path}"}}}}, '
            '"command_string": "DP3 msin=input.ms"}\n'
        ),
        encoding="utf-8",
    )
    recorder = RecordingArtifactWriters()

    publish_command_metrics_artifact(
        working_dir,
        artifact_writers=recorder.writers,
        in_run_context=lambda: True,
    )

    assert [call[0] for call in recorder.calls] == ["markdown", "image", "image"]
    markdown_call = recorder.calls[0][1]
    assert "## Perf flamegraphs" in markdown_call["markdown"]
    assert "`calibrate_1/solve`" in markdown_call["markdown"]

    flamegraph_call = recorder.calls[2][1]
    assert flamegraph_call["key"].startswith("rapthor-command-flamegraph-calibrate-1-solve-")
    assert flamegraph_call["image_url"].startswith("data:image/svg+xml;base64,")
    assert flamegraph_call["description"] == "Rapthor perf flamegraph for calibrate_1/solve."


def test_render_command_profile_chart_writes_png(tmp_path):
    chart = render_command_profile_chart(
        tmp_path / "work",
        [
            {
                "operation": "calibrate_1",
                "name": "solve",
                "duration_seconds": 3.0,
                "profile": {
                    "resource_metrics": {
                        "cpu_percent": 125.0,
                        "max_rss_kb": 1024,
                        "file_system_inputs": 4,
                        "file_system_outputs": 8,
                    }
                },
            }
        ],
    )

    assert chart == tmp_path / "work" / "logs" / "command-profile-summary.png"
    assert chart.read_bytes().startswith(b"\x89PNG")
