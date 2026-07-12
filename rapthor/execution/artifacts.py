"""Prefect artifact publication helpers for Rapthor outputs."""

import base64
import hashlib
import json
import logging
import mimetypes
import os
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional

from rapthor.execution.prefect_context import in_prefect_run_context
from rapthor.execution.task_metrics import TASK_LOG_FILENAME

log = logging.getLogger("rapthor")

IMAGE_ARTIFACT_SUFFIXES = {".gif", ".jpeg", ".jpg", ".png", ".svg", ".webp"}
FITS_ARTIFACT_SUFFIXES = (".fit", ".fits", ".fit.fz", ".fits.fz")
FITS_PREVIEW_ARTIFACT_KEY_PREFIX = "rapthor-fits-preview"
POSTAGE_STAMP_ARTIFACT_KEY_PREFIX = "rapthor-postage-stamp-preview"
JSON_ARTIFACT_SUFFIXES = {".json"}
PLOT_ARTIFACT_KEY_PREFIX = "rapthor-plot"
COMMAND_METRICS_ARTIFACT_KEY = "rapthor-command-metrics"
COMMAND_PROFILE_SUMMARY_ARTIFACT_KEY = "rapthor-command-profile-summary"
COMMAND_FLAMEGRAPH_ARTIFACT_KEY_PREFIX = "rapthor-command-flamegraph"
COMMAND_LOG_FILENAME = "commands.jsonl"


@dataclass(frozen=True)
class ArtifactWriters:
    """Callables used to create Prefect artifacts."""

    image: Callable[..., object]
    link: Callable[..., object]
    markdown: Callable[..., object]


def publish_file_artifacts(
    file_paths: list[Path],
    root_dir: Path,
    *,
    include_root_name: bool = False,
    artifact_writers: Optional[ArtifactWriters] = None,
    in_run_context: Callable[[], bool] = in_prefect_run_context,
    publish_index: bool = False,
) -> list[dict]:
    """Publish plot files as Prefect artifacts."""
    files = sorted(Path(path) for path in file_paths if Path(path).is_file())
    if not files:
        return []
    if not in_run_context():
        return []

    writers = artifact_writers or _prefect_artifact_writers()
    records = []
    root_dir = Path(root_dir)
    for plot_file in files:
        relative_path = _relative_artifact_path(plot_file, root_dir, include_root_name)
        artifact_key = _plot_artifact_key(relative_path)
        description = f"Rapthor plot output: {relative_path}"
        artifact_url = _data_url(plot_file)
        file_url = _local_file_url(plot_file)

        if _is_image_artifact(plot_file):
            artifact_id = writers.image(
                image_url=artifact_url,
                key=artifact_key,
                description=description,
            )
            artifact_type = "image"
        elif _is_json_artifact(plot_file):
            artifact_id = writers.markdown(
                markdown=_json_markdown(plot_file, relative_path),
                key=artifact_key,
                description=description,
            )
            artifact_type = "markdown"
        else:
            artifact_id = writers.link(
                link=artifact_url,
                link_text=relative_path,
                key=artifact_key,
                description=description,
            )
            artifact_type = "link"

        records.append(
            {
                "artifact_id": artifact_id,
                "artifact_key": artifact_key,
                "artifact_type": artifact_type,
                "file_url": file_url,
                "path": str(plot_file),
                "relative_path": relative_path,
            }
        )

    if publish_index:
        writers.markdown(
            markdown=_plot_index_markdown(root_dir, records),
            key="rapthor-plot-index",
            description="Index of Rapthor plot artifacts for this run.",
        )
    log.info("Published %d Rapthor plot artifacts from %s", len(records), root_dir)
    return records


def publish_fits_image_artifacts(
    file_records: list[Mapping[str, object]],
    root_dir: Path,
    *,
    clip_percentile: float = 99.9,
    artifact_writers: Optional[ArtifactWriters] = None,
    in_run_context: Callable[[], bool] = in_prefect_run_context,
    preview_dir: Optional[Path] = None,
) -> list[dict]:
    """Render FITS image records to PNG previews and publish them as image artifacts."""
    fits_paths = [path for path in _file_record_paths(file_records) if _is_fits_artifact(path)]
    if not fits_paths or not in_run_context():
        return []

    root_dir = Path(root_dir)
    preview_dir = Path(preview_dir or root_dir / ".rapthor-artifacts" / "fits-previews")
    writers = artifact_writers or _prefect_artifact_writers()
    records = []
    for fits_path in sorted(fits_paths):
        relative_path = _relative_artifact_path(fits_path, root_dir, include_root_name=True)
        try:
            png_path = render_fits_png(
                fits_path,
                preview_dir,
                root_dir=root_dir,
                clip_percentile=clip_percentile,
            )
        except Exception as err:
            log.warning("Failed to render FITS preview for %s: %s", fits_path, err)
            continue
        if png_path is None:
            continue

        artifact_key = _fits_preview_artifact_key(relative_path)
        artifact_id = writers.image(
            image_url=_data_url(png_path),
            key=artifact_key,
            description=f"Rapthor FITS preview: {relative_path}",
        )
        records.append(
            {
                "artifact_id": artifact_id,
                "artifact_key": artifact_key,
                "artifact_type": "fits-preview",
                "fits_path": str(fits_path),
                "path": str(png_path),
                "relative_path": relative_path,
                "clip_percentile": float(clip_percentile),
            }
        )

    if records:
        log.info("Published %d Rapthor FITS preview artifacts from %s", len(records), root_dir)
    return records


def publish_fits_postage_stamp_artifacts(
    image_record: Mapping[str, object],
    source_catalog_record: Mapping[str, object],
    root_dir: Path,
    *,
    max_sources: int = 5,
    stamp_size_px: int = 96,
    clip_percentile: float = 99.9,
    artifact_writers: Optional[ArtifactWriters] = None,
    in_run_context: Callable[[], bool] = in_prefect_run_context,
    preview_dir: Optional[Path] = None,
) -> list[dict]:
    """Render small previews around the brightest catalog sources and publish if possible."""
    if (
        image_record.get("class") != "File"
        or source_catalog_record.get("class") != "File"
        or max_sources <= 0
    ):
        return []

    root_dir = Path(root_dir)
    image_path = Path(str(image_record["path"]))
    source_catalog_path = Path(str(source_catalog_record["path"]))
    preview_dir = Path(preview_dir or _default_postage_stamp_dir(root_dir))
    try:
        stamp_records = render_fits_postage_stamp_pngs(
            image_path,
            source_catalog_path,
            preview_dir,
            root_dir=root_dir,
            max_sources=max_sources,
            stamp_size_px=stamp_size_px,
            clip_percentile=clip_percentile,
        )
    except Exception as err:
        log.warning(
            "Failed to render FITS postage-stamp previews for %s using %s: %s",
            image_path,
            source_catalog_path,
            err,
        )
        return []

    if not stamp_records:
        return []

    relative_path = _relative_artifact_path(image_path, root_dir, include_root_name=True)
    should_publish = in_run_context()
    writers = (artifact_writers or _prefect_artifact_writers()) if should_publish else None
    records = []
    for stamp_record in stamp_records:
        source_rank = int(stamp_record["source_rank"])
        source_relative_path = f"{relative_path}:source-{source_rank:02d}"
        artifact_key = _artifact_key(POSTAGE_STAMP_ARTIFACT_KEY_PREFIX, source_relative_path)
        record = {
            "artifact_key": artifact_key,
            "artifact_type": "fits-postage-stamp-preview",
            "fits_path": str(image_path),
            "fits_paths": stamp_record["fits_paths"],
            "path": str(stamp_record["path"]),
            "relative_path": source_relative_path,
            "source_catalog_path": str(source_catalog_path),
            "source_rank": source_rank,
            "source_ra_deg": stamp_record["source_ra_deg"],
            "source_dec_deg": stamp_record["source_dec_deg"],
            "source_coordinate_text": stamp_record["source_coordinate_text"],
            "source_flux": stamp_record["source_flux"],
            "source_flux_column": stamp_record["source_flux_column"],
            "clip_percentile": stamp_record["clip_percentile"],
            "display_vmin": stamp_record["display_vmin"],
            "display_vmax": stamp_record["display_vmax"],
        }
        if writers is not None:
            artifact_id = writers.image(
                image_url=_data_url(stamp_record["path"]),
                key=artifact_key,
                description=f"Rapthor FITS postage-stamp preview: {source_relative_path}",
            )
            record["artifact_id"] = artifact_id
        records.append(record)

    action = "Published" if should_publish else "Rendered"
    log.info("%s %d Rapthor FITS postage-stamp previews from %s", action, len(records), root_dir)
    return records


def publish_plot_artifacts(
    plots_dir: Path,
    artifact_writers: Optional[ArtifactWriters] = None,
    in_run_context: Callable[[], bool] = in_prefect_run_context,
    publish_index: bool = True,
) -> list[dict]:
    """Publish every file below a Rapthor ``plots`` directory as a Prefect artifact."""
    plots_dir = Path(plots_dir)
    return publish_file_artifacts(
        _plot_files(plots_dir),
        plots_dir,
        artifact_writers=artifact_writers,
        in_run_context=in_run_context,
        publish_index=publish_index,
    )


def publish_plot_file_records(
    file_records: list[Mapping[str, object]],
    root_dir: Path,
    artifact_writers: Optional[ArtifactWriters] = None,
    in_run_context: Callable[[], bool] = in_prefect_run_context,
) -> list[dict]:
    """Publish newly produced plot file records immediately."""
    paths = _file_record_paths(file_records)
    return publish_file_artifacts(
        paths,
        Path(root_dir),
        include_root_name=True,
        artifact_writers=artifact_writers,
        in_run_context=in_run_context,
    )


def publish_plot_artifacts_for_field(field: object, publish_index: bool = True) -> list[dict]:
    """Publish plot artifacts for a completed Rapthor field, if possible."""
    parset = getattr(field, "parset", {})
    if not isinstance(parset, dict):
        return []

    working_dir = parset.get("dir_working")
    if not working_dir:
        return []

    return publish_plot_artifacts(Path(working_dir) / "plots", publish_index=publish_index)


def publish_fits_image_artifacts_for_field(field: object) -> list[dict]:
    """Publish FITS preview artifacts for finalized image products, if possible."""
    parset = getattr(field, "parset", {})
    if not isinstance(parset, dict):
        return []

    cluster = parset.get("cluster_specific", parset.get("cluster", {}))
    if not isinstance(cluster, Mapping) or not _bool_setting_is_enabled(
        cluster.get("prefect_publish_fits_previews")
    ):
        return []

    working_dir = parset.get("dir_working")
    if not working_dir:
        return []

    images_dir = Path(working_dir) / "images"
    if not images_dir.is_dir():
        return []

    records = [
        {"class": "File", "path": str(path)}
        for path in sorted(images_dir.rglob("*"))
        if path.is_file() and _is_fits_artifact(path)
    ]
    return publish_fits_image_artifacts(
        records,
        images_dir,
        clip_percentile=float(cluster.get("prefect_fits_preview_clip_percentile", 99.9)),
    )


def publish_command_metrics_artifact(
    working_dir: Path,
    *,
    artifact_writers: Optional[ArtifactWriters] = None,
    in_run_context: Callable[[], bool] = in_prefect_run_context,
) -> Optional[object]:
    """Publish a Markdown summary of external-command timings, if available."""
    working_dir = Path(working_dir)
    records = _command_metric_records(working_dir)
    task_records = _task_metric_records(working_dir)
    if not records and not task_records:
        return None
    if not in_run_context():
        return None

    writers = artifact_writers or _prefect_artifact_writers()
    artifact_id = writers.markdown(
        markdown=_command_metrics_markdown(working_dir, records, task_records),
        key=COMMAND_METRICS_ARTIFACT_KEY,
        description="Rapthor task and external-command timing summary.",
    )
    try:
        chart_path = render_command_profile_chart(working_dir, records, task_records)
    except Exception as err:
        log.warning("Failed to render command profile chart for %s: %s", working_dir, err)
    else:
        if chart_path is not None:
            writers.image(
                image_url=_data_url(chart_path),
                key=COMMAND_PROFILE_SUMMARY_ARTIFACT_KEY,
                description="Rapthor task and external-command CPU, memory, I/O, and timing summary.",
            )
    for flamegraph in _command_flamegraph_records(records):
        writers.image(
            image_url=_data_url(flamegraph["path"]),
            key=flamegraph["artifact_key"],
            description=flamegraph["description"],
        )
    log.info("Published Rapthor command timing artifact from %s", _command_log_path(working_dir))
    return artifact_id


def publish_command_metrics_artifact_for_field(field: object) -> Optional[object]:
    """Publish command timing metrics for a completed Rapthor field, if possible."""
    parset = getattr(field, "parset", {})
    if not isinstance(parset, dict):
        return None

    working_dir = parset.get("dir_working")
    if not working_dir:
        return None
    return publish_command_metrics_artifact(Path(working_dir))


def render_fits_png(
    fits_path: Path,
    output_dir: Path,
    *,
    root_dir: Optional[Path] = None,
    clip_percentile: float = 99.9,
) -> Optional[Path]:
    """Render a FITS image to a PNG preview and return the PNG path."""
    fits_path = Path(fits_path)
    if not _is_fits_artifact(fits_path) or not fits_path.is_file():
        return None

    data = _load_fits_image_data(fits_path)
    if data is None:
        return None

    import matplotlib

    if matplotlib.get_backend().lower() != "agg":
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root_dir = Path(root_dir or fits_path.parent)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = _fits_preview_path(fits_path, root_dir, output_dir)

    lower_percentile, upper_percentile = _clip_percentiles(clip_percentile)
    vmin, vmax = _image_limits(
        data,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
    )
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    image = ax.imshow(data, origin="lower", cmap="inferno", vmin=vmin, vmax=vmax)
    ax.set_title(fits_path.name, fontsize=8)
    ax.set_axis_off()
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return png_path


def render_fits_postage_stamp_pngs(
    fits_path: Path,
    source_catalog_path: Path,
    output_dir: Path,
    *,
    root_dir: Optional[Path] = None,
    max_sources: int = 5,
    stamp_size_px: int = 96,
    clip_percentile: float = 99.9,
) -> list[dict]:
    """Render PNG postage stamps around the brightest catalog sources."""
    fits_path = Path(fits_path)
    source_catalog_path = Path(source_catalog_path)
    if (
        max_sources <= 0
        or stamp_size_px <= 0
        or not _is_fits_artifact(fits_path)
        or not fits_path.is_file()
    ):
        return []

    loaded = _load_fits_image_data_and_header(fits_path)
    if loaded is None:
        return []
    data, header = loaded

    sources = _bright_catalog_sources(source_catalog_path, max_sources)
    if not sources:
        return []

    import matplotlib
    import numpy as np
    from astropy.wcs import WCS

    if matplotlib.get_backend().lower() != "agg":
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root_dir = Path(root_dir or fits_path.parent)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    wcs = WCS(header).celestial
    half_size = max(1, stamp_size_px // 2)
    height, width = data.shape
    lower_percentile, upper_percentile = _clip_percentiles(clip_percentile)
    vmin, vmax = _image_limits(
        data,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
    )
    records = []
    for rank, source in enumerate(sources, start=1):
        try:
            x_value, y_value = wcs.world_to_pixel_values(
                source["ra_deg"],
                source["dec_deg"],
            )
        except Exception as err:
            log.debug("Could not project source %s in %s: %s", source, fits_path, err)
            continue
        if not np.isfinite(x_value) or not np.isfinite(y_value):
            continue

        x_center = int(round(float(x_value)))
        y_center = int(round(float(y_value)))
        if x_center < 0 or x_center >= width or y_center < 0 or y_center >= height:
            continue

        x_min = max(0, x_center - half_size)
        x_max = min(width, x_center + half_size + 1)
        y_min = max(0, y_center - half_size)
        y_max = min(height, y_center + half_size + 1)
        stamp = data[y_min:y_max, x_min:x_max]
        if stamp.size == 0:
            continue

        png_path = _fits_postage_stamp_path(fits_path, root_dir, output_dir, rank)
        coordinate_text = _source_coordinate_text(source, header)
        fig, ax = plt.subplots(figsize=(3.4, 3.4), dpi=150)
        image = ax.imshow(stamp, origin="lower", cmap="inferno", vmin=vmin, vmax=vmax)
        ax.scatter([x_center - x_min], [y_center - y_min], marker="+", color="cyan", s=80)
        ax.set_title(
            (f"{_postage_stamp_label(fits_path)} source {rank}\n{coordinate_text}"),
            fontsize=7,
        )
        ax.set_axis_off()
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.savefig(png_path, bbox_inches="tight", pad_inches=0.04)
        plt.close(fig)
        records.append(
            {
                "path": png_path,
                "fits_paths": [str(fits_path)],
                "clip_percentile": float(clip_percentile),
                "display_vmin": vmin,
                "display_vmax": vmax,
                "source_rank": rank,
                "catalog_index": source["catalog_index"],
                "source_ra_deg": source["ra_deg"],
                "source_dec_deg": source["dec_deg"],
                "source_coordinate_text": coordinate_text,
                "source_flux": source["flux"],
                "source_flux_column": source["flux_column"],
            }
        )
    return records


def render_command_profile_chart(
    working_dir: Path,
    records: list[dict],
    task_records: Optional[list[dict]] = None,
) -> Optional[Path]:
    """Render CPU, memory, I/O, and duration summaries for commands and tasks."""
    command_records = [
        record
        for record in records
        if isinstance(record.get("duration_seconds"), (int, float)) or _resource_metrics(record)
    ]
    task_records = [
        record
        for record in (task_records or [])
        if isinstance(record.get("duration_seconds"), (int, float))
    ]
    chart_records = [
        {**record, "_profile_label": _record_label(record), "_profile_type": "command"}
        for record in command_records
    ] + [
        {**record, "_profile_label": _task_record_label(record), "_profile_type": "task"}
        for record in task_records
    ]
    if not chart_records:
        return None

    top_records = sorted(
        chart_records,
        key=lambda record: float(record.get("duration_seconds") or 0.0),
        reverse=True,
    )[:20]
    top_records.reverse()
    labels = [f"{record['_profile_label']} ({record['_profile_type']})" for record in top_records]
    y_positions = list(range(len(top_records)))

    import matplotlib

    if matplotlib.get_backend().lower() != "agg":
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, max(7, len(top_records) * 0.36)), dpi=120)
    plots = [
        (
            axes[0][0],
            [float(record.get("duration_seconds") or 0.0) for record in top_records],
            "Duration (s)",
            "#4c78a8",
        ),
        (
            axes[0][1],
            [_numeric_metric(record, "cpu_percent") for record in top_records],
            "CPU (%)",
            "#f58518",
        ),
        (
            axes[1][0],
            [_numeric_metric(record, "max_rss_kb") / (1024.0 * 1024.0) for record in top_records],
            "Peak RSS (GB)",
            "#54a24b",
        ),
        (
            axes[1][1],
            [
                _numeric_metric(record, "file_system_inputs")
                + _numeric_metric(record, "file_system_outputs")
                for record in top_records
            ],
            "Filesystem I/O count",
            "#e45756",
        ),
    ]

    for ax, values, title, color in plots:
        ax.barh(y_positions, values, color=color)
        ax.set_title(title, fontsize=10)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(labels, fontsize=7)
        ax.grid(axis="x", alpha=0.25)

    fig.suptitle("Rapthor command and task profile summary", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    chart_path = _command_profile_chart_path(Path(working_dir))
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(chart_path, bbox_inches="tight")
    plt.close(fig)
    return chart_path


def _prefect_artifact_writers() -> ArtifactWriters:
    from prefect.artifacts import (
        create_image_artifact,
        create_link_artifact,
        create_markdown_artifact,
    )

    return ArtifactWriters(
        image=create_image_artifact,
        link=create_link_artifact,
        markdown=create_markdown_artifact,
    )


def _plot_files(plots_dir: Path) -> list[Path]:
    if not plots_dir.is_dir():
        return []
    return sorted(path for path in plots_dir.rglob("*") if path.is_file())


def _plot_artifact_key(relative_path: str) -> str:
    return _artifact_key(PLOT_ARTIFACT_KEY_PREFIX, relative_path)


def _fits_preview_artifact_key(relative_path: str) -> str:
    return _artifact_key(FITS_PREVIEW_ARTIFACT_KEY_PREFIX, relative_path)


def _artifact_key(prefix: str, relative_path: str) -> str:
    digest = hashlib.sha1(relative_path.encode("utf-8")).hexdigest()[:8]
    return f"{prefix}-{_slug(relative_path)}-{digest}"


def _data_url(path: Path) -> str:
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


def _local_file_path(path: Path) -> Path:
    resolved_path = path.resolve()
    container_workspace = os.environ.get("RAPTHOR_CONTAINER_WORKSPACE")
    host_workspace = os.environ.get("RAPTHOR_HOST_WORKSPACE")
    if not container_workspace or not host_workspace:
        return resolved_path

    try:
        relative_path = resolved_path.relative_to(Path(container_workspace).resolve())
    except ValueError:
        return resolved_path
    return Path(host_workspace).expanduser().resolve() / relative_path


def _local_file_url(path: Path) -> str:
    """Return a local file URL that Prefect Markdown handles for plot filenames."""
    return _local_file_path(path).as_uri().replace("%5B", "[").replace("%5D", "]")


def _is_image_artifact(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_ARTIFACT_SUFFIXES


def _is_fits_artifact(path: Path) -> bool:
    return path.name.lower().endswith(FITS_ARTIFACT_SUFFIXES)


def _is_json_artifact(path: Path) -> bool:
    return path.suffix.lower() in JSON_ARTIFACT_SUFFIXES


def _fits_preview_path(fits_path: Path, root_dir: Path, preview_dir: Path) -> Path:
    relative_path = _relative_artifact_path(fits_path, root_dir, include_root_name=False)
    return preview_dir / f"{_slug(relative_path, max_length=120)}.png"


def _fits_postage_stamp_path(
    fits_path: Path, root_dir: Path, preview_dir: Path, source_rank: int
) -> Path:
    relative_path = _relative_artifact_path(fits_path, root_dir, include_root_name=False)
    stem = _slug(relative_path, max_length=100)
    return preview_dir / f"{stem}-source-{source_rank:02d}.png"


def _default_postage_stamp_dir(root_dir: Path) -> Path:
    """Return the durable postage-stamp directory for a pipeline operation."""
    root_dir = Path(root_dir)
    if root_dir.parent.name == "pipelines" and root_dir.name:
        return root_dir.parent.parent / "images" / root_dir.name / "postage-stamps"
    return root_dir / ".rapthor-artifacts" / "postage-stamps"


def _postage_stamp_label(fits_path: Path) -> str:
    name = fits_path.name.lower()
    if "-dirty" in name:
        return "dirty"
    if "image-pb" in name:
        return "image-pb"
    if "-image" in name:
        return "image"
    return fits_path.name


def _load_fits_image_data_and_header(fits_path: Path):
    import numpy as np
    from astropy.io import fits

    with fits.open(fits_path, memmap=False) as hdulist:
        for hdu in hdulist:
            data = hdu.data
            if data is None:
                continue
            data = np.asarray(data)
            if data.ndim < 2 or not np.issubdtype(data.dtype, np.number):
                continue
            data = np.squeeze(data)
            while data.ndim > 2:
                data = data[0]
            if data.ndim == 2:
                return data.astype(float, copy=False), hdu.header
    return None


def _load_fits_image_data(fits_path: Path):
    loaded = _load_fits_image_data_and_header(fits_path)
    if loaded is None:
        return None
    data, _header = loaded
    return data


def _image_limits(
    data,
    *,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.5,
) -> tuple[float, float]:
    import numpy as np

    finite = data[np.isfinite(data)]
    if finite.size == 0:
        raise ValueError("FITS image has no finite pixels")

    vmin, vmax = np.nanpercentile(finite, [lower_percentile, upper_percentile])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin = float(np.nanmin(finite))
        vmax = float(np.nanmax(finite))
    if vmin == vmax:
        delta = abs(vmin) * 0.01 or 1.0
        vmin -= delta
        vmax += delta
    return float(vmin), float(vmax)


def _clip_percentiles(clip_percentile: float) -> tuple[float, float]:
    clip_percentile = float(clip_percentile)
    if not 50.0 < clip_percentile <= 100.0:
        raise ValueError("clip_percentile must be > 50 and <= 100")
    return 100.0 - clip_percentile, clip_percentile


def _coordinate_axis(header, prefix: str, default: int) -> int:
    naxis = int(header.get("WCSAXES", header.get("NAXIS", 2)) or 2)
    for axis in range(1, naxis + 1):
        if str(header.get(f"CTYPE{axis}", "")).upper().startswith(prefix):
            return axis
    return default


def _coordinate_value_text(value_deg: float, header, axis: int, *, signed: bool = False) -> str:
    from astropy import units as u

    unit_text = str(header.get(f"CUNIT{axis}", "deg")).strip() or "deg"
    try:
        value = (value_deg * u.deg).to_value(u.Unit(unit_text))
    except Exception:
        unit_text = "deg"
        value = value_deg
    format_spec = "+.6f" if signed else ".6f"
    return f"{value:{format_spec}} {unit_text}"


def _source_coordinate_text(source: Mapping[str, object], header) -> str:
    ra_axis = _coordinate_axis(header, "RA", 1)
    dec_axis = _coordinate_axis(header, "DEC", 2)
    ra_text = _coordinate_value_text(float(source["ra_deg"]), header, ra_axis)
    dec_text = _coordinate_value_text(float(source["dec_deg"]), header, dec_axis, signed=True)
    return f"RA {ra_text}, Dec {dec_text}"


def _catalog_column(columns: Mapping[str, str], names: list[str]) -> Optional[str]:
    for name in names:
        if name.lower() in columns:
            return columns[name.lower()]
    return None


def _bright_catalog_sources(source_catalog_path: Path, max_sources: int) -> list[dict]:
    import numpy as np
    from astropy.table import Table

    if max_sources <= 0 or not source_catalog_path.is_file():
        return []

    table = Table.read(source_catalog_path)
    columns = {name.lower(): name for name in table.colnames}
    ra_column = _catalog_column(columns, ["RA", "RA_deg", "RAJ2000"])
    dec_column = _catalog_column(columns, ["DEC", "DEC_deg", "DEJ2000"])
    flux_column = _catalog_column(columns, ["Total_flux", "Peak_flux", "Isl_Total_flux", "I"])
    if ra_column is None or dec_column is None or flux_column is None:
        return []

    ra_values = np.asarray(table[ra_column], dtype=float)
    dec_values = np.asarray(table[dec_column], dtype=float)
    flux_values = np.asarray(table[flux_column], dtype=float)
    finite = np.isfinite(ra_values) & np.isfinite(dec_values) & np.isfinite(flux_values)
    order = np.argsort(flux_values[finite])[::-1]
    finite_indices = np.flatnonzero(finite)
    selected_indices = finite_indices[order[:max_sources]]
    return [
        {
            "catalog_index": int(index),
            "ra_deg": float(ra_values[index]),
            "dec_deg": float(dec_values[index]),
            "flux": float(flux_values[index]),
            "flux_column": flux_column,
        }
        for index in selected_indices
    ]


def _json_markdown(path: Path, relative_path: str) -> str:
    text = path.read_text(encoding="utf-8")
    try:
        content = json.dumps(json.loads(text), indent=2, sort_keys=True)
    except json.JSONDecodeError:
        content = text
    return f"# `{relative_path}`\n\n```json\n{content}\n```"


def _relative_artifact_path(path: Path, root_dir: Path, include_root_name: bool) -> str:
    try:
        relative_path = path.relative_to(root_dir).as_posix()
    except ValueError:
        relative_path = path.name

    if include_root_name and root_dir.name:
        return f"{root_dir.name}/{relative_path}"
    return relative_path


def _plot_index_markdown(plots_dir: Path, records: list[dict]) -> str:
    lines = [
        "# Rapthor plots",
        "",
        f"Published {len(records)} plot file artifacts from `{plots_dir}`.",
        "",
    ]
    for record in records:
        lines.append(
            f"- `{record['relative_path']}` ({record['artifact_type']} artifact): "
            f"[local file]({record['file_url']})"
        )
    return "\n".join(lines)


def _command_log_path(working_dir: Path) -> Path:
    return working_dir / "logs" / COMMAND_LOG_FILENAME


def _task_log_path(working_dir: Path) -> Path:
    return working_dir / "logs" / TASK_LOG_FILENAME


def _jsonl_records(path: Path, *, record_type: str) -> list[dict]:
    if not path.exists():
        return []

    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            log.warning("Skipping malformed %s metric line in %s", record_type, path)
    return records


def _command_metric_records(working_dir: Path) -> list[dict]:
    return _jsonl_records(_command_log_path(working_dir), record_type="command")


def _task_metric_records(working_dir: Path) -> list[dict]:
    return _jsonl_records(_task_log_path(working_dir), record_type="task")


def _markdown_cell(value: object, *, max_length: int = 120) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\n", " ").replace("|", "\\|")
    if len(text) > max_length:
        text = f"{text[: max_length - 3]}..."
    return text


def _duration_text(value: object) -> str:
    try:
        duration = float(value)
    except (TypeError, ValueError):
        return ""
    if duration < 1.0:
        return f"{duration * 1000:.0f} ms"
    if duration < 60.0:
        return f"{duration:.2f} s"
    return f"{duration / 60.0:.2f} min"


def _resource_metrics(record: Mapping[str, object]) -> Mapping[str, object]:
    profile = record.get("profile")
    if not isinstance(profile, Mapping):
        return {}
    resource_metrics = profile.get("resource_metrics")
    if not isinstance(resource_metrics, Mapping):
        return {}
    return resource_metrics


def _numeric_metric(record: Mapping[str, object], key: str, default: float = 0.0) -> float:
    value = _resource_metrics(record).get(key)
    if not isinstance(value, (int, float)):
        return default
    return float(value)


def _memory_text(value: object) -> str:
    try:
        kb = float(value)
    except (TypeError, ValueError):
        return ""
    if kb < 1024:
        return f"{kb:.0f} KB"
    mb = kb / 1024.0
    if mb < 1024:
        return f"{mb:.1f} MB"
    return f"{mb / 1024.0:.2f} GB"


def _count_text(value: object) -> str:
    if not isinstance(value, (int, float)):
        return ""
    return f"{value:,.0f}"


def _cpu_text(value: object) -> str:
    if not isinstance(value, (int, float)):
        return ""
    return f"{value:.0f}%"


def _command_tokens(record: Mapping[str, object]) -> list[str]:
    command = record.get("command")
    if isinstance(command, list):
        return [str(token) for token in command if str(token)]

    command_string = record.get("command_string")
    if not isinstance(command_string, str) or not command_string.strip():
        return []

    try:
        return shlex.split(command_string)
    except ValueError:
        return command_string.split()


def _command_name(record: Mapping[str, object]) -> str:
    explicit_name = record.get("name")
    if explicit_name:
        return str(explicit_name)

    tokens = _command_tokens(record)
    if not tokens:
        return "command"

    command_name = Path(tokens[0]).name
    if command_name.startswith("python") and len(tokens) > 1:
        if tokens[1] == "-m" and len(tokens) > 2:
            return tokens[2]
        if not tokens[1].startswith("-"):
            return Path(tokens[1]).name

    return command_name


def _record_label(record: Mapping[str, object]) -> str:
    operation = str(record.get("operation") or "operation")
    name = _command_name(record)
    return f"{operation}/{name}"


def _task_record_name(record: Mapping[str, object]) -> str:
    return str(record.get("task_run_name") or record.get("task_name") or "task")


def _task_record_label(record: Mapping[str, object]) -> str:
    operation = str(record.get("operation") or "operation")
    return f"{operation}/{_task_record_name(record)}"


def _top_metric_line(
    label: str,
    records: list[dict],
    value_fn: Callable[[dict], float],
    format_fn: Callable[[object], str],
) -> Optional[str]:
    candidates = [(record, value_fn(record)) for record in records]
    candidates = [(record, value) for record, value in candidates if value > 0]
    if not candidates:
        return None
    record, value = max(candidates, key=lambda item: item[1])
    return f"- {label}: `{_record_label(record)}` at `{format_fn(value)}`"


def _command_bottleneck_lines(records: list[dict]) -> list[str]:
    lines = []
    for line in [
        _top_metric_line(
            "Slowest command",
            records,
            lambda record: float(record.get("duration_seconds") or 0.0),
            _duration_text,
        ),
        _top_metric_line(
            "Highest peak memory",
            records,
            lambda record: _numeric_metric(record, "max_rss_kb"),
            _memory_text,
        ),
        _top_metric_line(
            "Highest CPU utilisation",
            records,
            lambda record: _numeric_metric(record, "cpu_percent"),
            _cpu_text,
        ),
        _top_metric_line(
            "Largest filesystem input count",
            records,
            lambda record: _numeric_metric(record, "file_system_inputs"),
            _count_text,
        ),
        _top_metric_line(
            "Largest filesystem output count",
            records,
            lambda record: _numeric_metric(record, "file_system_outputs"),
            _count_text,
        ),
    ]:
        if line is not None:
            lines.append(line)
    return lines


def _command_flamegraph_records(records: list[dict]) -> list[dict]:
    flamegraphs = []
    for record in records:
        profile = record.get("profile")
        if not isinstance(profile, Mapping):
            continue
        artifacts = profile.get("artifacts")
        if not isinstance(artifacts, Mapping):
            continue
        flamegraph_path = artifacts.get("perf_flamegraph")
        if not flamegraph_path:
            continue
        path = Path(str(flamegraph_path))
        if not path.is_file():
            continue

        label = _record_label(record)
        flamegraphs.append(
            {
                "artifact_key": _artifact_key(
                    COMMAND_FLAMEGRAPH_ARTIFACT_KEY_PREFIX,
                    f"{label}/{path.name}",
                ),
                "description": f"Rapthor perf flamegraph for {label}.",
                "label": label,
                "path": path,
            }
        )
    return flamegraphs


def _command_profile_chart_path(working_dir: Path) -> Path:
    return working_dir / "logs" / "command-profile-summary.png"


def _tag_text(value: object) -> str:
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(item) for item in value)
    return "" if value is None else str(value)


def _command_metrics_markdown(
    working_dir: Path,
    records: list[dict],
    task_records: Optional[list[dict]] = None,
) -> str:
    log_path = _command_log_path(working_dir)
    task_log = _task_log_path(working_dir)
    durations = [
        float(record["duration_seconds"])
        for record in records
        if isinstance(record.get("duration_seconds"), (int, float))
    ]
    task_records = task_records or []
    task_durations = [
        float(record["duration_seconds"])
        for record in task_records
        if isinstance(record.get("duration_seconds"), (int, float))
    ]
    lines = [
        "# Rapthor command timings",
        "",
        f"Source: `{log_path}`",
        f"Task source: `{task_log}`",
        "",
    ]
    if durations:
        lines.extend(
            [
                f"Total recorded external-command time: `{_duration_text(sum(durations))}`",
                "",
            ]
        )
    if task_durations:
        lines.extend(
            [
                f"Total recorded Prefect task time: `{_duration_text(sum(task_durations))}`",
                "",
            ]
        )
    bottleneck_lines = _command_bottleneck_lines(records)
    if bottleneck_lines:
        lines.extend(["## Bottleneck summary", "", *bottleneck_lines, ""])

    flamegraph_records = _command_flamegraph_records(records)
    if flamegraph_records:
        lines.extend(
            [
                "## Perf flamegraphs",
                "",
                "Generated from Linux `perf record` samples for commands run with "
                "`prefect_command_profile = perf`.",
                "",
            ]
        )
        for record in flamegraph_records:
            lines.append(f"- `{record['label']}`: [local SVG]({_local_file_url(record['path'])})")
        lines.append("")

    if task_records:
        lines.extend(
            [
                "## Prefect task runtimes",
                "",
                "| Operation | Task run | Task | Tags | Status | Duration |",
                "| --- | --- | --- | --- | --- | ---: |",
            ]
        )
        for record in task_records:
            lines.append(
                "| "
                f"{_markdown_cell(record.get('operation'))} | "
                f"{_markdown_cell(record.get('task_run_name'))} | "
                f"{_markdown_cell(record.get('task_name'))} | "
                f"{_markdown_cell(_tag_text(record.get('task_tags')))} | "
                f"{_markdown_cell(record.get('status'))} | "
                f"{_markdown_cell(_duration_text(record.get('duration_seconds')))} |"
            )
        lines.append("")

    lines.extend(
        [
            "## External command runtimes",
            "",
            "| Operation | Task run | Name | Status | Duration | CPU | Max RSS | FS In | FS Out | Command |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for record in records:
        command = record.get("command_string")
        if command is None and isinstance(record.get("command"), list):
            command = " ".join(str(token) for token in record["command"])
        metrics = _resource_metrics(record)
        lines.append(
            "| "
            f"{_markdown_cell(record.get('operation'))} | "
            f"{_markdown_cell(record.get('task_run_name'))} | "
            f"{_markdown_cell(_command_name(record))} | "
            f"{_markdown_cell(record.get('status'))} | "
            f"{_markdown_cell(_duration_text(record.get('duration_seconds')))} | "
            f"{_markdown_cell(_cpu_text(metrics.get('cpu_percent')))} | "
            f"{_markdown_cell(_memory_text(metrics.get('max_rss_kb')))} | "
            f"{_markdown_cell(_count_text(metrics.get('file_system_inputs')))} | "
            f"{_markdown_cell(_count_text(metrics.get('file_system_outputs')))} | "
            f"`{_markdown_cell(command)}` |"
        )
    return "\n".join(lines)


def _file_record_paths(file_records: list[Mapping[str, object]]) -> list[Path]:
    return [
        Path(str(record["path"]))
        for record in file_records
        if record.get("class") == "File" and record.get("path")
    ]


def _bool_setting_is_enabled(value: object, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1", "on"}
    return bool(value)


def _slug(value: str, max_length: int = 80) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    if not slug:
        return "file"
    return slug[:max_length].strip("-") or "file"
