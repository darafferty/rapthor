"""Prefect artifact publication helpers for Rapthor outputs."""

import base64
import hashlib
import json
import logging
import mimetypes
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional

log = logging.getLogger("rapthor")

IMAGE_ARTIFACT_SUFFIXES = {".gif", ".jpeg", ".jpg", ".png", ".svg", ".webp"}
FITS_ARTIFACT_SUFFIXES = (".fit", ".fits", ".fit.fz", ".fits.fz")
FITS_PREVIEW_ARTIFACT_KEY_PREFIX = "rapthor-fits-preview"
JSON_ARTIFACT_SUFFIXES = {".json"}
PLOT_ARTIFACT_KEY_PREFIX = "rapthor-plot"
COMMAND_METRICS_ARTIFACT_KEY = "rapthor-command-metrics"
COMMAND_LOG_FILENAME = "commands.jsonl"


@dataclass(frozen=True)
class ArtifactWriters:
    """Callables used to create Prefect artifacts."""

    image: Callable[..., object]
    link: Callable[..., object]
    markdown: Callable[..., object]


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


def _in_prefect_run_context() -> bool:
    try:
        from prefect.context import get_run_context

        get_run_context()
    except Exception:
        return False
    return True


def _plot_files(plots_dir: Path) -> list[Path]:
    if not plots_dir.is_dir():
        return []
    return sorted(path for path in plots_dir.rglob("*") if path.is_file())


def _slug(value: str, max_length: int = 80) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    if not slug:
        return "file"
    return slug[:max_length].strip("-") or "file"


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


def _is_image_artifact(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_ARTIFACT_SUFFIXES


def _is_fits_artifact(path: Path) -> bool:
    return path.name.lower().endswith(FITS_ARTIFACT_SUFFIXES)


def _is_json_artifact(path: Path) -> bool:
    return path.suffix.lower() in JSON_ARTIFACT_SUFFIXES


def _fits_preview_path(fits_path: Path, root_dir: Path, preview_dir: Path) -> Path:
    relative_path = _relative_artifact_path(fits_path, root_dir, include_root_name=False)
    return preview_dir / f"{_slug(relative_path, max_length=120)}.png"


def _load_fits_image_data(fits_path: Path):
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
                return data.astype(float, copy=False)
    return None


def _image_limits(data) -> tuple[float, float]:
    import numpy as np

    finite = data[np.isfinite(data)]
    if finite.size == 0:
        raise ValueError("FITS image has no finite pixels")

    vmin, vmax = np.nanpercentile(finite, [1.0, 99.5])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin = float(np.nanmin(finite))
        vmax = float(np.nanmax(finite))
    if vmin == vmax:
        delta = abs(vmin) * 0.01 or 1.0
        vmin -= delta
        vmax += delta
    return float(vmin), float(vmax)


def render_fits_png(
    fits_path: Path,
    output_dir: Path,
    *,
    root_dir: Optional[Path] = None,
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

    vmin, vmax = _image_limits(data)
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    image = ax.imshow(data, origin="lower", cmap="inferno", vmin=vmin, vmax=vmax)
    ax.set_title(fits_path.name, fontsize=8)
    ax.set_axis_off()
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return png_path


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


def _command_metric_records(working_dir: Path) -> list[dict]:
    log_path = _command_log_path(working_dir)
    if not log_path.exists():
        return []

    records = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            log.warning("Skipping malformed command metric line in %s", log_path)
    return records


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


def _command_metrics_markdown(working_dir: Path, records: list[dict]) -> str:
    log_path = _command_log_path(working_dir)
    durations = [
        float(record["duration_seconds"])
        for record in records
        if isinstance(record.get("duration_seconds"), (int, float))
    ]
    lines = [
        "# Rapthor command timings",
        "",
        f"Source: `{log_path}`",
        "",
    ]
    if durations:
        lines.extend(
            [
                f"Total recorded external-command time: `{_duration_text(sum(durations))}`",
                "",
            ]
        )
    lines.extend(
        [
            "| Operation | Name | Status | Duration | Command |",
            "| --- | --- | --- | ---: | --- |",
        ]
    )
    for record in records:
        command = record.get("command_string")
        if command is None and isinstance(record.get("command"), list):
            command = " ".join(str(token) for token in record["command"])
        lines.append(
            "| "
            f"{_markdown_cell(record.get('operation'))} | "
            f"{_markdown_cell(record.get('name'))} | "
            f"{_markdown_cell(record.get('status'))} | "
            f"{_markdown_cell(_duration_text(record.get('duration_seconds')))} | "
            f"`{_markdown_cell(command)}` |"
        )
    return "\n".join(lines)


def publish_file_artifacts(
    file_paths: list[Path],
    root_dir: Path,
    *,
    include_root_name: bool = False,
    artifact_writers: Optional[ArtifactWriters] = None,
    in_run_context: Callable[[], bool] = _in_prefect_run_context,
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
        file_url = plot_file.resolve().as_uri()

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


def _file_record_paths(file_records: list[Mapping[str, object]]) -> list[Path]:
    return [
        Path(str(record["path"]))
        for record in file_records
        if record.get("class") == "File" and record.get("path")
    ]


def publish_fits_image_artifacts(
    file_records: list[Mapping[str, object]],
    root_dir: Path,
    *,
    artifact_writers: Optional[ArtifactWriters] = None,
    in_run_context: Callable[[], bool] = _in_prefect_run_context,
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
            png_path = render_fits_png(fits_path, preview_dir, root_dir=root_dir)
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
            }
        )

    if records:
        log.info("Published %d Rapthor FITS preview artifacts from %s", len(records), root_dir)
    return records


def publish_plot_artifacts(
    plots_dir: Path,
    artifact_writers: Optional[ArtifactWriters] = None,
    in_run_context: Callable[[], bool] = _in_prefect_run_context,
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
    in_run_context: Callable[[], bool] = _in_prefect_run_context,
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
    return publish_fits_image_artifacts(records, images_dir)


def publish_command_metrics_artifact(
    working_dir: Path,
    *,
    artifact_writers: Optional[ArtifactWriters] = None,
    in_run_context: Callable[[], bool] = _in_prefect_run_context,
) -> Optional[object]:
    """Publish a Markdown summary of external-command timings, if available."""
    working_dir = Path(working_dir)
    records = _command_metric_records(working_dir)
    if not records:
        return None
    if not in_run_context():
        return None

    writers = artifact_writers or _prefect_artifact_writers()
    artifact_id = writers.markdown(
        markdown=_command_metrics_markdown(working_dir, records),
        key=COMMAND_METRICS_ARTIFACT_KEY,
        description="Rapthor external-command timing summary.",
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
