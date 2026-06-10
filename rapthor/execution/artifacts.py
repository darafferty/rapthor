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
JSON_ARTIFACT_SUFFIXES = {".json"}
PLOT_ARTIFACT_KEY_PREFIX = "rapthor-plot"


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
    digest = hashlib.sha1(relative_path.encode("utf-8")).hexdigest()[:8]
    return f"{PLOT_ARTIFACT_KEY_PREFIX}-{_slug(relative_path)}-{digest}"


def _data_url(path: Path) -> str:
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


def _is_image_artifact(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_ARTIFACT_SUFFIXES


def _is_json_artifact(path: Path) -> bool:
    return path.suffix.lower() in JSON_ARTIFACT_SUFFIXES


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
    paths = [
        Path(str(record["path"]))
        for record in file_records
        if record.get("class") == "File" and record.get("path")
    ]
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
