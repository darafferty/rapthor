"""Shell command wrappers used by Prefect tasks."""

import hashlib
import html
import json
import logging
import os
import queue
import re
import resource
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Mapping, Optional

from rapthor.execution.commands import CommandInput, command_to_string, normalize_command
from rapthor.execution.config import ExecutionConfig


class MissingPrefectShellError(RuntimeError):
    """Raised when prefect-shell is required but not installed."""


class ShellCommandError(RuntimeError):
    """Raised when an external command exits unsuccessfully."""

    def __init__(self, message: str, returncode: int):
        super().__init__(message)
        self.message = message
        self.returncode = returncode

    def __reduce__(self):
        return (type(self), (self.message, self.returncode))


log = logging.getLogger("rapthor:shell")
STREAM_LOG_FLUSH_INTERVAL_SECONDS = 1.0
STREAM_LOG_MAX_LINES_PER_RECORD = 40
PERF_PROFILE_MODE = "perf"
TIME_PROFILE_MODES = {"auto", "time", PERF_PROFILE_MODE}
_STREAM_DONE = object()
_FLAMEGRAPH_WIDTH = 1200
_FLAMEGRAPH_FRAME_HEIGHT = 18
_FLAMEGRAPH_HEADER_HEIGHT = 48
_FLAMEGRAPH_MARGIN = 8

_TIME_METRIC_FIELDS = {
    "User time (seconds)": ("user_seconds", float),
    "System time (seconds)": ("system_seconds", float),
    "Percent of CPU this job got": ("cpu_percent", lambda value: float(value.rstrip("%"))),
    "Elapsed (wall clock) time (h:mm:ss or m:ss)": ("elapsed_seconds", "elapsed"),
    "Maximum resident set size (kbytes)": ("max_rss_kb", int),
    "Average resident set size (kbytes)": ("average_rss_kb", int),
    "Major (requiring I/O) page faults": ("major_page_faults", int),
    "Minor (reclaiming a frame) page faults": ("minor_page_faults", int),
    "File system inputs": ("file_system_inputs", int),
    "File system outputs": ("file_system_outputs", int),
    "Voluntary context switches": ("voluntary_context_switches", int),
    "Involuntary context switches": ("involuntary_context_switches", int),
    "Exit status": ("exit_status", int),
}


@dataclass(frozen=True)
class ShellCommand:
    """A shell command plus execution metadata."""

    command: CommandInput
    environment: Mapping[str, str] = field(default_factory=dict)
    working_directory: Optional[str] = None
    name: Optional[str] = None

    @property
    def command_string(self) -> str:
        return command_to_string(self.command)


def _load_shell_operation_cls():
    try:
        from prefect_shell import ShellOperation
    except ImportError as err:
        raise MissingPrefectShellError("prefect-shell is required to execute shell tasks") from err
    return ShellOperation


def _is_prefect_shell_operation_cls(operation_cls) -> bool:
    return (
        getattr(operation_cls, "__module__", None) == "prefect_shell.commands"
        and getattr(operation_cls, "__name__", None) == "ShellOperation"
    )


def _get_command_logger():
    try:
        from prefect.logging import get_run_logger

        return get_run_logger()
    except Exception:
        return log


def _slug(value: str, max_length: int = 80) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    if not slug:
        return "command"
    return slug[:max_length].strip("-") or "command"


@lru_cache
def _gnu_time_path() -> Optional[str]:
    candidates = ["/usr/bin/time", shutil.which("time")]
    for candidate in dict.fromkeys(path for path in candidates if path):
        try:
            result = subprocess.run(
                [candidate, "-v", "true"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except OSError:
            continue
        if result.returncode == 0:
            return candidate
    return None


@lru_cache
def _perf_path() -> Optional[str]:
    return shutil.which("perf")


@lru_cache
def _perf_record_support() -> tuple[bool, str]:
    perf_path = _perf_path()
    if perf_path is None:
        return False, "perf is not available"

    try:
        with tempfile.TemporaryDirectory(prefix="rapthor-perf-check-") as temp_dir:
            result = subprocess.run(
                [perf_path, "record", "-o", str(Path(temp_dir) / "perf.data"), "--", "true"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
    except OSError as err:
        return False, str(err)

    if result.returncode != 0:
        reason = result.stderr.strip() or f"perf record exited with {result.returncode}"
        return False, reason
    return True, ""


def _parse_elapsed_seconds(value: str) -> float:
    parts = value.strip().split(":")
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    return float(value)


def parse_gnu_time_metrics(path: Path) -> dict:
    """Parse selected metrics from GNU ``time -v`` output."""
    metrics = {}
    if not path.exists():
        return metrics

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = re.match(r"\s*(?P<key>.*?):\s+(?P<value>.*)$", line)
        if match is None:
            continue
        key = match.group("key").strip()
        value = match.group("value").strip()
        if key not in _TIME_METRIC_FIELDS:
            continue

        metric_name, parser = _TIME_METRIC_FIELDS[key]
        try:
            if parser == "elapsed":
                metrics[metric_name] = _parse_elapsed_seconds(value)
            else:
                metrics[metric_name] = parser(value)
        except ValueError:
            log.debug("Could not parse GNU time metric %s=%r from %s", key, value, path)
    return metrics


def _perf_frame_name(line: str) -> Optional[str]:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None

    parts = stripped.split(None, 1)
    if len(parts) == 2 and re.fullmatch(r"(?:0x)?[0-9a-fA-F]+|\?+", parts[0]):
        frame = parts[1]
    else:
        frame = stripped

    frame = re.sub(r"\s+\([^)]*\)\s*$", "", frame).strip()
    frame = re.sub(r"\+0x[0-9a-fA-F]+(?:/[0-9a-fA-F]+)?$", "", frame).strip()
    return frame.replace(";", ":") or None


def collapse_perf_script(script_text: str) -> dict[tuple[str, ...], int]:
    """Collapse ``perf script`` call stacks into flamegraph-style stack counts."""
    collapsed: dict[tuple[str, ...], int] = {}
    sample_frames: list[str] = []

    def flush_sample() -> None:
        if not sample_frames:
            return
        stack = tuple(reversed(sample_frames))
        collapsed[stack] = collapsed.get(stack, 0) + 1
        sample_frames.clear()

    for line in script_text.splitlines():
        if not line.strip():
            flush_sample()
            continue
        if not line.startswith((" ", "\t")):
            flush_sample()
            continue

        frame = _perf_frame_name(line)
        if frame:
            sample_frames.append(frame)

    flush_sample()
    return collapsed


def _flamegraph_color(name: str) -> str:
    digest = hashlib.sha1(name.encode("utf-8")).digest()
    red = 200 + digest[0] % 46
    green = 80 + digest[1] % 120
    blue = 35 + digest[2] % 45
    return f"rgb({red},{green},{blue})"


def _flamegraph_depth(node: dict) -> int:
    children = node["children"]
    if not children:
        return 0
    return 1 + max(_flamegraph_depth(child) for child in children.values())


def _shorten_svg_label(label: str, width: float) -> str:
    max_chars = max(0, int((width - 6) / 7))
    if max_chars <= 2:
        return ""
    if len(label) <= max_chars:
        return label
    return f"{label[: max_chars - 1]}..."


def render_perf_flamegraph_svg(
    collapsed_stacks: Mapping[tuple[str, ...], int],
    *,
    title: str = "Rapthor perf flamegraph",
) -> str:
    """Render collapsed perf stacks as a self-contained SVG flamegraph."""
    root = {"name": "all", "count": 0, "children": {}}
    for stack, count in collapsed_stacks.items():
        if not stack or count <= 0:
            continue
        root["count"] += count
        node = root
        for frame in stack:
            children = node["children"]
            if frame not in children:
                children[frame] = {"name": frame, "count": 0, "children": {}}
            node = children[frame]
            node["count"] += count

    total_samples = root["count"]
    if total_samples <= 0:
        return ""

    max_depth = _flamegraph_depth(root)
    height = (
        _FLAMEGRAPH_HEADER_HEIGHT
        + (max_depth + 1) * _FLAMEGRAPH_FRAME_HEIGHT
        + _FLAMEGRAPH_MARGIN * 2
    )
    inner_width = _FLAMEGRAPH_WIDTH - _FLAMEGRAPH_MARGIN * 2
    elements = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{_FLAMEGRAPH_WIDTH}" '
            f'height="{height}" viewBox="0 0 {_FLAMEGRAPH_WIDTH} {height}">'
        ),
        "<style>"
        "text{font-family:Arial,sans-serif;font-size:12px;fill:#111}"
        ".frame rect{stroke:#fff;stroke-width:.5;rx:2;ry:2}"
        ".subtitle{font-size:11px;fill:#555}"
        "</style>",
        f"<title>{html.escape(title)}</title>",
        f'<text x="{_FLAMEGRAPH_MARGIN}" y="20">{html.escape(title)}</text>',
        (
            f'<text class="subtitle" x="{_FLAMEGRAPH_MARGIN}" y="38">'
            f"{total_samples:,} perf samples; wider frames consumed more sampled CPU time"
            "</text>"
        ),
    ]

    def render_node(node: dict, x: float, width: float, depth: int) -> None:
        if width < 0.25:
            return
        y = _FLAMEGRAPH_HEADER_HEIGHT + (max_depth - depth) * _FLAMEGRAPH_FRAME_HEIGHT
        escaped_name = html.escape(str(node["name"]))
        sample_count = int(node["count"])
        percent = sample_count / total_samples * 100.0
        color = _flamegraph_color(str(node["name"]))
        label = html.escape(_shorten_svg_label(str(node["name"]), width))
        elements.extend(
            [
                '<g class="frame">',
                (f"<title>{escaped_name} - {sample_count:,} samples ({percent:.1f}%)</title>"),
                (
                    f'<rect x="{x:.3f}" y="{y:.3f}" width="{width:.3f}" '
                    f'height="{_FLAMEGRAPH_FRAME_HEIGHT - 1}" fill="{color}" />'
                ),
            ]
        )
        if label:
            elements.append(f'<text x="{x + 3:.3f}" y="{y + 12.5:.3f}">{label}</text>')
        elements.append("</g>")

        child_x = x
        children = sorted(
            node["children"].values(),
            key=lambda child: (-int(child["count"]), str(child["name"])),
        )
        for child in children:
            child_width = width * int(child["count"]) / max(1, int(node["count"]))
            render_node(child, child_x, child_width, depth + 1)
            child_x += child_width

    render_node(root, _FLAMEGRAPH_MARGIN, inner_width, 0)
    elements.append("</svg>")
    return "\n".join(elements)


def write_perf_flamegraph_files(
    perf_script_path: Path,
    *,
    title: Optional[str] = None,
) -> dict[str, object]:
    """Write folded stacks and an SVG flamegraph beside a ``perf.script`` file."""
    perf_script_path = Path(perf_script_path)
    collapsed_stacks = collapse_perf_script(
        perf_script_path.read_text(encoding="utf-8", errors="replace")
    )
    if not collapsed_stacks:
        return {"status": "no_samples"}

    folded_path = perf_script_path.with_suffix(".folded")
    flamegraph_path = perf_script_path.with_suffix(".flamegraph.svg")
    folded_lines = [
        f"{';'.join(stack)} {count}"
        for stack, count in sorted(collapsed_stacks.items(), key=lambda item: item[0])
    ]
    folded_path.write_text("\n".join(folded_lines) + "\n", encoding="utf-8")
    flamegraph_path.write_text(
        render_perf_flamegraph_svg(
            collapsed_stacks,
            title=title or f"Rapthor perf flamegraph: {perf_script_path.parent.name}",
        ),
        encoding="utf-8",
    )
    return {
        "status": "created",
        "perf_folded": str(folded_path),
        "perf_flamegraph": str(flamegraph_path),
        "samples": sum(collapsed_stacks.values()),
        "stacks": len(collapsed_stacks),
    }


def _profile_directory(shell_command: ShellCommand, started_at: datetime) -> Optional[Path]:
    log_path = command_log_path(shell_command.working_directory)
    if log_path is None:
        return None

    operation = Path(str(shell_command.working_directory)).name
    name = shell_command.name or normalize_command(shell_command.command)[0]
    timestamp = started_at.strftime("%Y%m%dT%H%M%S%fZ")
    digest = hashlib.sha1(shell_command.command_string.encode("utf-8")).hexdigest()[:8]
    return (
        log_path.parent
        / "profiles"
        / (f"{_slug(operation)}-{_slug(str(name))}-{timestamp}-{digest}")
    )


def _profiled_process_args(
    base_args: list[str],
    shell_command: ShellCommand,
    execution_config: ExecutionConfig,
    started_at: datetime,
    profile: dict,
) -> list[str]:
    if not execution_config.log_commands or execution_config.command_profile == "off":
        return base_args
    if execution_config.command_profile not in TIME_PROFILE_MODES:
        return base_args

    profile_dir = _profile_directory(shell_command, started_at)
    if profile_dir is None:
        return base_args

    profile_dir.mkdir(parents=True, exist_ok=True)
    profile.update(
        {
            "mode": execution_config.command_profile,
            "status": "resource",
            "resource_source": "python_resource",
        }
    )

    time_path = _gnu_time_path()
    if time_path is None:
        if execution_config.command_profile != "auto":
            profile["reason"] = "GNU time -v is not available"
        return base_args

    time_output_path = profile_dir / "time.txt"
    profile.update(
        {
            "mode": execution_config.command_profile,
            "status": "time",
            "resource_source": "gnu_time",
            "artifacts": {"gnu_time": str(time_output_path)},
        }
    )
    profiled_args = [time_path, "-v", "-o", str(time_output_path), *base_args]

    if execution_config.command_profile == PERF_PROFILE_MODE:
        perf_supported, perf_reason = _perf_record_support()
        if not perf_supported:
            profile["perf_status"] = "unavailable"
            profile["perf_reason"] = perf_reason
            return profiled_args

        perf_data_path = profile_dir / "perf.data"
        profile["status"] = "perf"
        profile["artifacts"]["perf_data"] = str(perf_data_path)
        return [
            str(_perf_path()),
            "record",
            "-F",
            "99",
            "-g",
            "-o",
            str(perf_data_path),
            "--",
            *profiled_args,
        ]

    return profiled_args


def _resource_usage_snapshot():
    return resource.getrusage(resource.RUSAGE_CHILDREN)


def _resource_usage_metrics(before, after, elapsed_seconds: float) -> dict:
    if before is None or after is None:
        return {}

    user_seconds = max(0.0, after.ru_utime - before.ru_utime)
    system_seconds = max(0.0, after.ru_stime - before.ru_stime)
    cpu_seconds = user_seconds + system_seconds
    metrics = {
        "user_seconds": user_seconds,
        "system_seconds": system_seconds,
        "elapsed_seconds": max(0.0, elapsed_seconds),
        "max_rss_kb": max(0, after.ru_maxrss),
        "major_page_faults": max(0, after.ru_majflt - before.ru_majflt),
        "minor_page_faults": max(0, after.ru_minflt - before.ru_minflt),
        "file_system_inputs": max(0, after.ru_inblock - before.ru_inblock),
        "file_system_outputs": max(0, after.ru_oublock - before.ru_oublock),
        "voluntary_context_switches": max(0, after.ru_nvcsw - before.ru_nvcsw),
        "involuntary_context_switches": max(0, after.ru_nivcsw - before.ru_nivcsw),
    }
    if elapsed_seconds > 0:
        metrics["cpu_percent"] = cpu_seconds / elapsed_seconds * 100.0
    return metrics


def _write_perf_script(profile: dict) -> None:
    artifacts = profile.get("artifacts")
    if not isinstance(artifacts, dict):
        return
    perf_data = artifacts.get("perf_data")
    perf_path = _perf_path()
    if not perf_data or perf_path is None or not Path(perf_data).exists():
        return

    perf_script_path = Path(perf_data).with_suffix(".script")
    try:
        result = subprocess.run(
            [perf_path, "script", "-i", perf_data],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except OSError as err:
        profile["perf_script_status"] = "failed"
        profile["perf_script_error"] = str(err)
        return

    if result.returncode != 0:
        profile["perf_script_status"] = "failed"
        profile["perf_script_error"] = result.stderr.strip()
        return

    perf_script_path.write_text(result.stdout, encoding="utf-8")
    artifacts["perf_script"] = str(perf_script_path)
    profile["perf_script_status"] = "created"


def _write_perf_flamegraph(profile: dict) -> None:
    artifacts = profile.get("artifacts")
    if not isinstance(artifacts, dict):
        return
    perf_script_path = artifacts.get("perf_script")
    if not perf_script_path or not Path(perf_script_path).exists():
        return

    try:
        result = write_perf_flamegraph_files(Path(perf_script_path))
    except OSError as err:
        profile["flamegraph_status"] = "failed"
        profile["flamegraph_error"] = str(err)
        return

    profile["flamegraph_status"] = result["status"]
    if result["status"] != "created":
        return

    artifacts["perf_folded"] = result["perf_folded"]
    artifacts["perf_flamegraph"] = result["perf_flamegraph"]
    profile["flamegraph_samples"] = result["samples"]
    profile["flamegraph_stacks"] = result["stacks"]


def _finish_command_profile(
    profile: dict,
    resource_before=None,
    resource_after=None,
    elapsed_seconds: float = 0.0,
) -> None:
    artifacts = profile.get("artifacts")
    if isinstance(artifacts, dict) and artifacts.get("gnu_time"):
        metrics = parse_gnu_time_metrics(Path(artifacts["gnu_time"]))
        if metrics:
            profile["resource_metrics"] = metrics
    if "resource_metrics" not in profile:
        metrics = _resource_usage_metrics(resource_before, resource_after, elapsed_seconds)
        if metrics:
            profile["resource_metrics"] = metrics
            profile["resource_source"] = "python_resource"
    if profile.get("status") == "perf":
        _write_perf_script(profile)
        _write_perf_flamegraph(profile)


def _queue_process_output(output, output_queue) -> None:
    try:
        for raw_line in iter(output.readline, b""):
            output_queue.put(raw_line)
    finally:
        output_queue.put(_STREAM_DONE)


def _output_lines(raw_line: bytes) -> list[str]:
    text = raw_line.decode(errors="replace").rstrip("\r\n")
    return text.splitlines() or [text]


def _flush_stream_log(command_logger, buffered_lines: list[str]) -> None:
    if not buffered_lines:
        return
    message = "\n".join(buffered_lines).rstrip("\n")
    buffered_lines.clear()
    if message:
        command_logger.info(message)


def shell_operation_kwargs(
    shell_command: ShellCommand,
    execution_config: ExecutionConfig,
) -> dict:
    """Build keyword arguments for `prefect_shell.ShellOperation`."""
    kwargs = {
        "commands": [shell_command.command_string],
        "stream_output": execution_config.stream_output,
    }
    if shell_command.environment:
        kwargs["env"] = dict(shell_command.environment)
    if shell_command.working_directory is not None:
        kwargs["working_dir"] = shell_command.working_directory
    return kwargs


def command_log_path(working_directory: Optional[str]) -> Optional[Path]:
    """Return the backend-neutral command log path for an operation workdir."""
    if working_directory is None:
        return None
    workdir = Path(working_directory)
    if workdir.parent.name != "pipelines":
        return None
    return workdir.parent.parent / "logs" / "commands.jsonl"


def write_command_log_record(
    shell_command: ShellCommand,
    execution_config: ExecutionConfig,
    *,
    started_at: Optional[datetime] = None,
    finished_at: Optional[datetime] = None,
    duration_seconds: Optional[float] = None,
    status: Optional[str] = None,
    returncode: Optional[int] = None,
    error: Optional[str] = None,
    profile: Optional[Mapping[str, object]] = None,
) -> Optional[Path]:
    """Append a structured command record for integration assertions."""
    if not execution_config.log_commands:
        return None

    log_path = command_log_path(shell_command.working_directory)
    if log_path is None:
        return None

    cwd = None if shell_command.working_directory is None else str(shell_command.working_directory)
    record = {
        "backend": "prefect",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "operation": Path(cwd).name if cwd is not None else None,
        "name": shell_command.name,
        "cwd": cwd,
        "command": normalize_command(shell_command.command),
        "command_string": shell_command.command_string,
        "environment": dict(shell_command.environment),
    }
    if started_at is not None:
        record["started_at"] = started_at.isoformat()
    if finished_at is not None:
        record["finished_at"] = finished_at.isoformat()
    if duration_seconds is not None:
        record["duration_seconds"] = round(duration_seconds, 6)
    if status is not None:
        record["status"] = status
    if returncode is not None:
        record["returncode"] = returncode
    if error is not None:
        record["error"] = error
    if profile:
        record["profile"] = dict(profile)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    return log_path


def _run_streaming_shell_command(
    shell_command: ShellCommand,
    execution_config: ExecutionConfig,
    started_at: datetime,
    profile: dict,
) -> list[str]:
    command_logger = _get_command_logger()
    buffered_log_lines = []
    output_lines = []
    process = None
    reader_thread = None
    temp_file = None
    resource_before = None
    process_start_time = None
    try:
        temp_file = tempfile.NamedTemporaryFile(
            prefix="rapthor-prefect-",
            suffix=".sh",
            delete=False,
        )
        temp_file.write(shell_command.command_string.encode())
        temp_file.close()

        env = os.environ.copy()
        env.update(shell_command.environment)
        process_args = _profiled_process_args(
            ["bash", temp_file.name],
            shell_command,
            execution_config,
            started_at,
            profile,
        )
        if profile:
            resource_before = _resource_usage_snapshot()
            process_start_time = time.monotonic()
        process = subprocess.Popen(
            process_args,
            cwd=shell_command.working_directory,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if process.stdout is None:
            raise RuntimeError("Command output pipe was not created")

        output_queue = queue.Queue()
        reader_thread = threading.Thread(
            target=_queue_process_output,
            args=(process.stdout, output_queue),
            daemon=True,
        )
        reader_thread.start()

        while True:
            try:
                raw_line = output_queue.get(timeout=STREAM_LOG_FLUSH_INTERVAL_SECONDS)
            except queue.Empty:
                _flush_stream_log(command_logger, buffered_log_lines)
                continue

            if raw_line is _STREAM_DONE:
                break

            lines = _output_lines(raw_line)
            output_lines.extend(lines)
            buffered_log_lines.extend(lines)
            if len(buffered_log_lines) >= STREAM_LOG_MAX_LINES_PER_RECORD:
                _flush_stream_log(command_logger, buffered_log_lines)

        _flush_stream_log(command_logger, buffered_log_lines)

        process.wait()
        resource_after = _resource_usage_snapshot() if resource_before is not None else None
        profile_elapsed_seconds = (
            time.monotonic() - process_start_time if process_start_time is not None else 0.0
        )
        _finish_command_profile(
            profile,
            resource_before=resource_before,
            resource_after=resource_after,
            elapsed_seconds=profile_elapsed_seconds,
        )
        if process.returncode != 0:
            raise ShellCommandError(
                "Command failed with return code "
                f"{process.returncode}: {shell_command.command_string}",
                process.returncode,
            )
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        if reader_thread is not None:
            reader_thread.join(timeout=1)
        if temp_file is not None and os.path.exists(temp_file.name):
            os.remove(temp_file.name)

    return output_lines


def run_shell_command(
    shell_command: ShellCommand,
    execution_config: ExecutionConfig,
    shell_operation_cls=None,
):
    """Execute one command using the configured Prefect shell runner."""
    started_at = datetime.now(timezone.utc)
    start_time = time.monotonic()
    status = "completed"
    returncode = 0
    error = None
    profile = {}
    try:
        operation_cls = shell_operation_cls or _load_shell_operation_cls()
        if (
            shell_operation_cls is None
            and execution_config.stream_output
            and _is_prefect_shell_operation_cls(operation_cls)
        ):
            return _run_streaming_shell_command(
                shell_command,
                execution_config,
                started_at,
                profile,
            )

        operation = operation_cls(**shell_operation_kwargs(shell_command, execution_config))
        return operation.run()
    except ShellCommandError as err:
        status = "failed"
        returncode = err.returncode
        error = str(err)
        raise
    except Exception as err:
        status = "failed"
        returncode = None
        error = str(err)
        raise
    finally:
        finished_at = datetime.now(timezone.utc)
        write_command_log_record(
            shell_command,
            execution_config,
            started_at=started_at,
            finished_at=finished_at,
            duration_seconds=time.monotonic() - start_time,
            status=status,
            returncode=returncode,
            error=error,
            profile=profile,
        )


def run_external_command(
    command: CommandInput,
    working_directory: Optional[str],
    execution_config: ExecutionConfig,
    *,
    environment: Optional[Mapping[str, str]] = None,
    name: Optional[str] = None,
    shell_operation_cls=None,
):
    """Execute a command list with the configured shell runner."""
    return run_shell_command(
        ShellCommand(
            command=command,
            environment={} if environment is None else dict(environment),
            working_directory=working_directory,
            name=name,
        ),
        execution_config,
        shell_operation_cls=shell_operation_cls,
    )
