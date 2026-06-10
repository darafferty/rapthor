"""Helpers for CWL-to-Prefect equivalence tests."""

import configparser
import json
import math
import os
import shutil
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from string import Template
from typing import Any, Callable, Mapping, Optional, Sequence

BackendRunner = Callable[[Path, Path, str], Any]
ParsetMaterializer = Callable[[Path, Path], Path]
PRODUCT_ROOTS = ("images", "h5parms", "solutions", "skymodels", "regions")
FITS_SUFFIXES = (".fits", ".fits.fz")
REFERENCE_ARTIFACT_ROOT_ENV = "RAPTHOR_CWL_REFERENCE_ROOT"
EQUIVALENCE_INPUT_MS_ENV = "RAPTHOR_EQUIVALENCE_INPUT_MS"
EQUIVALENCE_INPUT_SKYMODEL_ENV = "RAPTHOR_EQUIVALENCE_INPUT_SKYMODEL"
EQUIVALENCE_APPARENT_SKYMODEL_ENV = "RAPTHOR_EQUIVALENCE_APPARENT_SKYMODEL"
EQUIVALENCE_STRATEGY_ENV = "RAPTHOR_EQUIVALENCE_STRATEGY"

REFERENCE_ARTIFACT_ITEM_ORDER = (
    "artifact_dir",
    "pipelines",
    "operation_order",
    "product_roots",
    "fits_products",
    "h5parm_products",
    "skymodel_products",
    "region_products",
    "restart_state",
)

DEFAULT_PARSET_ENV_OVERRIDES = (
    ("global", "input_ms", EQUIVALENCE_INPUT_MS_ENV),
    ("global", "input_skymodel", EQUIVALENCE_INPUT_SKYMODEL_ENV),
    ("global", "apparent_skymodel", EQUIVALENCE_APPARENT_SKYMODEL_ENV),
    ("global", "strategy", EQUIVALENCE_STRATEGY_ENV),
    ("imaging", "photometry_skymodel", EQUIVALENCE_INPUT_SKYMODEL_ENV),
    ("imaging", "astrometry_skymodel", EQUIVALENCE_INPUT_SKYMODEL_ENV),
)


@dataclass(frozen=True)
class EquivalenceRunDirs:
    """Working directories for one CWL/Prefect comparison run."""

    root: Path
    cwl: Path
    prefect: Path


@dataclass(frozen=True)
class EquivalenceRun:
    """Result metadata for one backend invocation."""

    backend: str
    parset_file: Path
    working_dir: Path
    result: Any = None


@dataclass(frozen=True)
class EquivalenceDifference:
    """A single normalized difference between reference and candidate outputs."""

    path: str
    reference: Any
    candidate: Any


@dataclass(frozen=True)
class EquivalenceScenarioResult:
    """Result of comparing one saved-CWL scenario with a Prefect candidate run."""

    scenario_id: str
    reference_run: EquivalenceRun
    candidate_run: EquivalenceRun
    differences: tuple[EquivalenceDifference, ...]

    @property
    def ok(self) -> bool:
        """Return True when the candidate matches the saved reference."""
        return not self.differences


@dataclass(frozen=True)
class ReferenceArtifactCheck:
    """Availability check for one saved CWL reference artifact directory."""

    scenario_id: str
    artifact_dir: Path
    missing_items: tuple[str, ...]

    @property
    def ok(self) -> bool:
        """Return True when all required artifacts are present."""
        return not self.missing_items


def reference_artifact_root_from_environment(
    environ: Optional[Mapping[str, str]] = None,
) -> Optional[Path]:
    """Return the configured saved-CWL reference root, if one was provided."""
    environment = os.environ if environ is None else environ
    root = environment.get(REFERENCE_ARTIFACT_ROOT_ENV)
    if root in (None, ""):
        return None
    return Path(root)


def reference_artifact_dir(root_dir: Any, scenario: Mapping[str, Any]) -> Path:
    """Return the saved-CWL artifact directory for one equivalence scenario."""
    artifact_name = scenario.get("cwl_reference_artifact_dir") or scenario["id"]
    return Path(root_dir) / str(artifact_name)


def required_reference_artifact_items(scenario: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the saved-CWL artifact categories required by a scenario."""
    scopes = set(scenario.get("comparison_scopes", []))
    required = set()

    if "operations" in scopes:
        required.update({"pipelines", "operation_order"})
    if "products" in scopes:
        required.add("product_roots")
    if "fits" in scopes:
        required.add("fits_products")
    if "h5parm" in scopes:
        required.add("h5parm_products")
    if "skymodel" in scopes:
        required.add("skymodel_products")
    if "regions" in scopes:
        required.add("region_products")
    if "restart" in scopes:
        required.update({"pipelines", "operation_order", "restart_state"})

    return tuple(item for item in REFERENCE_ARTIFACT_ITEM_ORDER if item in required)


def _has_file(root: Path, predicate: Callable[[Path], bool]) -> bool:
    return root.exists() and any(predicate(path) for path in root.rglob("*") if path.is_file())


def _has_operation_order(artifact_dir: Path) -> bool:
    return any(
        path.is_file()
        for path in (
            artifact_dir / "operation_order.json",
            artifact_dir / "logs" / "operation_order.json",
        )
    )


def _has_restart_state(artifact_dir: Path) -> bool:
    pipelines_dir = artifact_dir / "pipelines"
    if not pipelines_dir.exists():
        return False
    for operation_dir in (path for path in pipelines_dir.rglob("*") if path.is_dir()):
        has_done = (operation_dir / ".done").is_file()
        has_outputs = any(
            (operation_dir / filename).is_file()
            for filename in (".outputs.json", "pipeline_outputs.json")
        )
        if has_done and has_outputs:
            return True
    return False


def _artifact_item_present(artifact_dir: Path, item: str) -> bool:
    if item == "artifact_dir":
        return artifact_dir.is_dir()
    if item == "pipelines":
        return (artifact_dir / "pipelines").is_dir()
    if item == "operation_order":
        return _has_operation_order(artifact_dir)
    if item == "product_roots":
        return any((artifact_dir / root_name).is_dir() for root_name in PRODUCT_ROOTS)
    if item == "fits_products":
        return any(
            _has_file(artifact_dir / root_name, lambda path: path.name.endswith(FITS_SUFFIXES))
            for root_name in PRODUCT_ROOTS
        )
    if item == "h5parm_products":
        return any(
            _has_file(artifact_dir / root_name, lambda path: path.suffix in {".h5", ".h5parm"})
            for root_name in ("h5parms", "solutions")
        )
    if item == "skymodel_products":
        return _has_file(
            artifact_dir / "skymodels", lambda path: path.suffix in {".txt", ".skymodel"}
        )
    if item == "region_products":
        return _has_file(artifact_dir / "regions", lambda path: True)
    if item == "restart_state":
        return _has_restart_state(artifact_dir)
    raise ValueError(f"Unknown reference artifact item: {item}")


def check_reference_artifacts(
    scenarios: Sequence[Mapping[str, Any]],
    root_dir: Optional[Any] = None,
    environ: Optional[Mapping[str, str]] = None,
) -> list[ReferenceArtifactCheck]:
    """Validate saved-CWL artifacts for the equivalence scenario manifest.

    If no root is passed and ``RAPTHOR_CWL_REFERENCE_ROOT`` is unset, no checks
    are returned. This lets ordinary unit tests run without staging artifacts
    while still providing a concrete validation hook for target environments.
    """
    root = (
        Path(root_dir)
        if root_dir is not None
        else reference_artifact_root_from_environment(environ)
    )
    if root is None:
        return []

    checks = []
    for scenario in scenarios:
        artifact_dir = reference_artifact_dir(root, scenario)
        required_items = ("artifact_dir", *required_reference_artifact_items(scenario))
        missing_items = tuple(
            item for item in required_items if not _artifact_item_present(artifact_dir, item)
        )
        checks.append(
            ReferenceArtifactCheck(
                scenario_id=str(scenario["id"]),
                artifact_dir=artifact_dir,
                missing_items=missing_items,
            )
        )
    return checks


def prepare_equivalence_run_dirs(root_dir: Any) -> EquivalenceRunDirs:
    """Create isolated CWL and Prefect working directories for a comparison."""
    root = Path(root_dir)
    cwl_dir = root / "cwl"
    prefect_dir = root / "prefect"
    cwl_dir.mkdir(parents=True, exist_ok=True)
    prefect_dir.mkdir(parents=True, exist_ok=True)
    return EquivalenceRunDirs(root=root, cwl=cwl_dir, prefect=prefect_dir)


def copy_parset_for_backend(parset_file: Any, working_dir: Any) -> Path:
    """Copy a source parset into a backend-specific working directory."""
    source = Path(parset_file)
    destination_dir = Path(working_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / source.name
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
    return destination


def _is_blank_parset_value(value: str) -> bool:
    return value.strip() in {"", "None"}


def _iter_parset_overrides(overrides: Optional[Mapping[str, Any]]):
    if not overrides:
        return
    for key, value in overrides.items():
        if isinstance(value, Mapping):
            for option, option_value in value.items():
                yield str(key), str(option), option_value
        elif "." in str(key):
            section, option = str(key).split(".", 1)
            yield section, option, value
        else:
            raise ValueError(
                "Parset overrides must use nested section mappings or dotted section.option keys"
            )


def _resolve_parset_override_value(value: Any, repo_root: Any, environ: Mapping[str, str]) -> str:
    if value is None:
        return "None"
    resolved = str(value)
    if resolved.startswith("repo:"):
        resolved = str(Path(repo_root) / resolved.removeprefix("repo:"))
    resolved = Template(resolved).safe_substitute(environ)
    if "${" in resolved:
        raise ValueError(f"Parset override has unresolved environment reference: {value!r}")
    return resolved


def materialize_scenario_parset(
    scenario: Mapping[str, Any],
    source_parset: Any,
    working_dir: Any,
    repo_root: Any = ".",
    environ: Optional[Mapping[str, str]] = None,
) -> Path:
    """Create a candidate parset for one saved-reference equivalence scenario.

    The materialized parset always points ``dir_working`` at ``working_dir``.
    Blank common template fields can be filled with
    ``RAPTHOR_EQUIVALENCE_INPUT_MS``, ``RAPTHOR_EQUIVALENCE_INPUT_SKYMODEL``,
    ``RAPTHOR_EQUIVALENCE_APPARENT_SKYMODEL``, and
    ``RAPTHOR_EQUIVALENCE_STRATEGY``. Scenarios may also provide
    ``parset_overrides`` either as nested section mappings or dotted
    ``section.option`` keys.
    """
    source = Path(source_parset)
    destination_dir = Path(working_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir = destination_dir / "scratch"
    scratch_dir.mkdir(exist_ok=True)

    environment = os.environ if environ is None else environ
    parser = configparser.ConfigParser(interpolation=None)
    parser.read(source)

    if not parser.has_section("global"):
        parser.add_section("global")
    parser.set("global", "dir_working", str(destination_dir))

    if parser.has_section("cluster"):
        if parser.has_option("cluster", "local_scratch_dir"):
            parser.set("cluster", "local_scratch_dir", str(scratch_dir))
        if parser.has_option("cluster", "global_scratch_dir"):
            parser.set("cluster", "global_scratch_dir", str(scratch_dir))
        if parser.has_option("cluster", "allow_internet_access") and _is_blank_parset_value(
            parser.get("cluster", "allow_internet_access")
        ):
            parser.set("cluster", "allow_internet_access", "False")

    for section, option, variable in DEFAULT_PARSET_ENV_OVERRIDES:
        if not parser.has_section(section) or not parser.has_option(section, option):
            continue
        if not _is_blank_parset_value(parser.get(section, option)):
            continue
        value = environment.get(variable)
        if value not in (None, ""):
            parser.set(section, option, value)

    for section, option, value in _iter_parset_overrides(scenario.get("parset_overrides")) or ():
        if not parser.has_section(section):
            parser.add_section(section)
        parser.set(
            section,
            option,
            _resolve_parset_override_value(value, repo_root=repo_root, environ=environment),
        )

    destination = destination_dir / source.name
    with destination.open("w") as handle:
        parser.write(handle)
    return destination


def scenario_parset_materializer(
    scenario: Mapping[str, Any],
    repo_root: Any = ".",
    environ: Optional[Mapping[str, str]] = None,
) -> ParsetMaterializer:
    """Return a materializer bound to one equivalence scenario."""

    def materializer(source_parset: Path, working_dir: Path) -> Path:
        return materialize_scenario_parset(
            scenario,
            source_parset,
            working_dir,
            repo_root=repo_root,
            environ=environ,
        )

    return materializer


def run_legacy_cwl_process(
    parset_file: Path,
    working_dir: Path,
    logging_level: str = "info",
) -> Any:
    """Invoke the retained legacy process route used as the CWL reference."""
    from rapthor import process

    return process.run(parset_file, logging_level=logging_level)


def run_prefect_process(
    parset_file: Path,
    working_dir: Path,
    logging_level: str = "info",
) -> Any:
    """Invoke the side-by-side Prefect top-level process flow."""
    from rapthor.execution.flows.process import process_flow

    return process_flow(parset_file, logging_level=logging_level)


def run_equivalence_pair(
    parset_file: Any,
    root_dir: Any,
    cwl_runner: BackendRunner = run_legacy_cwl_process,
    prefect_runner: BackendRunner = run_prefect_process,
    logging_level: str = "info",
    parset_materializer: ParsetMaterializer = copy_parset_for_backend,
) -> tuple[EquivalenceRun, EquivalenceRun]:
    """Run the same parset through isolated CWL and Prefect backends."""
    source_parset = Path(parset_file)
    run_dirs = prepare_equivalence_run_dirs(root_dir)
    cwl_parset = parset_materializer(source_parset, run_dirs.cwl)
    prefect_parset = parset_materializer(source_parset, run_dirs.prefect)

    cwl_result = cwl_runner(cwl_parset, run_dirs.cwl, logging_level)
    prefect_result = prefect_runner(prefect_parset, run_dirs.prefect, logging_level)

    return (
        EquivalenceRun("cwl", cwl_parset, run_dirs.cwl, cwl_result),
        EquivalenceRun("prefect", prefect_parset, run_dirs.prefect, prefect_result),
    )


def scenario_parset_file(scenario: Mapping[str, Any], repo_root: Any = ".") -> Path:
    """Return the parset fixture used to run one equivalence scenario."""
    for fixture_ref in scenario.get("fixture_refs", []):
        path = Path(str(fixture_ref))
        if path.suffix != ".parset":
            continue
        parset_path = path if path.is_absolute() else Path(repo_root) / path
        if not parset_path.is_file():
            raise FileNotFoundError(f"Scenario {scenario['id']} parset fixture not found: {path}")
        return parset_path
    raise ValueError(f"Scenario {scenario['id']} does not define a parset fixture")


def run_saved_reference_equivalence_scenario(
    scenario: Mapping[str, Any],
    reference_root: Any,
    run_root: Any,
    repo_root: Any = ".",
    prefect_runner: BackendRunner = run_prefect_process,
    logging_level: str = "info",
    parset_materializer: Optional[ParsetMaterializer] = None,
) -> tuple[EquivalenceRun, EquivalenceRun]:
    """Run the Prefect side of one scenario and pair it with saved CWL artifacts.

    The saved reference is expected at
    ``reference_root / scenario["cwl_reference_artifact_dir"]``. The
    materializer is responsible for producing a candidate parset whose
    ``dir_working`` points at the candidate working directory when real runs are
    executed.
    """
    scenario_id = str(scenario["id"])
    reference_dir = reference_artifact_dir(reference_root, scenario)
    candidate_dir = Path(run_root) / scenario_id / "prefect"
    candidate_dir.mkdir(parents=True, exist_ok=True)

    source_parset = scenario_parset_file(scenario, repo_root=repo_root)
    if parset_materializer is None:
        parset_materializer = scenario_parset_materializer(scenario, repo_root=repo_root)
    candidate_parset = parset_materializer(source_parset, candidate_dir)
    candidate_result = prefect_runner(candidate_parset, candidate_dir, logging_level)

    return (
        EquivalenceRun("cwl", source_parset, reference_dir),
        EquivalenceRun("prefect", candidate_parset, candidate_dir, candidate_result),
    )


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _optional_text(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    return path.read_text()


def _normalize_output_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        if value.get("class") in {"File", "Directory"} and "path" in value:
            return {"class": value["class"], "basename": Path(str(value["path"])).name}
        return {key: _normalize_output_value(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_normalize_output_value(item) for item in value]
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return round(value, 12)
    return value


def _numeric_stats(array: Any) -> dict[str, Any]:
    np = import_module("numpy")

    data = np.asarray(array)
    summary = {
        "shape": list(data.shape),
        "dtype": str(data.dtype),
    }
    if not np.issubdtype(data.dtype, np.number):
        return summary

    finite = data[np.isfinite(data)]
    summary["finite_count"] = int(finite.size)
    summary["nan_count"] = int(np.isnan(data).sum())
    if finite.size:
        summary.update(
            {
                "min": _jsonable(float(np.min(finite))),
                "max": _jsonable(float(np.max(finite))),
                "mean": _jsonable(float(np.mean(finite))),
                "std": _jsonable(float(np.std(finite))),
            }
        )
    return summary


def _fits_product_summary(path: Path) -> dict[str, Any]:
    fits = import_module("astropy.io.fits")

    hdus = []
    with fits.open(path) as hdul:
        for index, hdu in enumerate(hdul):
            data = getattr(hdu, "data", None)
            if data is None:
                continue
            hdu_summary = {"index": index}
            hdu_summary.update(_numeric_stats(data))
            hdus.append(hdu_summary)
    return {"type": "fits", "hdus": hdus}


def _h5parm_product_summary(path: Path) -> dict[str, Any]:
    try:
        h5py = import_module("h5py")
    except ModuleNotFoundError:
        return _h5parm_product_summary_pytables(path)

    solsets = []
    soltabs: dict[str, list[str]] = {}
    datasets = {}
    axes = {}
    with h5py.File(path, "r") as h5_file:

        def collect_item(name, node):
            parts = name.split("/")
            if isinstance(node, h5py.Group):
                if len(parts) == 1:
                    solsets.append(name)
                elif len(parts) == 2 and "val" in node:
                    soltabs.setdefault(parts[0], []).append(parts[1])
            elif isinstance(node, h5py.Dataset):
                datasets[name] = {
                    "shape": list(node.shape),
                    "dtype": str(node.dtype),
                }
                if "AXES" in node.attrs:
                    axes[name] = _jsonable(node.attrs["AXES"])

        h5_file.visititems(collect_item)

    return {
        "type": "h5parm",
        "solsets": sorted(solsets),
        "soltabs": {key: sorted(value) for key, value in sorted(soltabs.items())},
        "datasets": {key: datasets[key] for key in sorted(datasets)},
        "axes": {key: axes[key] for key in sorted(axes)},
    }


def _h5parm_product_summary_pytables(path: Path) -> dict[str, Any]:
    tables = import_module("tables")

    solsets = []
    soltabs: dict[str, list[str]] = {}
    datasets = {}
    axes = {}

    with tables.open_file(path, "r") as h5_file:
        for node in h5_file.walk_nodes("/"):
            name = node._v_pathname.strip("/")
            if not name:
                continue
            parts = name.split("/")
            if isinstance(node, tables.Group):
                if len(parts) == 1:
                    solsets.append(name)
                elif len(parts) == 2 and "val" in node._v_children:
                    soltabs.setdefault(parts[0], []).append(parts[1])
            elif isinstance(node, tables.Leaf):
                datasets[name] = {
                    "shape": list(node.shape),
                    "dtype": str(node.dtype),
                }
                if "AXES" in node._v_attrs._v_attrnames:
                    axes[name] = _jsonable(node._v_attrs["AXES"])

    return {
        "type": "h5parm",
        "solsets": sorted(solsets),
        "soltabs": {key: sorted(value) for key, value in sorted(soltabs.items())},
        "datasets": {key: datasets[key] for key in sorted(datasets)},
        "axes": {key: axes[key] for key in sorted(axes)},
    }


def _skymodel_product_summary(path: Path) -> dict[str, Any]:
    source_count = 0
    patch_count = 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("FORMAT"):
            continue
        first_column = stripped.split(",", 1)[0].strip()
        if first_column:
            source_count += 1
        else:
            patch_count += 1
    return {
        "type": "skymodel",
        "source_count": source_count,
        "patch_count": patch_count,
    }


def _region_product_summary(path: Path) -> dict[str, Any]:
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip()
    ]
    return {"type": "region", "lines": lines}


def _generic_product_summary(path: Path) -> dict[str, Any]:
    return {"type": "file", "basename": path.name}


def _product_summary(path: Path) -> dict[str, Any]:
    if path.name.endswith(FITS_SUFFIXES):
        return _fits_product_summary(path)
    if path.suffix in {".h5", ".h5parm"}:
        return _h5parm_product_summary(path)
    if path.parent.parts and "skymodels" in path.parts and path.suffix in {".txt", ".skymodel"}:
        return _skymodel_product_summary(path)
    if path.parent.parts and "regions" in path.parts:
        return _region_product_summary(path)
    return _generic_product_summary(path)


def collect_product_summaries(working_dir: Any) -> dict[str, Any]:
    """Collect backend-neutral summaries for final products."""
    root = Path(working_dir)
    products = {}
    for product_root_name in PRODUCT_ROOTS:
        product_root = root / product_root_name
        if not product_root.exists():
            continue
        for path in sorted(item for item in product_root.rglob("*") if item.is_file()):
            relative_path = path.relative_to(root).as_posix()
            products[relative_path] = _product_summary(path)
    return products


def _operation_outputs(operation_dir: Path) -> Any:
    outputs_path = operation_dir / ".outputs.json"
    if outputs_path.exists():
        return _load_json(outputs_path)
    pipeline_outputs_path = operation_dir / "pipeline_outputs.json"
    if pipeline_outputs_path.exists():
        return _load_json(pipeline_outputs_path)
    return {}


def collect_operation_states(working_dir: Any) -> dict[str, Any]:
    """Collect normalized operation completion and output-record state."""
    pipelines_dir = Path(working_dir) / "pipelines"
    if not pipelines_dir.exists():
        return {}

    states = {}
    for operation_dir in sorted(path for path in pipelines_dir.iterdir() if path.is_dir()):
        outputs = _operation_outputs(operation_dir)
        states[operation_dir.name] = {
            "done": (operation_dir / ".done").is_file(),
            "outputs": _normalize_output_value(outputs),
        }
    return states


def _operation_order(working_dir: Path, operation_states: Mapping[str, Any]) -> list[str]:
    for order_file in (
        working_dir / "operation_order.json",
        working_dir / "logs" / "operation_order.json",
    ):
        if order_file.exists():
            return list(_load_json(order_file))

    rapthor_log = working_dir / "logs" / "rapthor.log"
    if rapthor_log.exists():
        operation_order = _operation_order_from_rapthor_log(rapthor_log)
        if operation_order:
            return operation_order

    return sorted(operation_states)


def _operation_order_from_rapthor_log(log_path: Path) -> list[str]:
    operation_order = []
    seen = set()
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        marker = "<-- Operation "
        if marker not in line:
            continue
        operation = line.split(marker, 1)[1].split(" started", 1)[0].strip()
        if not operation or operation in seen:
            continue
        operation_order.append(operation)
        seen.add(operation)
    return operation_order


def collect_backend_summary(working_dir: Any) -> dict[str, Any]:
    """Collect backend-neutral state for a CWL/Prefect comparison."""
    root = Path(working_dir)
    operation_states = collect_operation_states(root)
    field_state_path = root / "field_state.json"
    return {
        "operation_order": _operation_order(root, operation_states),
        "operations": operation_states,
        "products": collect_product_summaries(root),
        "report": _optional_text(root / "logs" / "diagnostics.txt"),
        "field_state": _load_json(field_state_path) if field_state_path.exists() else None,
    }


def _compare_values(reference: Any, candidate: Any, path: str) -> list[EquivalenceDifference]:
    if _is_empty_optional(reference) and _is_empty_optional(candidate):
        return []

    if isinstance(reference, Mapping) and isinstance(candidate, Mapping):
        differences = []
        for key in sorted(set(reference) | set(candidate)):
            child_path = f"{path}.{key}"
            if key not in reference:
                if _is_empty_optional(candidate[key]):
                    continue
                differences.append(EquivalenceDifference(child_path, "<missing>", candidate[key]))
            elif key not in candidate:
                if _is_empty_optional(reference[key]):
                    continue
                differences.append(EquivalenceDifference(child_path, reference[key], "<missing>"))
            else:
                differences.extend(_compare_values(reference[key], candidate[key], child_path))
        return differences

    if isinstance(reference, list) and isinstance(candidate, list):
        differences = []
        if len(reference) != len(candidate):
            differences.append(
                EquivalenceDifference(f"{path}.length", len(reference), len(candidate))
            )
        for index, (reference_item, candidate_item) in enumerate(zip(reference, candidate)):
            differences.extend(_compare_values(reference_item, candidate_item, f"{path}[{index}]"))
        return differences

    if _equivalent_float(reference, candidate):
        return []

    if reference != candidate:
        return [EquivalenceDifference(path, reference, candidate)]
    return []


def _equivalent_float(reference: Any, candidate: Any) -> bool:
    return (
        isinstance(reference, float)
        and isinstance(candidate, float)
        and math.isclose(reference, candidate, rel_tol=1e-6, abs_tol=1e-9)
    )


def _is_empty_optional(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, list):
        return all(_is_empty_optional(item) for item in value)
    if isinstance(value, Mapping):
        return not value
    return False


def compare_backend_summaries(
    reference_summary: Mapping[str, Any],
    candidate_summary: Mapping[str, Any],
) -> list[EquivalenceDifference]:
    """Return structured differences between normalized backend summaries."""
    return _compare_values(reference_summary, candidate_summary, "$")


def compare_backend_runs(
    reference_working_dir: Any, candidate_working_dir: Any
) -> list[EquivalenceDifference]:
    """Collect and compare normalized CWL-reference and Prefect-candidate state."""
    return compare_backend_summaries(
        collect_backend_summary(reference_working_dir),
        collect_backend_summary(candidate_working_dir),
    )


def compare_saved_reference_equivalence_scenario(
    scenario: Mapping[str, Any],
    reference_root: Any,
    run_root: Any,
    repo_root: Any = ".",
    prefect_runner: BackendRunner = run_prefect_process,
    logging_level: str = "info",
    parset_materializer: Optional[ParsetMaterializer] = None,
) -> EquivalenceScenarioResult:
    """Compare one saved-CWL scenario against a freshly run Prefect candidate."""
    reference_run, candidate_run = run_saved_reference_equivalence_scenario(
        scenario,
        reference_root,
        run_root,
        repo_root=repo_root,
        prefect_runner=prefect_runner,
        logging_level=logging_level,
        parset_materializer=parset_materializer,
    )
    differences = tuple(compare_backend_runs(reference_run.working_dir, candidate_run.working_dir))
    return EquivalenceScenarioResult(
        scenario_id=str(scenario["id"]),
        reference_run=reference_run,
        candidate_run=candidate_run,
        differences=differences,
    )


def compare_saved_reference_equivalence_manifest(
    scenarios: Sequence[Mapping[str, Any]],
    reference_root: Any,
    run_root: Any,
    repo_root: Any = ".",
    prefect_runner: BackendRunner = run_prefect_process,
    logging_level: str = "info",
    parset_materializer: Optional[ParsetMaterializer] = None,
) -> list[EquivalenceScenarioResult]:
    """Compare every scenario in an equivalence manifest against saved CWL artifacts."""
    return [
        compare_saved_reference_equivalence_scenario(
            scenario,
            reference_root,
            run_root,
            repo_root=repo_root,
            prefect_runner=prefect_runner,
            logging_level=logging_level,
            parset_materializer=parset_materializer,
        )
        for scenario in scenarios
    ]


def _numeric_array_differences(
    reference: Any,
    candidate: Any,
    path: str,
    rtol: float,
    atol: float,
) -> list[EquivalenceDifference]:
    np = import_module("numpy")

    reference_array = np.asarray(reference)
    candidate_array = np.asarray(candidate)
    if reference_array.shape != candidate_array.shape:
        return [
            EquivalenceDifference(
                f"{path}.shape",
                tuple(reference_array.shape),
                tuple(candidate_array.shape),
            )
        ]

    if np.allclose(reference_array, candidate_array, rtol=rtol, atol=atol, equal_nan=True):
        return []

    absolute_difference = np.abs(reference_array - candidate_array)
    max_absolute_difference = 0.0
    if absolute_difference.size:
        max_absolute_difference = float(np.nanmax(absolute_difference))
    return [
        EquivalenceDifference(
            path,
            {"shape": tuple(reference_array.shape), "rtol": rtol, "atol": atol},
            {"max_absolute_difference": max_absolute_difference},
        )
    ]


def compare_fits_product(
    reference_path: Any,
    candidate_path: Any,
    rtol: float = 1e-6,
    atol: float = 0.0,
    hdu_index: int = 0,
) -> list[EquivalenceDifference]:
    """Compare FITS HDU data with numeric tolerances."""
    fits = import_module("astropy.io.fits")

    with fits.open(reference_path) as reference_hdul, fits.open(candidate_path) as candidate_hdul:
        return _numeric_array_differences(
            reference_hdul[hdu_index].data,
            candidate_hdul[hdu_index].data,
            f"$.fits[{hdu_index}].data",
            rtol,
            atol,
        )


def _h5_numeric_datasets(path: Any) -> dict[str, Any]:
    h5py = import_module("h5py")
    np = import_module("numpy")

    datasets = {}
    with h5py.File(path, "r") as h5_file:

        def collect_dataset(name, node):
            if isinstance(node, h5py.Dataset) and np.issubdtype(node.dtype, np.number):
                datasets[name] = node[()]

        h5_file.visititems(collect_dataset)
    return datasets


def compare_h5parm_product(
    reference_path: Any,
    candidate_path: Any,
    rtol: float = 1e-6,
    atol: float = 0.0,
) -> list[EquivalenceDifference]:
    """Compare numeric datasets in h5parm files with numeric tolerances."""
    reference_datasets = _h5_numeric_datasets(reference_path)
    candidate_datasets = _h5_numeric_datasets(candidate_path)
    differences = _compare_values(
        {dataset_name: None for dataset_name in reference_datasets},
        {dataset_name: None for dataset_name in candidate_datasets},
        "$.h5parm.datasets",
    )
    for dataset_name in sorted(set(reference_datasets) & set(candidate_datasets)):
        differences.extend(
            _numeric_array_differences(
                reference_datasets[dataset_name],
                candidate_datasets[dataset_name],
                f"$.h5parm.{dataset_name}",
                rtol,
                atol,
            )
        )
    return differences


def format_differences(differences: list[EquivalenceDifference]) -> str:
    """Format structured differences for assertion messages."""
    return "\n".join(
        f"{difference.path}: reference={difference.reference!r} candidate={difference.candidate!r}"
        for difference in differences
    )


def assert_backend_equivalent(reference_working_dir: Any, candidate_working_dir: Any) -> None:
    """Assert that two backend working directories have equivalent summaries."""
    differences = compare_backend_runs(reference_working_dir, candidate_working_dir)
    if differences:
        raise AssertionError(format_differences(differences))
