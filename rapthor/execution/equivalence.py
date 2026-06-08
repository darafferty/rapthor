"""Helpers for CWL-to-Prefect equivalence tests."""

import json
import math
import shutil
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

BackendRunner = Callable[[Path, Path, str], Any]
ParsetMaterializer = Callable[[Path, Path], Path]
PRODUCT_ROOTS = ("images", "h5parms", "skymodels", "regions")
FITS_SUFFIXES = (".fits", ".fits.fz")


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
    h5py = import_module("h5py")

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
    return sorted(operation_states)


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
    if isinstance(reference, Mapping) and isinstance(candidate, Mapping):
        differences = []
        for key in sorted(set(reference) | set(candidate)):
            child_path = f"{path}.{key}"
            if key not in reference:
                differences.append(EquivalenceDifference(child_path, "<missing>", candidate[key]))
            elif key not in candidate:
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

    if reference != candidate:
        return [EquivalenceDifference(path, reference, candidate)]
    return []


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
