"""Architecture fitness checks for Rapthor's refactor boundaries."""

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RAPTHOR_ROOT = REPO_ROOT / "rapthor"

FRAMEWORK_PREFIXES = (
    "dask",
    "distributed",
    "prefect",
    "prefect_dask",
    "prefect_shell",
)

DOMAIN_FORBIDDEN_PREFIXES = FRAMEWORK_PREFIXES + ("rapthor.execution",)

PURE_EXECUTION_MODULES = (
    RAPTHOR_ROOT / "execution" / "commands.py",
    RAPTHOR_ROOT / "execution" / "calibrate" / "collection.py",
    RAPTHOR_ROOT / "execution" / "calibrate" / "commands.py",
    RAPTHOR_ROOT / "execution" / "calibrate" / "h5parm_sources.py",
    RAPTHOR_ROOT / "execution" / "calibrate" / "h5parm_combination.py",
    RAPTHOR_ROOT / "execution" / "calibrate" / "payloads.py",
    RAPTHOR_ROOT / "execution" / "calibrate" / "prediction.py",
    RAPTHOR_ROOT / "execution" / "calibrate" / "gain_processing.py",
    RAPTHOR_ROOT / "execution" / "calibrate" / "screen_h5parms.py",
    RAPTHOR_ROOT / "execution" / "calibrate" / "solves.py",
    RAPTHOR_ROOT / "execution" / "concatenate" / "commands.py",
    RAPTHOR_ROOT / "execution" / "concatenate" / "measurement_sets.py",
    RAPTHOR_ROOT / "execution" / "concatenate" / "payloads.py",
    RAPTHOR_ROOT / "execution" / "image" / "commands.py",
    RAPTHOR_ROOT / "execution" / "image" / "cubes.py",
    RAPTHOR_ROOT / "execution" / "image" / "diagnostic_calculation.py",
    RAPTHOR_ROOT / "execution" / "image" / "flux_normalization.py",
    RAPTHOR_ROOT / "execution" / "image" / "outputs.py",
    RAPTHOR_ROOT / "execution" / "image" / "payloads.py",
    RAPTHOR_ROOT / "execution" / "image" / "restoration.py",
    RAPTHOR_ROOT / "execution" / "image" / "skymodel_filter.py",
    RAPTHOR_ROOT / "execution" / "mosaic" / "commands.py",
    RAPTHOR_ROOT / "execution" / "mosaic" / "images.py",
    RAPTHOR_ROOT / "execution" / "mosaic" / "payloads.py",
    RAPTHOR_ROOT / "execution" / "outputs.py",
    RAPTHOR_ROOT / "execution" / "payloads.py",
    RAPTHOR_ROOT / "execution" / "pipeline" / "lifecycle.py",
    RAPTHOR_ROOT / "execution" / "pipeline" / "plan.py",
    RAPTHOR_ROOT / "execution" / "predict" / "commands.py",
    RAPTHOR_ROOT / "execution" / "predict" / "payloads.py",
    RAPTHOR_ROOT / "execution" / "predict" / "sector_model_addition.py",
    RAPTHOR_ROOT / "execution" / "predict" / "sector_model_subtraction.py",
    RAPTHOR_ROOT / "execution" / "regions.py",
)

PURE_EXECUTION_FORBIDDEN_PREFIXES = FRAMEWORK_PREFIXES + (
    "rapthor.execution.calibrate.flow",
    "rapthor.execution.concatenate.flow",
    "rapthor.execution.image.flow",
    "rapthor.execution.mosaic.flow",
    "rapthor.execution.pipeline.flow",
    "rapthor.execution.predict.flow",
    "rapthor.operations",
)

LIGHTWEIGHT_PACKAGE_INITIALIZERS = (
    RAPTHOR_ROOT / "execution" / "__init__.py",
    RAPTHOR_ROOT / "execution" / "calibrate" / "__init__.py",
    RAPTHOR_ROOT / "execution" / "concatenate" / "__init__.py",
    RAPTHOR_ROOT / "execution" / "image" / "__init__.py",
    RAPTHOR_ROOT / "execution" / "mosaic" / "__init__.py",
    RAPTHOR_ROOT / "execution" / "pipeline" / "__init__.py",
    RAPTHOR_ROOT / "execution" / "predict" / "__init__.py",
)


def _python_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.py") if path.is_file())


def _relative_path(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def _imported_modules(path: Path) -> list[tuple[str, int]]:
    tree = ast.parse(path.read_text(), filename=str(path))
    modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend((alias.name, node.lineno) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append((node.module, node.lineno))
    return modules


def _matches_prefix(module: str, prefix: str) -> bool:
    return module == prefix or module.startswith(f"{prefix}.")


def _forbidden_import_messages(
    paths: list[Path],
    forbidden_prefixes: tuple[str, ...],
) -> list[str]:
    messages = []
    for path in paths:
        relative_path = _relative_path(path)
        for module, line_number in _imported_modules(path):
            if not any(_matches_prefix(module, prefix) for prefix in forbidden_prefixes):
                continue
            messages.append(f"{relative_path}:{line_number} imports {module}")
    return messages


def test_domain_layer_does_not_gain_new_execution_or_framework_imports():
    messages = _forbidden_import_messages(
        _python_files(RAPTHOR_ROOT / "lib"),
        DOMAIN_FORBIDDEN_PREFIXES,
    )

    assert messages == []


def test_pure_execution_helpers_do_not_import_frameworks_or_flows():
    messages = _forbidden_import_messages(
        list(PURE_EXECUTION_MODULES),
        PURE_EXECUTION_FORBIDDEN_PREFIXES,
    )

    assert messages == []


def test_execution_package_initializers_do_not_rebuild_broad_facades():
    messages = []
    for path in LIGHTWEIGHT_PACKAGE_INITIALIZERS:
        imports = _imported_modules(path)
        if imports:
            relative_path = _relative_path(path)
            messages.extend(
                f"{relative_path}:{line_number} imports {module}" for module, line_number in imports
            )

    assert messages == []
