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
    RAPTHOR_ROOT / "execution" / "calibrate" / "builders.py",
    RAPTHOR_ROOT / "execution" / "calibrate" / "commands.py",
    RAPTHOR_ROOT / "execution" / "calibrate" / "contracts.py",
    RAPTHOR_ROOT / "execution" / "calibrate" / "h5parm_sources.py",
    RAPTHOR_ROOT / "execution" / "calibrate" / "h5parm_combination.py",
    RAPTHOR_ROOT / "execution" / "calibrate" / "plotting.py",
    RAPTHOR_ROOT / "execution" / "calibrate" / "plotting_cli.py",
    RAPTHOR_ROOT / "execution" / "calibrate" / "prediction.py",
    RAPTHOR_ROOT / "execution" / "calibrate" / "gain_processing.py",
    RAPTHOR_ROOT / "execution" / "calibrate" / "screen_h5parms.py",
    RAPTHOR_ROOT / "execution" / "calibrate" / "solves.py",
    RAPTHOR_ROOT / "execution" / "calibrate" / "validation.py",
    RAPTHOR_ROOT / "execution" / "concatenate" / "measurement_sets.py",
    RAPTHOR_ROOT / "execution" / "concatenate" / "payloads.py",
    RAPTHOR_ROOT / "execution" / "image" / "commands.py",
    RAPTHOR_ROOT / "execution" / "image" / "builders.py",
    RAPTHOR_ROOT / "execution" / "image" / "contracts.py",
    RAPTHOR_ROOT / "execution" / "image" / "cube_catalog_cli.py",
    RAPTHOR_ROOT / "execution" / "image" / "cubes.py",
    RAPTHOR_ROOT / "execution" / "image" / "diagnostic_calculation.py",
    RAPTHOR_ROOT / "execution" / "image" / "flux_normalization.py",
    RAPTHOR_ROOT / "execution" / "image" / "outputs.py",
    RAPTHOR_ROOT / "execution" / "image" / "restoration.py",
    RAPTHOR_ROOT / "execution" / "image" / "skymodel_filter.py",
    RAPTHOR_ROOT / "execution" / "image" / "skymodel_filter_cli.py",
    RAPTHOR_ROOT / "execution" / "image" / "validation.py",
    RAPTHOR_ROOT / "execution" / "mosaic" / "commands.py",
    RAPTHOR_ROOT / "execution" / "mosaic" / "images.py",
    RAPTHOR_ROOT / "execution" / "mosaic" / "payloads.py",
    RAPTHOR_ROOT / "execution" / "outputs.py",
    RAPTHOR_ROOT / "execution" / "payloads.py",
    RAPTHOR_ROOT / "execution" / "pipeline" / "lifecycle.py",
    RAPTHOR_ROOT / "execution" / "pipeline" / "plan.py",
    RAPTHOR_ROOT / "execution" / "predict" / "commands.py",
    RAPTHOR_ROOT / "execution" / "predict" / "measurement_sets.py",
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

RETIRED_HELPER_SCRIPT_NAMES = (
    "add_sector_models.py",
    "adjust_h5parm_sources.py",
    "blank_image.py",
    "calculate_image_diagnostics.py",
    "check_image_beam.py",
    "collect_screen_h5parms.py",
    "combine_h5parms.py",
    "concat_ms.py",
    "filter_skymodel.py",
    "make_catalog_from_image_cube.py",
    "make_image_cube.py",
    "make_mosaic.py",
    "make_mosaic_template.py",
    "make_region_file.py",
    "normalize_flux_scale.py",
    "process_gains.py",
    "regrid_image.py",
    "restore_skymodel.py",
    "subtract_sector_models.py",
)

RETIRED_EXECUTABLE_NAMES = ("plotrapthor",)

RETIREMENT_TEXT_SCAN_FILES = (
    REPO_ROOT / "pyproject.toml",
    REPO_ROOT / "tests" / "execution" / "fixtures" / "command_reference.json",
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


def _string_constants(path: Path) -> list[tuple[str, int]]:
    tree = ast.parse(path.read_text(), filename=str(path))
    constants = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            constants.append((node.value, node.lineno))
    return constants


def _text_lines(path: Path) -> list[tuple[str, int]]:
    return [
        (line, line_number) for line_number, line in enumerate(path.read_text().splitlines(), 1)
    ]


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


def _script_retirement_scan_files() -> list[Path]:
    files = [
        path for path in _python_files(RAPTHOR_ROOT) if RAPTHOR_ROOT / "scripts" not in path.parents
    ]
    return sorted(files)


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


def test_production_code_does_not_use_retired_helper_script_wrappers():
    files = _script_retirement_scan_files()
    messages = _forbidden_import_messages(files, ("rapthor.scripts",))
    retired_names = RETIRED_HELPER_SCRIPT_NAMES + RETIRED_EXECUTABLE_NAMES

    for path in files:
        relative_path = _relative_path(path)
        for value, line_number in _string_constants(path):
            for script_name in retired_names:
                if script_name in value:
                    messages.append(f"{relative_path}:{line_number} references {script_name}")

    for path in RETIREMENT_TEXT_SCAN_FILES:
        relative_path = _relative_path(path)
        for value, line_number in _text_lines(path):
            for script_name in retired_names:
                if script_name in value:
                    messages.append(f"{relative_path}:{line_number} references {script_name}")

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
