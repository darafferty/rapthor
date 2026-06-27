"""Architecture fitness checks for Rapthor's refactor boundaries."""

import ast
from pathlib import Path
from typing import Optional

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

DOMAIN_IMPORT_ALLOWLIST = {
    (
        "rapthor/lib/operation.py",
        "rapthor.execution.config",
    ): "Transitional Operation.run_prefect_flow helper; remove with operation lifecycle split.",
}

PURE_EXECUTION_MODULES = (
    RAPTHOR_ROOT / "execution" / "commands.py",
    RAPTHOR_ROOT / "execution" / "calibrate" / "commands.py",
    RAPTHOR_ROOT / "execution" / "calibrate" / "payloads.py",
    RAPTHOR_ROOT / "execution" / "calibrate" / "runner.py",
    RAPTHOR_ROOT / "execution" / "image" / "commands.py",
    RAPTHOR_ROOT / "execution" / "image" / "outputs.py",
    RAPTHOR_ROOT / "execution" / "image" / "payloads.py",
    RAPTHOR_ROOT / "execution" / "outputs.py",
    RAPTHOR_ROOT / "execution" / "payloads.py",
    RAPTHOR_ROOT / "execution" / "process_lifecycle.py",
    RAPTHOR_ROOT / "execution" / "process_plan.py",
)

PURE_EXECUTION_FORBIDDEN_PREFIXES = FRAMEWORK_PREFIXES + (
    "rapthor.execution.flows",
    "rapthor.operations",
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
    allowlist: Optional[dict[tuple[str, str], str]] = None,
) -> list[str]:
    allowlist = allowlist or {}
    messages = []
    for path in paths:
        relative_path = _relative_path(path)
        for module, line_number in _imported_modules(path):
            if not any(_matches_prefix(module, prefix) for prefix in forbidden_prefixes):
                continue
            if (relative_path, module) in allowlist:
                continue
            messages.append(f"{relative_path}:{line_number} imports {module}")
    return messages


def test_domain_layer_does_not_gain_new_execution_or_framework_imports():
    messages = _forbidden_import_messages(
        _python_files(RAPTHOR_ROOT / "lib"),
        DOMAIN_FORBIDDEN_PREFIXES,
        allowlist=DOMAIN_IMPORT_ALLOWLIST,
    )

    assert messages == []


def test_pure_execution_helpers_do_not_import_frameworks_or_flows():
    messages = _forbidden_import_messages(
        list(PURE_EXECUTION_MODULES),
        PURE_EXECUTION_FORBIDDEN_PREFIXES,
    )

    assert messages == []
