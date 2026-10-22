import shlex
from pathlib import Path

import pytest

import rapthor.execution.outputs as outputs_module
from rapthor.execution.config import ExecutionConfig
from rapthor.execution.outputs import cleanup_intermediate_outputs, make_product
from rapthor.lib.records import directory_record, file_record, partial_product_path
from tests.conftest import file_tree_state


@pytest.mark.parametrize("scratch_workspace", [False, True])
def test_cleanup_removes_only_owned_intermediates_and_protects_nested_outputs(
    tmp_path, scratch_workspace
):
    root = tmp_path / "operation"
    if scratch_workspace:
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        root.symlink_to(scratch, target_is_directory=True)
    else:
        root.mkdir()
    retained_dir = root / "retained.ms"
    ancestor = root / "products"
    for path in (retained_dir, ancestor, root / "temporary.ms"):
        path.mkdir()
    retained_file = ancestor / "solution.h5parm"
    secondary_file = root / "secondary.txt"
    input_file = root / "input.fits"
    paths = [retained_dir / "table.dat", retained_file, secondary_file, root / "temporary.fits"]
    unrelated = root / "unrelated.log"
    for path in paths + [input_file, unrelated]:
        path.write_text("data")
    output = file_record(retained_file)
    output["secondaryFiles"] = [file_record(secondary_file)]

    cleanup_intermediate_outputs(
        paths + [input_file, retained_dir, ancestor, root, root / "temporary.ms"],
        str(root),
        {"nested": [[output], directory_record(retained_dir)]},
        input_paths=[input_file],
    )

    assert all(path.is_file() for path in paths[:3] + [input_file, unrelated])
    assert not (root / "temporary.fits").exists()
    assert not (root / "temporary.ms").exists()


def test_cleanup_preserves_retained_symlink_targets(tmp_path):
    model = tmp_path / "model.fits"
    model.write_text("model")
    output = tmp_path / "retained-model.fits"
    output.symlink_to(model.name)
    ms = tmp_path / "model.ms"
    ms.mkdir()
    data = ms / "table.dat"
    data.write_text("table")
    retained_ms = tmp_path / "retained.ms"
    retained_ms.symlink_to(ms.name, target_is_directory=True)

    cleanup_intermediate_outputs(
        [model, ms, data], str(tmp_path), [file_record(output), directory_record(retained_ms)]
    )

    assert output.read_text() == "model"
    assert (retained_ms / "table.dat").read_text() == "table"


def test_cleanup_cannot_delete_outside_the_operation(tmp_path):
    root = tmp_path / "operation"
    root.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    file = external / "input.fits"
    file.write_text("input")
    alias = root / "external"
    alias.symlink_to(external, target_is_directory=True)
    owned_link = root / "temporary-link.fits"
    owned_link.symlink_to(file)

    cleanup_intermediate_outputs([file, external, alias / file.name, owned_link], str(root), {})

    assert file.read_text() == "input"
    assert alias.is_symlink()
    assert not owned_link.is_symlink()


class _FakeShellOperation:
    """Fake shell operation that writes a directory product and can fail after writing."""

    commands = []
    fail = False

    def __init__(self, **kwargs):
        self.command = shlex.split(kwargs["commands"][-1])

    def run(self):
        self.commands.append(self.command)
        output = Path(self.command[-1])
        output.mkdir()
        (output / "table.dat").write_text(self.command[0])
        if self.fail:
            raise RuntimeError("command failed")
        return "OK"


@pytest.fixture
def shell_operation_cls():
    _FakeShellOperation.commands = []
    _FakeShellOperation.fail = False
    return _FakeShellOperation


def _make_product(output, shell_operation_cls, *, settings="a", reuse=True):
    return make_product(
        str(output),
        [settings, partial_product_path(str(output))],
        "Test product",
        str(output.parent),
        ExecutionConfig(task_runner="sync"),
        directory=True,
        reuse=reuse,
        shell_operation_cls=shell_operation_cls,
    )


def test_make_product_gives_a_complete_product_its_final_name(tmp_path, shell_operation_cls):
    output = tmp_path / "product.ms"

    record = _make_product(output, shell_operation_cls)

    assert record == directory_record(output)
    assert shell_operation_cls.commands == [["a", str(tmp_path / ".product.ms.partial")]]
    assert (output / "table.dat").read_text() == "a"
    assert not (tmp_path / ".product.ms.partial").exists()


def test_make_product_restarts_after_a_failure_during_creation(tmp_path, shell_operation_cls):
    output = tmp_path / "product.ms"
    partial = tmp_path / ".product.ms.partial"
    shell_operation_cls.fail = True
    with pytest.raises(RuntimeError, match="command failed"):
        _make_product(output, shell_operation_cls)
    # Nothing gets the final name; the partial output is kept for inspection
    assert not output.exists()
    assert (partial / "table.dat").is_file()

    shell_operation_cls.fail = False
    assert _make_product(output, shell_operation_cls) == directory_record(output)
    assert len(shell_operation_cls.commands) == 2
    assert not partial.exists()


def test_make_product_reuses_a_complete_product(tmp_path, shell_operation_cls):
    output = tmp_path / "product.ms"
    _make_product(output, shell_operation_cls)
    state = file_tree_state(output)
    leftover = tmp_path / ".product.ms.partial"
    leftover.mkdir()

    assert _make_product(output, shell_operation_cls, settings="b") == directory_record(output)

    assert len(shell_operation_cls.commands) == 1
    assert file_tree_state(output) == state
    assert not leftover.exists()


def test_make_product_replaces_an_earlier_product_when_not_reusing(tmp_path, shell_operation_cls):
    output = tmp_path / "product.ms"
    output.mkdir()
    (output / "table.dat").write_text("earlier attempt")

    assert _make_product(output, shell_operation_cls, reuse=False) == directory_record(output)

    assert len(shell_operation_cls.commands) == 1
    assert (output / "table.dat").read_text() == "a"


def test_cleanup_moves_a_product_aside_before_removing_it(
    tmp_path, shell_operation_cls, monkeypatch
):
    removed = tmp_path / "removed.ms"
    kept = tmp_path / "kept.ms"
    for output in (removed, kept):
        _make_product(output, shell_operation_cls)

    def stop(path):
        raise KeyboardInterrupt

    with monkeypatch.context() as patch:
        patch.setattr(outputs_module.shutil, "rmtree", stop)
        with pytest.raises(KeyboardInterrupt):
            cleanup_intermediate_outputs([removed, kept], str(tmp_path), [directory_record(kept)])
    # A removal that is stopped leaves no incomplete product with its final name
    assert not removed.exists()
    assert (tmp_path / ".removed.ms.partial" / "table.dat").is_file()
    assert kept.is_dir()

    # The next attempt makes the product again and removes what was left
    _make_product(removed, shell_operation_cls)
    assert (removed / "table.dat").is_file()
    assert not (tmp_path / ".removed.ms.partial").exists()
