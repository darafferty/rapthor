import os
from pathlib import Path

import pytest

import rapthor.execution.outputs as outputs_module
from rapthor.execution.outputs import cleanup_intermediate_outputs, make_command_product
from rapthor.lib.records import directory_record, file_record


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


class _ProductCommand:
    """Fake command that writes a directory product, optionally failing part-way."""

    def __init__(self, fail_after_writing=False):
        self.fail_after_writing = fail_after_writing
        self.commands = []

    def __call__(self, command):
        self.commands.append(command)
        output = Path(command[-1])
        output.mkdir()
        (output / "table.dat").write_text(command[0])
        if self.fail_after_writing:
            raise RuntimeError("command failed")


def _make_product(output, run, *, settings="a", inputs=()):
    return make_command_product(
        str(output),
        lambda partial: [settings, partial],
        run,
        description="Test product",
        inputs=[str(path) for path in inputs],
        directory=True,
    )


def _tree_state(path):
    return sorted(
        (str(item.relative_to(path)), item.stat().st_mtime_ns, item.read_bytes())
        for item in path.rglob("*")
        if item.is_file()
    )


def test_make_command_product_promotes_a_complete_product(tmp_path):
    output = tmp_path / "product.ms"
    run = _ProductCommand()

    record = _make_product(output, run)

    assert record == directory_record(output)
    assert run.commands == [["a", str(tmp_path / ".product.ms.partial")]]
    assert (output / "table.dat").read_text() == "a"
    assert not (tmp_path / ".product.ms.partial").exists()
    assert (tmp_path / ".product.ms.product.json").is_file()


def test_make_command_product_restarts_after_a_failure_during_creation(tmp_path):
    output = tmp_path / "product.ms"
    partial = tmp_path / ".product.ms.partial"

    with pytest.raises(RuntimeError, match="command failed"):
        _make_product(output, _ProductCommand(fail_after_writing=True))

    # Nothing is promoted; the partial output is kept for inspection.
    assert not output.exists()
    assert (partial / "table.dat").is_file()

    run = _ProductCommand()
    assert _make_product(output, run) == directory_record(output)
    assert len(run.commands) == 1
    assert not partial.exists()


def test_make_command_product_restarts_after_a_failure_before_promotion(tmp_path, monkeypatch):
    output = tmp_path / "product.ms"

    original_replace = os.replace

    def stop_promotion(source, destination):
        if destination == str(output):
            raise KeyboardInterrupt
        return original_replace(source, destination)

    with monkeypatch.context() as patch:
        patch.setattr(outputs_module.os, "replace", stop_promotion)
        with pytest.raises(KeyboardInterrupt):
            _make_product(output, _ProductCommand())
    # The record was written, but it does not make a missing product complete.
    assert not output.exists()
    assert (tmp_path / ".product.ms.product.json").is_file()

    run = _ProductCommand()
    assert _make_product(output, run) == directory_record(output)
    assert len(run.commands) == 1


def test_make_command_product_reuses_a_complete_product(tmp_path):
    output = tmp_path / "product.ms"
    _make_product(output, _ProductCommand())
    state = _tree_state(output)
    leftover = tmp_path / ".product.ms.partial"
    leftover.mkdir()

    run = _ProductCommand()
    assert _make_product(output, run) == directory_record(output)

    assert run.commands == []
    assert _tree_state(output) == state
    assert not leftover.exists()


def test_make_command_product_remakes_a_product_without_a_matching_record(tmp_path):
    output = tmp_path / "product.ms"
    output.mkdir()
    (output / "table.dat").write_text("unknown")

    run = _ProductCommand()
    _make_product(output, run)
    assert len(run.commands) == 1
    assert (output / "table.dat").read_text() == "a"

    _make_product(output, run, settings="b")
    assert len(run.commands) == 2
    assert (output / "table.dat").read_text() == "b"


def test_make_command_product_remakes_a_product_when_an_input_changes(tmp_path):
    source = tmp_path / "source.fits"
    source.write_text("one")
    upstream = tmp_path / "upstream.ms"
    output = tmp_path / "product.ms"
    run = _ProductCommand()
    _make_product(upstream, run)
    _make_product(output, run, inputs=[source, upstream])
    assert len(run.commands) == 2

    _make_product(output, run, inputs=[source, upstream])
    assert len(run.commands) == 2

    os.utime(source, ns=(1, 1))
    _make_product(output, run, inputs=[source, upstream])
    assert len(run.commands) == 3

    # An upstream product made with other settings changes the downstream one.
    _make_product(upstream, run, settings="b")
    _make_product(output, run, inputs=[source, upstream])
    assert len(run.commands) == 5


def test_make_command_product_never_removes_an_input(tmp_path):
    original = tmp_path / "input.ms"
    original.mkdir()
    (original / "table.dat").write_text("input")
    state = _tree_state(original)
    run = _ProductCommand()

    for output in (original, original / "nested.ms", tmp_path):
        with pytest.raises(ValueError, match="would overwrite the input"):
            _make_product(output, run, inputs=[original])

    assert run.commands == []
    assert _tree_state(original) == state


def test_cleanup_removes_the_record_of_a_removed_product(tmp_path):
    removed = tmp_path / "removed.ms"
    kept = tmp_path / "kept.ms"
    run = _ProductCommand()
    _make_product(removed, run)
    _make_product(kept, run)

    cleanup_intermediate_outputs([removed, kept], str(tmp_path), [directory_record(kept)])

    assert not removed.exists()
    assert not (tmp_path / ".removed.ms.product.json").exists()
    assert kept.is_dir()
    assert (tmp_path / ".kept.ms.product.json").is_file()
