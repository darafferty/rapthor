import pytest

from rapthor.execution.outputs import cleanup_intermediate_outputs
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
