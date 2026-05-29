from rapthor.execution.workdirs import WorkDirectoryLayout


def test_work_directory_layout_builds_deterministic_task_dirs(tmp_path):
    layout = WorkDirectoryLayout.from_paths(tmp_path, "image_0")

    assert layout.operation_dir == tmp_path / "pipelines" / "image_0"
    assert layout.task_dir("wsclean", "sector/0") == (
        tmp_path / "pipelines" / "image_0" / "tasks" / "wsclean" / "sector_0"
    )


def test_ensure_task_dir_creates_directory(tmp_path):
    layout = WorkDirectoryLayout.from_paths(tmp_path, "predict")

    task_dir = layout.ensure_task_dir("dp3", "sector_0")

    assert task_dir.is_dir()


def test_atomic_write_text_replaces_tmp_file(tmp_path):
    layout = WorkDirectoryLayout.from_paths(tmp_path, "image")
    output = tmp_path / "result.txt"

    layout.atomic_write_text(output, "content")

    assert output.read_text() == "content"
    assert not layout.temp_path(output).exists()


def test_atomic_write_json_writes_stable_json(tmp_path):
    layout = WorkDirectoryLayout.from_paths(tmp_path, "image")
    output = tmp_path / "outputs.json"

    layout.atomic_write_json(output, {"b": 2, "a": 1})

    assert output.read_text() == '{\n    "a": 1,\n    "b": 2\n}'
