from rapthor.execution.run_names import (
    operation_cycle,
    operation_run_name,
    task_run_name,
    task_run_options,
    task_tags,
)


def test_operation_run_name_adds_mode_to_cycle_operation():
    payload = {
        "pipeline_working_dir": "/work/pipelines/calibrate_4",
        "mode": "dd",
    }

    assert operation_run_name(payload, "calibrate", mode=payload["mode"]) == "calibrate_dd_4"


def test_operation_run_name_keeps_di_cycle_readable():
    payload = {
        "pipeline_working_dir": "/work/pipelines/predict_di_1",
        "mode": "di",
    }

    assert operation_run_name(payload, "predict", mode=payload["mode"]) == "predict_di_1"


def test_operation_run_name_uses_directory_for_unmoded_operation():
    payload = {"pipeline_working_dir": "/work/pipelines/image_3"}

    assert operation_run_name(payload, "image") == "image_3"


def test_operation_run_name_falls_back_when_cycle_is_not_encoded():
    payload = {"pipeline_working_dir": "/tmp/random-operation-dir", "mode": "dd"}

    assert operation_run_name(payload, "calibrate", mode=payload["mode"]) == "calibrate_dd"


def test_operation_cycle_returns_none_when_missing():
    assert operation_cycle({"pipeline_working_dir": "/tmp/image"}) is None


def test_task_run_name_appends_clean_suffixes():
    assert task_run_name("chunk", 1) == "chunk_1"
    assert task_run_name("sector 1", "filter skymodel") == "sector_1_filter_skymodel"


def test_task_tags_are_sanitized_and_stable():
    assert task_tags("DP3", "python helper", "DP3", "", None) == ["dp3", "python_helper"]


def test_task_run_options_builds_name_and_tags_for_prefect():
    assert task_run_options("solve chunk", 1, tags=["DP3"]) == {
        "task_run_name": "solve_chunk_1",
        "tags": ["dp3"],
    }
