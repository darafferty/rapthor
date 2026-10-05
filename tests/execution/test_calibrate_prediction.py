import os
import shlex
from pathlib import Path

import pytest

import rapthor.execution.calibrate.prediction as calibrate_prediction
from rapthor.execution.config import ExecutionConfig


@pytest.mark.parametrize(
    ("frequencies", "fallback_bandwidth", "max_bandwidth", "expected_chunks"),
    [
        (
            [150_000_000.0],
            1_000_000.0,
            2_000_000.0,
            [
                {
                    "frequency_bandwidth": [150_000_000.0, 1_000_000.0],
                    "channel_range": (0, 1),
                }
            ],
        ),
        (
            [100.0, 110.0, 120.0, 130.0],
            40.0,
            20.0,
            [
                {"frequency_bandwidth": [105.0, 20.0], "channel_range": (0, 2)},
                {"frequency_bandwidth": [125.0, 20.0], "channel_range": (2, 4)},
            ],
        ),
        (
            [100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0],
            80.0,
            30.0,
            [
                {"frequency_bandwidth": [110.0, 30.0], "channel_range": (0, 3)},
                {"frequency_bandwidth": [140.0, 30.0], "channel_range": (3, 6)},
                {"frequency_bandwidth": [165.0, 20.0], "channel_range": (6, 8)},
            ],
        ),
    ],
    ids=["single-channel", "exact-division", "uneven-final-chunk"],
)
def test_wsclean_prediction_frequency_chunks_cover_each_channel_once(
    monkeypatch,
    frequencies,
    fallback_bandwidth,
    max_bandwidth,
    expected_chunks,
):
    """Frequency chunks must be complete, ordered, and non-overlapping."""
    monkeypatch.setattr(
        calibrate_prediction,
        "_measurement_set_channel_frequencies",
        lambda _msin: frequencies,
    )

    chunks = calibrate_prediction._frequency_chunks_for_ms(
        "input.ms",
        [frequencies[0], fallback_bandwidth],
        max_bandwidth_hz=max_bandwidth,
    )

    assert chunks == expected_chunks
    covered_channels = [channel for chunk in chunks for channel in range(*chunk["channel_range"])]
    assert covered_channels == list(range(len(frequencies)))


@pytest.mark.parametrize(
    "scratch_option, keep_temporary_files",
    [("local_scratch_dir", False), ("local_scratch_dir", True), (None, False)],
)
@pytest.mark.parametrize("fail_prediction", [False, True])
def test_wsclean_prediction_uses_owned_scratch_for_command_and_environment(
    tmp_path, monkeypatch, scratch_option, keep_temporary_files, fail_prediction
):
    workdir = tmp_path / "working"
    workdir.mkdir()
    input_ms = tmp_path / "input.ms"
    input_ms.mkdir()
    scratch = tmp_path / "local-scratch"
    scratch.mkdir()
    unrelated_file = scratch / "another-job.txt"
    unrelated_file.write_text("preserve")
    inherited_tmpdir = os.environ.get("TMPDIR")
    prediction_directories = []
    monkeypatch.setattr(
        calibrate_prediction,
        "_measurement_set_channel_frequencies",
        lambda _msin: [150_000_000.0],
    )

    class RecordingShellOperation:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def run(self):
            command = shlex.split(self.kwargs["commands"][-1])
            if "-draw-model" in command:
                root = command[command.index("-name") + 1]
                Path(f"{root}-term-0.fits").write_bytes(b"model")
                return
            assert "-predict" in command
            if scratch_option is None:
                assert "-temp-dir" not in command
                assert "TMPDIR" not in self.kwargs.get("env", {})
            else:
                temp_dir = Path(command[command.index("-temp-dir") + 1])
                assert temp_dir.parent == scratch
                assert temp_dir.is_dir()
                for variable in ("TMPDIR", "TMP", "TEMP"):
                    assert self.kwargs["env"][variable] == str(temp_dir)
                (temp_dir / "reordered-data").write_text("temporary")
                prediction_directories.append(temp_dir)
            if fail_prediction:
                raise RuntimeError("predict failed")

    config_kwargs = {"keep_temporary_files": keep_temporary_files}
    if scratch_option is not None:
        config_kwargs[scratch_option] = str(scratch)
    config = ExecutionConfig(**config_kwargs)
    payload = {
        "pipeline_working_dir": str(workdir),
        "max_threads": 1,
        "image_predict": {
            "skymodel": "model.txt",
            "facet_region_path": "facets.reg",
            "model_image_ra_dec": ["12:00:00.0", "+45.00.00.0"],
            "model_image_frequency_bandwidth": [150_000_000.0, 1_000_000.0],
            "model_image_cellsize": 0.001,
            "model_image_imsize": [16, 16],
        },
    }

    def predict():
        return calibrate_prediction.prepare_wsclean_predict_chunk(
            payload,
            {"msin": str(input_ms)},
            0,
            {"patch_names": ["patch1", "patch2"]},
            config,
            shell_operation_cls=RecordingShellOperation,
        )

    if fail_prediction:
        with pytest.raises(RuntimeError, match="predict failed"):
            predict()
    else:
        prepared = predict()
        assert Path(prepared["msin"]).parent == workdir
        assert Path(prepared["msin"]).is_dir()
    if scratch_option is not None:
        assert len(prediction_directories) == (1 if fail_prediction else 2)
        assert len(set(prediction_directories)) == len(prediction_directories)
        for temp_dir in prediction_directories:
            assert temp_dir.exists() == keep_temporary_files
        if not keep_temporary_files:
            assert list(scratch.iterdir()) == [unrelated_file]
    assert unrelated_file.read_text() == "preserve"
    assert os.environ.get("TMPDIR") == inherited_tmpdir
