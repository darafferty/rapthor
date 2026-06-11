"""
Test cases for the `rapthor.process` module.
"""

import pytest

import rapthor.process as process
from rapthor.process import (
    _do_calibrate_mode,
    chunk_observations,
    do_final_pass,
    make_report,
    run,
    run_steps,
)


def test_run_routes_to_prefect_process_flow(monkeypatch):
    calls = []

    def fake_process_flow(parset_file, logging_level="info"):
        calls.append((parset_file, logging_level))
        return "prefect-result"

    monkeypatch.setattr(
        "rapthor.execution.flows.process.process_flow",
        fake_process_flow,
    )

    assert run("input.parset", logging_level="debug") == "prefect-result"
    assert calls == [("input.parset", "debug")]


@pytest.mark.parametrize(
    "strategy, expected",
    [
        (
            {
                "di": [],
                "dd": ["fast_phase", "full_jones"],
            },
            {"di": False, "dd": True},
        ),  # No DI calibration
        (
            {
                "di": ["fast_phase"],
                "dd": [],
            },
            {"di": True, "dd": False},
        ),  # Fast DI calibration
        (
            {
                "di": ["full_jones"],
                "dd": [],
            },
            {"di": True, "dd": False},
        ),  # Full DI calibration
        (
            {
                "di": [],
                "dd": ["fast_phase", "full_jones"],
            },
            {"di": False, "dd": True},
        ),  # Fast DD calibration
        (
            {
                "di": ["fast_phase"],
                "dd": ["full_jones"],
            },
            {"di": True, "dd": True},
        ),  # Full Jones calibration
        (
            {
                "di": [],
                "dd": [],
            },
            {"di": False, "dd": False},
        ),  # No DD calibration
    ],
)
def test_do_calibrate_mode(strategy, expected):
    """Test function that determines whether or not to do DI or DD calibration"""
    assert _do_calibrate_mode(strategy) == expected


def test_do_calibrate_mode_with_unrecognized_modes_raises_error():
    """Test that _do_calibrate_mode raises a ValueError when no calibration modes are present"""
    with pytest.raises(
        ValueError,
        match=r"Calibration strategy {'unknown_mode': \['fast_phase', 'full_jones'\]} does not contain any of the calibration modes \['di', 'dd'\]",
    ):
        _do_calibrate_mode({"unknown_mode": ["fast_phase", "full_jones"]})


@pytest.mark.parametrize(
    "calibration_strategy, expected_calls",
    [
        ({"di": ["full_jones"]}, [("predict", "di", 1), ("calibrate", "di", 1)]),
        ({"dd": ["fast_phase"]}, [("calibrate", "dd", 1)]),
        (
            {"di": ["full_jones"], "dd": ["fast_phase"]},
            [("predict", "di", 1), ("calibrate", "di", 1), ("calibrate", "dd", 1)],
        ),
        (
            {"dd": ["fast_phase"], "di": ["full_jones"]},
            [("calibrate", "dd", 1), ("predict", "di", 1), ("calibrate", "di", 1)],
        ),
    ],
)
def test_run_steps_preserves_calibration_strategy_order(
    monkeypatch, calibration_strategy, expected_calls
):
    """Test that run_steps preserves the DI/DD ordering from calibration_strategy."""

    calls = []

    class RecordingOperation:
        operation_name = None

        def __init__(self, mode, field, index):
            self.mode = mode
            self.index = index

        def run(self):
            calls.append((self.operation_name, self.mode, self.index))

    class RecordingPredict(RecordingOperation):
        operation_name = "predict"

    class RecordingCalibrate(RecordingOperation):
        operation_name = "calibrate"

    class Field:
        cycle_number = 1
        dde_mode = "single"
        do_predict = False
        do_image = False
        do_check = False

        def update(self, step, index, final=False):
            self.__dict__.update(step)

    monkeypatch.setattr("rapthor.process.Predict", RecordingPredict)
    monkeypatch.setattr("rapthor.process.Calibrate", RecordingCalibrate)

    run_steps(
        Field(),
        [
            {
                "do_calibrate": True,
                "calibration_strategy": calibration_strategy,
            }
        ],
    )

    assert calls == expected_calls


@pytest.mark.parametrize(
    "name, calibration_strategy, expected_order, expected_handoffs",
    [
        (
            "di_only",
            {"di": ["fast_phase", "medium_phase"]},
            [
                ("predict", "di"),
                ("calibrate", "di"),
                ("image", None),
                ("mosaic", None),
            ],
            {
                "calibrate_di_predict_input": "cycle1-predict-di.ms",
                "image_di_h5parm": "cycle1-di-solutions.h5",
                "image_dd_h5parm": None,
            },
        ),
        (
            "dd_only",
            {"dd": ["fast_phase", "medium_phase"]},
            [
                ("calibrate", "dd"),
                ("image", None),
                ("mosaic", None),
            ],
            {
                "image_di_h5parm": None,
                "image_dd_h5parm": "cycle1-dd-solutions.h5",
                "image_region_file": "cycle1-field_facets_ds9.reg",
            },
        ),
        (
            "di_then_dd",
            {"di": ["fast_phase", "medium_phase"], "dd": ["fast_phase", "medium_phase"]},
            [
                ("predict", "di"),
                ("calibrate", "di"),
                ("calibrate", "dd"),
                ("image", None),
                ("mosaic", None),
            ],
            {
                "calibrate_di_predict_input": "cycle1-predict-di.ms",
                "calibrate_dd_di_h5parm": "cycle1-di-solutions.h5",
                "image_di_h5parm": "cycle1-di-solutions.h5",
                "image_dd_h5parm": "cycle1-dd-solutions.h5",
                "image_region_file": "cycle1-field_facets_ds9.reg",
            },
        ),
        (
            "dd_then_di",
            {"dd": ["fast_phase", "medium_phase"], "di": ["fast_phase", "medium_phase"]},
            [
                ("calibrate", "dd"),
                ("predict", "di"),
                ("calibrate", "di"),
                ("image", None),
                ("mosaic", None),
            ],
            {
                "predict_di_dd_h5parm": "cycle1-dd-solutions.h5",
                "calibrate_di_predict_input": "cycle1-predict-di.ms",
                "image_di_h5parm": "cycle1-di-solutions.h5",
                "image_dd_h5parm": "cycle1-dd-solutions.h5",
                "image_region_file": "cycle1-field_facets_ds9.reg",
            },
        ),
    ],
)
def test_run_steps_calibration_strategy_handoffs(
    monkeypatch, name, calibration_strategy, expected_order, expected_handoffs
):
    """Exercise the four calibration strategy paths from CALIBRATION_STRATEGY.md."""

    events = []

    def record(operation, mode, field, index):
        event = {
            "operation": operation,
            "mode": mode,
            "index": index,
            "di_h5parm": getattr(field, "di_h5parm_filename", None),
            "dd_h5parm": getattr(field, "dd_h5parm_filename", None),
            "predict_di_output": getattr(field, "predict_di_output_filename", None),
            "region_file": getattr(field, "field_region_file", None),
        }
        events.append(event)
        return event

    class RecordingPredict:
        def __init__(self, mode, field, index):
            self.mode = mode
            self.field = field
            self.index = index

        def run(self):
            record("predict", self.mode, self.field, self.index)
            if self.mode == "di":
                self.field.predict_di_output_filename = f"cycle{self.index}-predict-di.ms"
            else:
                self.field.predict_dd_output_filename = f"cycle{self.index}-predict-dd.ms"

    class RecordingCalibrate:
        def __init__(self, mode, field, index):
            self.mode = mode
            self.field = field
            self.index = index

        def run(self):
            record("calibrate", self.mode, self.field, self.index)
            if self.mode == "di":
                self.field.di_h5parm_filename = f"cycle{self.index}-di-solutions.h5"
                self.field.h5parm_filename = self.field.di_h5parm_filename
            else:
                self.field.dd_h5parm_filename = f"cycle{self.index}-dd-solutions.h5"
                self.field.h5parm_filename = self.field.dd_h5parm_filename
                self.field.field_region_file = f"cycle{self.index}-field_facets_ds9.reg"

    class RecordingImage:
        def __init__(self, field, index):
            self.field = field
            self.index = index

        def run(self):
            record("image", None, self.field, self.index)
            self.field.image_region_file = f"cycle{self.index}-sector_facets_ds9.reg"

    class RecordingMosaic:
        def __init__(self, field, index):
            self.field = field
            self.index = index

        def run(self):
            record("mosaic", None, self.field, self.index)

    class Field:
        cycle_number = 1
        dde_mode = "single"
        do_check = False
        do_normalize = False
        make_quv_images = False
        disable_iquv_clean = False
        save_image_cube = False
        image_cube_stokes_list = ["I"]
        parset = {"imaging_specific": {"skip_final_major_iteration": True}}

        def update(self, step, index, final=False):
            self.__dict__.update(step)
            self.cycle_number_seen = index
            self.final_seen = final

    monkeypatch.setattr("rapthor.process.Predict", RecordingPredict)
    monkeypatch.setattr("rapthor.process.Calibrate", RecordingCalibrate)
    monkeypatch.setattr("rapthor.process.Image", RecordingImage)
    monkeypatch.setattr("rapthor.process.Mosaic", RecordingMosaic)

    field = Field()
    run_steps(
        field,
        [
            {
                "do_calibrate": True,
                "do_predict": False,
                "do_image": True,
                "calibration_strategy": calibration_strategy,
            }
        ],
    )

    assert [(event["operation"], event["mode"]) for event in events] == expected_order, name

    image_event = next(event for event in events if event["operation"] == "image")
    assert image_event["di_h5parm"] == expected_handoffs["image_di_h5parm"]
    assert image_event["dd_h5parm"] == expected_handoffs["image_dd_h5parm"]

    if "image_region_file" in expected_handoffs:
        assert image_event["region_file"] == expected_handoffs["image_region_file"]

    predict_di_events = [
        event for event in events if event["operation"] == "predict" and event["mode"] == "di"
    ]
    if "predict_di_dd_h5parm" in expected_handoffs:
        assert predict_di_events[0]["dd_h5parm"] == expected_handoffs["predict_di_dd_h5parm"]

    calibrate_di_events = [
        event for event in events if event["operation"] == "calibrate" and event["mode"] == "di"
    ]
    if "calibrate_di_predict_input" in expected_handoffs:
        assert (
            calibrate_di_events[0]["predict_di_output"]
            == expected_handoffs["calibrate_di_predict_input"]
        )

    calibrate_dd_events = [
        event for event in events if event["operation"] == "calibrate" and event["mode"] == "dd"
    ]
    if "calibrate_dd_di_h5parm" in expected_handoffs:
        assert calibrate_dd_events[0]["di_h5parm"] == expected_handoffs["calibrate_dd_di_h5parm"]
