from dataclasses import dataclass, field

import pytest
from prefect.testing.utilities import prefect_test_harness

from rapthor.execution.config import ExecutionConfig
from rapthor.execution.flows.process import (
    ProcessOperationFactories,
    process_steps_flow,
    run_process_steps,
)


@dataclass
class SelfcalState:
    converged: bool = False
    diverged: bool = False
    failed: bool = False

    def __iter__(self):
        return iter((self.converged, self.diverged, self.failed))


@dataclass
class RecordingField:
    cycle_number: int = 1
    dde_mode: str = "single"
    do_check: bool = False
    do_normalize: bool = False
    make_quv_images: bool = False
    disable_iquv_clean: bool = False
    save_image_cube: bool = False
    image_cube_stokes_list: list[str] = field(default_factory=lambda: ["I"])
    parset: dict = field(
        default_factory=lambda: {"imaging_specific": {"skip_final_major_iteration": True}}
    )
    convergence_ratio: float = 1.05
    divergence_ratio: float = 1.25
    failure_ratio: float = 2.0
    events: list[dict] = field(default_factory=list)
    selfcal_result: SelfcalState = field(default_factory=SelfcalState)

    def update(self, step, index, final=False):
        self.__dict__.update(step)
        self.cycle_number_seen = index
        self.final_seen = final

    def check_selfcal_progress(self):
        return self.selfcal_result

    def define_normalize_sector(self):
        self.normalization_sector_defined = True


def _record(field, operation, mode, index):
    event = {
        "operation": operation,
        "mode": mode,
        "index": index,
        "di_h5parm": getattr(field, "di_h5parm_filename", None),
        "dd_h5parm": getattr(field, "dd_h5parm_filename", None),
        "predict_di_output": getattr(field, "predict_di_output_filename", None),
        "field_region_file": getattr(field, "field_region_file", None),
    }
    field.events.append(event)
    return event


class RecordingPredict:
    def __init__(self, mode, field, index):
        self.mode = mode
        self.field = field
        self.index = index

    def run(self):
        _record(self.field, "predict", self.mode, self.index)
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
        _record(self.field, "calibrate", self.mode, self.index)
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
        _record(self.field, "image", None, self.index)
        self.field.image_region_file = f"cycle{self.index}-sector_facets_ds9.reg"


class RecordingMosaic:
    def __init__(self, field, index):
        self.field = field
        self.index = index

    def run(self):
        _record(self.field, "mosaic", None, self.index)


class RecordingImageNormalize:
    def __init__(self, field, index):
        self.field = field
        self.index = index

    def run(self):
        _record(self.field, "image_normalize", None, self.index)


RECORDING_FACTORIES = ProcessOperationFactories(
    predict=RecordingPredict,
    calibrate=RecordingCalibrate,
    image=RecordingImage,
    mosaic=RecordingMosaic,
    image_normalize=RecordingImageNormalize,
)


STRATEGY_CASES = [
    pytest.param(
        {"di": ["fast_phase", "medium_phase"]},
        [
            ("predict", "di"),
            ("calibrate", "di"),
            ("image", None),
            ("mosaic", None),
        ],
        {
            "image_di_h5parm": "cycle1-di-solutions.h5",
            "image_dd_h5parm": None,
        },
        id="di-only",
    ),
    pytest.param(
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
        id="dd-only",
    ),
    pytest.param(
        {"di": ["fast_phase", "medium_phase"], "dd": ["fast_phase", "medium_phase"]},
        [
            ("predict", "di"),
            ("calibrate", "di"),
            ("calibrate", "dd"),
            ("image", None),
            ("mosaic", None),
        ],
        {
            "calibrate_dd_di_h5parm": "cycle1-di-solutions.h5",
            "image_di_h5parm": "cycle1-di-solutions.h5",
            "image_dd_h5parm": "cycle1-dd-solutions.h5",
            "image_region_file": "cycle1-field_facets_ds9.reg",
        },
        id="di-then-dd",
    ),
    pytest.param(
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
            "image_di_h5parm": "cycle1-di-solutions.h5",
            "image_dd_h5parm": "cycle1-dd-solutions.h5",
            "image_region_file": "cycle1-field_facets_ds9.reg",
        },
        id="dd-then-di",
    ),
]


def _single_step(calibration_strategy, **updates):
    step = {
        "do_calibrate": True,
        "do_predict": False,
        "do_image": True,
        "calibration_strategy": calibration_strategy,
    }
    step.update(updates)
    return step


def _assert_strategy_handoffs(field, expected_order, expected_handoffs):
    assert [(event["operation"], event["mode"]) for event in field.events] == expected_order

    image_event = next(event for event in field.events if event["operation"] == "image")
    assert image_event["di_h5parm"] == expected_handoffs["image_di_h5parm"]
    assert image_event["dd_h5parm"] == expected_handoffs["image_dd_h5parm"]
    if "image_region_file" in expected_handoffs:
        assert image_event["field_region_file"] == expected_handoffs["image_region_file"]

    predict_di_events = [
        event for event in field.events if event["operation"] == "predict" and event["mode"] == "di"
    ]
    if "predict_di_dd_h5parm" in expected_handoffs:
        assert predict_di_events[0]["dd_h5parm"] == expected_handoffs["predict_di_dd_h5parm"]

    calibrate_dd_events = [
        event
        for event in field.events
        if event["operation"] == "calibrate" and event["mode"] == "dd"
    ]
    if "calibrate_dd_di_h5parm" in expected_handoffs:
        assert calibrate_dd_events[0]["di_h5parm"] == expected_handoffs["calibrate_dd_di_h5parm"]


@pytest.mark.parametrize("calibration_strategy, expected_order, expected_handoffs", STRATEGY_CASES)
def test_run_process_steps_calibration_strategy_handoffs(
    calibration_strategy, expected_order, expected_handoffs
):
    field = RecordingField()

    run_process_steps(
        field,
        [_single_step(calibration_strategy)],
        operation_factories=RECORDING_FACTORIES,
    )

    _assert_strategy_handoffs(field, expected_order, expected_handoffs)


@pytest.mark.parametrize("calibration_strategy, expected_order, expected_handoffs", STRATEGY_CASES)
def test_process_steps_flow_calibration_strategy_handoffs(
    calibration_strategy, expected_order, expected_handoffs
):
    field = RecordingField()

    with prefect_test_harness():
        process_steps_flow(
            field,
            [_single_step(calibration_strategy)],
            operation_factories=RECORDING_FACTORIES,
            execution_config=ExecutionConfig(task_runner="sync"),
        )

    _assert_strategy_handoffs(field, expected_order, expected_handoffs)


def test_run_process_steps_stops_after_selfcal_converges():
    field = RecordingField(selfcal_result=SelfcalState(converged=True))
    steps = [
        _single_step({"dd": ["fast_phase"]}, do_check=True),
        _single_step({"di": ["fast_phase"]}, do_check=True),
    ]

    run_process_steps(field, steps, operation_factories=RECORDING_FACTORIES)

    assert [(event["operation"], event["mode"]) for event in field.events] == [
        ("calibrate", "dd"),
        ("image", None),
        ("mosaic", None),
    ]
    assert field.selfcal_state.converged is True
    assert field.cycle_number == 1
