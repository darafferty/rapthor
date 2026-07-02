import copy
import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from prefect.testing.utilities import prefect_test_harness

from rapthor.execution.capabilities import PreflightError, preflight_execution
from rapthor.execution.config import ExecutionConfig
from rapthor.execution.pipeline.flow import (
    PipelineLifecycleHooks,
    PipelineOperationFactories,
    pipeline_flow,
    run_pipeline,
    run_pipeline_steps,
)
from rapthor.execution.pipeline.plan import (
    SUPPORTED_PIPELINE_FEATURES,
    collect_pipeline_features,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures"
REPO_ROOT = Path(__file__).parents[2]


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
    epoch_observations: list[list[object]] = field(default_factory=lambda: [[object()]])
    do_check: bool = False
    do_normalize: bool = False
    make_quv_images: bool = False
    disable_iquv_clean: bool = False
    save_image_cube: bool = False
    use_mpi: bool = False
    generate_screens: bool = False
    apply_screens: bool = False
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

    def define_full_field_sector(self, radius):
        self.full_field_radius = radius
        self.events.append({"operation": "define_full_field_sector", "radius": radius})


def _record(field, operation, mode, index):
    event = {
        "operation": operation,
        "mode": mode,
        "index": index,
        "di_h5parm": getattr(field, "di_h5parm_filename", None),
        "dd_h5parm": getattr(field, "dd_h5parm_filename", None),
        "predict_di_output": getattr(field, "predict_di_output_filename", None),
        "field_region_file": getattr(field, "field_region_file", None),
        "image_pol": getattr(field, "image_pol", None),
        "disable_clean": getattr(field, "disable_clean", None),
        "make_image_cube": getattr(field, "make_image_cube", None),
        "image_cube_stokes_list": list(getattr(field, "image_cube_stokes_list", [])),
        "generate_screens": getattr(field, "generate_screens", None),
        "apply_screens": getattr(field, "apply_screens", None),
        "skip_final_major_iteration": getattr(field, "skip_final_major_iteration", None),
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


class RecordingConcatenate:
    def __init__(self, field, index):
        self.field = field
        self.index = index

    def run(self):
        _record(self.field, "concatenate", None, self.index)


class RecordingImageInitial:
    def __init__(self, field):
        self.field = field

    def run(self):
        _record(self.field, "image_initial", None, None)


RECORDING_FACTORIES = PipelineOperationFactories(
    predict=RecordingPredict,
    calibrate=RecordingCalibrate,
    image=RecordingImage,
    mosaic=RecordingMosaic,
    image_normalize=RecordingImageNormalize,
    concatenate=RecordingConcatenate,
    image_initial=RecordingImageInitial,
)


def _run_prefect_pipeline(parset, strategy_steps, field_updates=None):
    lifecycle = RecordingPipelineLifecycle(
        parset=parset,
        strategy_steps=strategy_steps,
        supported_features=SUPPORTED_PIPELINE_FEATURES,
        field_updates=field_updates or {},
    )
    field = run_pipeline(
        "input.parset",
        logging_level="debug",
        operation_factories=RECORDING_FACTORIES,
        lifecycle_hooks=lifecycle.hooks(),
        execution_config=ExecutionConfig(task_runner="sync"),
    )
    return {
        "field": field,
        "read_files": lifecycle.read_files,
        "logging_levels": lifecycle.logging_levels,
    }


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


def _image_step(**updates):
    step = {
        "do_calibrate": False,
        "do_predict": False,
        "do_image": True,
        "do_check": False,
        "peel_outliers": False,
        "peel_bright_sources": False,
    }
    step.update(updates)
    return step


def _pipeline_parset(**updates):
    parset = {
        "strategy": "test-strategy",
        "generate_initial_skymodel": False,
        "download_initial_skymodel": False,
        "generate_initial_skymodel_radius": 3.0,
        "generate_initial_skymodel_data_fraction": 0.25,
        "selfcal_data_fraction": 0.5,
        "final_data_fraction": 1.0,
        "ntimes_to_repeat_final_cycle": 0,
        "input_h5parm": "input-solutions.h5",
        "input_skymodel": None,
        "imaging_specific": {"skip_final_major_iteration": True},
    }
    parset.update(updates)
    return parset


def _merge_feature_matrix():
    return json.loads((FIXTURE_DIR / "supported_merge_feature_matrix.json").read_text())


def _execution_config_from_matrix(entry):
    return ExecutionConfig(**entry.get("execution_config", {"task_runner": "sync"}))


@dataclass
class RecordingPipelineLifecycle:
    parset: dict
    strategy_steps: list[dict]
    final_pass: bool = True
    supported_features: set[str] | None = None
    epoch_observations: list[list[object]] = field(default_factory=lambda: [[object()]])
    read_files: list[object] = field(default_factory=list)
    logging_levels: list[str] = field(default_factory=list)
    preflight_features: set[str] = field(default_factory=set)
    field_updates: dict = field(default_factory=dict)
    field: object = None

    def hooks(self):
        return PipelineLifecycleHooks(
            read_parset=self.read_parset,
            set_logging_level=self.set_logging_level,
            build_field=self.build_field,
            set_strategy=self.set_strategy,
            validate_strategy=self.validate_strategy,
            preflight_execution=self.preflight_execution,
            chunk_observations=self.chunk_observations,
            do_final_pass=self.do_final_pass,
            make_report=self.make_report,
        )

    def read_parset(self, parset_file):
        self.read_files.append(parset_file)
        return self.parset

    def set_logging_level(self, logging_level):
        self.logging_levels.append(logging_level)

    def build_field(self, parset):
        self.field = RecordingField(parset=parset, epoch_observations=self.epoch_observations)
        for name, value in self.field_updates.items():
            setattr(self.field, name, value)
        return self.field

    def set_strategy(self, field):
        field.events.append({"operation": "set_strategy"})
        return self.strategy_steps

    def validate_strategy(self, strategy_steps, parset):
        self.field.events.append(
            {"operation": "validate_strategy", "step_count": len(strategy_steps)}
        )

    def preflight_execution(self, field, strategy_steps, execution_config, requested_features):
        self.preflight_features = set(requested_features)
        field.events.append(
            {
                "operation": "preflight",
                "feature_count": len(requested_features),
                "task_runner": execution_config.task_runner,
            }
        )
        preflight_execution(
            execution_config,
            requested_features=requested_features,
            supported_features=self.supported_features,
        )

    def chunk_observations(self, field, steps, data_fraction):
        field.events.append(
            {
                "operation": "chunk",
                "step_count": len(steps),
                "data_fraction": data_fraction,
            }
        )

    def do_final_pass(self, field, selfcal_steps, final_step):
        field.events.append(
            {
                "operation": "do_final_pass",
                "selfcal_step_count": len(selfcal_steps),
                "final_do_image": final_step["do_image"],
            }
        )
        return self.final_pass

    def make_report(self, field):
        field.events.append({"operation": "report"})


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
def test_run_pipeline_steps_calibration_strategy_handoffs(
    calibration_strategy, expected_order, expected_handoffs
):
    field = RecordingField()

    run_pipeline_steps(
        field,
        [_single_step(calibration_strategy)],
        operation_factories=RECORDING_FACTORIES,
    )

    _assert_strategy_handoffs(field, expected_order, expected_handoffs)


def test_run_pipeline_steps_stops_after_selfcal_converges():
    field = RecordingField(selfcal_result=SelfcalState(converged=True))
    steps = [
        _single_step({"dd": ["fast_phase"]}, do_check=True),
        _single_step({"di": ["fast_phase"]}, do_check=True),
    ]

    run_pipeline_steps(field, steps, operation_factories=RECORDING_FACTORIES)

    assert [(event["operation"], event["mode"]) for event in field.events] == [
        ("calibrate", "dd"),
        ("image", None),
        ("mosaic", None),
    ]
    assert field.selfcal_state.converged is True
    assert field.cycle_number == 1


@pytest.mark.parametrize(
    "selfcal_state, expected_flag",
    [
        pytest.param(SelfcalState(diverged=True), "diverged", id="diverged"),
        pytest.param(SelfcalState(failed=True), "failed", id="failed"),
    ],
)
def test_run_pipeline_steps_stops_after_selfcal_diverges_or_fails(selfcal_state, expected_flag):
    field = RecordingField(selfcal_result=selfcal_state)
    steps = [
        _single_step({"dd": ["fast_phase"]}, do_check=True),
        _single_step({"di": ["fast_phase"]}, do_check=True),
    ]

    run_pipeline_steps(field, steps, operation_factories=RECORDING_FACTORIES)

    assert [(event["operation"], event["mode"]) for event in field.events] == [
        ("calibrate", "dd"),
        ("image", None),
        ("mosaic", None),
    ]
    assert getattr(field.selfcal_state, expected_flag) is True
    assert field.cycle_number == 1


def test_collect_pipeline_features_describes_strategy_and_runtime_options():
    parset = _pipeline_parset(
        generate_initial_skymodel=True,
        ntimes_to_repeat_final_cycle=1,
        imaging_specific={
            "skip_final_major_iteration": True,
            "shared_facet_rw": "True",
            "use_mpi": "True",
        },
    )
    field = RecordingField(
        dde_mode="hybrid",
        epoch_observations=[["obs-a", "obs-b"]],
        make_quv_images=True,
        disable_iquv_clean=True,
        save_image_cube=True,
        use_mpi=True,
        parset=parset,
    )
    steps = [
        _single_step(
            {"dd": ["fast_phase"], "di": ["full_jones"]},
            do_predict=True,
            do_normalize=True,
            do_check=True,
            peel_outliers=True,
            peel_bright_sources=True,
        ),
        _image_step(),
    ]

    features = collect_pipeline_features(field, steps, parset)

    assert {
        "calibration",
        "calibration_dd",
        "calibration_di",
        "calibration_dd_then_di",
        "clean_disabled_full_stokes",
        "concatenate",
        "final_cycle",
        "full_stokes_imaging",
        "hybrid_screens",
        "image",
        "image_cube",
        "initial_skymodel",
        "mpi_wsclean",
        "normalize",
        "peel_bright_sources",
        "peel_outliers",
        "predict_dd",
        "repeat_final_cycle",
        "selfcal",
        "selfcal_check",
        "shared_facet_rw",
        "solve_dd_fast_phase",
        "solve_di_full_jones",
    } <= features


def test_supported_merge_feature_matrix_entries_pass_process_preflight():
    matrix = _merge_feature_matrix()

    assert matrix["supported"], "The supported merge matrix must not be empty"
    for entry in matrix["supported"]:
        features = set(entry["features"])
        fixture_paths = [REPO_ROOT / fixture_ref for fixture_ref in entry["fixture_refs"]]

        assert features <= SUPPORTED_PIPELINE_FEATURES, entry["id"]
        assert fixture_paths, entry["id"]
        assert all(path.exists() for path in fixture_paths), entry["id"]
        assert entry["test_types"], entry["id"]

        preflight_execution(
            _execution_config_from_matrix(entry),
            requested_features=features,
            supported_features=SUPPORTED_PIPELINE_FEATURES,
        )


def test_deferred_merge_feature_matrix_entries_fail_process_preflight():
    matrix = _merge_feature_matrix()

    assert matrix["deferred"], "The deferred merge matrix must not be empty"
    for entry in matrix["deferred"]:
        fixture_paths = [REPO_ROOT / fixture_ref for fixture_ref in entry["fixture_refs"]]

        assert fixture_paths, entry["id"]
        assert all(path.exists() for path in fixture_paths), entry["id"]
        assert entry["expected_issue_codes"], entry["id"]
        assert entry["test_types"], entry["id"]

        with pytest.raises(PreflightError) as exc:
            preflight_execution(
                _execution_config_from_matrix(entry),
                requested_features=entry["features"],
                supported_features=SUPPORTED_PIPELINE_FEATURES,
            )

        assert [issue.code for issue in exc.value.issues] == entry["expected_issue_codes"]


def test_run_pipeline_preflight_rejects_unsupported_feature_before_operations():
    parset = _pipeline_parset()
    lifecycle = RecordingPipelineLifecycle(
        parset=parset,
        strategy_steps=[_image_step(do_normalize=True)],
        supported_features={"final_cycle", "image"},
    )

    with pytest.raises(PreflightError) as exc:
        run_pipeline(
            "input.parset",
            operation_factories=RECORDING_FACTORIES,
            lifecycle_hooks=lifecycle.hooks(),
            execution_config=ExecutionConfig(task_runner="sync"),
        )

    assert exc.value.issues[0].code == "unsupported_feature"
    assert "normalize" in exc.value.issues[0].message
    assert "normalize" in lifecycle.preflight_features
    assert [
        event["operation"]
        for event in lifecycle.field.events
        if event["operation"] in {"image_normalize", "image", "mosaic"}
    ] == []


def test_run_pipeline_preflight_rejects_runtime_before_operations():
    parset = _pipeline_parset()
    lifecycle = RecordingPipelineLifecycle(
        parset=parset,
        strategy_steps=[_image_step()],
    )

    with pytest.raises(PreflightError) as exc:
        run_pipeline(
            "input.parset",
            operation_factories=RECORDING_FACTORIES,
            lifecycle_hooks=lifecycle.hooks(),
            execution_config=ExecutionConfig(task_runner="sync", use_container=True),
        )

    assert exc.value.issues[0].code == "unsupported_container"
    assert [
        event["operation"]
        for event in lifecycle.field.events
        if event["operation"] in {"image", "mosaic"}
    ] == []


def test_run_pipeline_preflight_supports_shared_facet_rw():
    parset = _pipeline_parset(
        imaging_specific={"skip_final_major_iteration": True, "shared_facet_rw": "True"}
    )
    lifecycle = RecordingPipelineLifecycle(
        parset=parset,
        strategy_steps=[_image_step()],
    )

    field = run_pipeline(
        "input.parset",
        operation_factories=RECORDING_FACTORIES,
        lifecycle_hooks=lifecycle.hooks(),
        execution_config=ExecutionConfig(task_runner="sync"),
    )

    assert field is lifecycle.field
    assert "shared_facet_rw" in lifecycle.preflight_features
    assert [
        event["operation"] for event in field.events if event["operation"] in {"image", "mosaic"}
    ] == ["image", "mosaic"]


def test_run_pipeline_syncs_effective_execution_config_to_operation_parset():
    parset = _pipeline_parset(
        cluster_specific={
            "prefect_task_runner": "local_dask",
            "dask_scheduler": None,
            "dask_dashboard_address": None,
        }
    )
    lifecycle = RecordingPipelineLifecycle(
        parset=parset,
        strategy_steps=[_image_step()],
    )
    execution_config = ExecutionConfig(
        task_runner="external_dask",
        dask_scheduler="tcp://127.0.0.1:8786",
        dask_dashboard_address=":8787",
        local_dask_workers=2,
        cpus_per_task=4,
    )

    field = run_pipeline(
        "input.parset",
        operation_factories=RECORDING_FACTORIES,
        lifecycle_hooks=lifecycle.hooks(),
        execution_config=execution_config,
    )

    assert field.parset["cluster_specific"]["prefect_task_runner"] == "external_dask"
    assert field.parset["cluster_specific"]["dask_scheduler"] == "tcp://127.0.0.1:8786"
    assert field.parset["cluster_specific"]["dask_dashboard_address"] == ":8787"
    assert field.parset["cluster_specific"]["local_dask_workers"] == 2
    assert field.parset["cluster_specific"]["cpus_per_task"] == 4


def test_run_pipeline_lifecycle_runs_initial_selfcal_and_repeated_final_cycles():
    parset = _pipeline_parset(
        generate_initial_skymodel=True,
        ntimes_to_repeat_final_cycle=1,
    )
    strategy_steps = [
        _single_step(
            {"di": ["fast_phase"]},
            peel_outliers=True,
            peel_bright_sources=False,
        ),
        _image_step(),
    ]
    lifecycle = RecordingPipelineLifecycle(
        parset=parset,
        strategy_steps=strategy_steps,
        epoch_observations=[["obs-a", "obs-b"]],
    )

    field = run_pipeline(
        "input.parset",
        logging_level="debug",
        operation_factories=RECORDING_FACTORIES,
        lifecycle_hooks=lifecycle.hooks(),
    )

    assert field is lifecycle.field
    assert lifecycle.read_files == ["input.parset"]
    assert lifecycle.logging_levels == ["debug"]
    assert field.full_field_radius == parset["generate_initial_skymodel_radius"]
    assert field.do_final is True
    assert field.cycle_number == 3

    operation_events = [
        event
        for event in field.events
        if event["operation"]
        in {
            "concatenate",
            "image_initial",
            "predict",
            "calibrate",
            "image",
            "mosaic",
        }
    ]
    assert [
        (event["operation"], event.get("mode"), event.get("index")) for event in operation_events
    ] == [
        ("concatenate", None, 1),
        ("image_initial", None, None),
        ("predict", "di", 1),
        ("calibrate", "di", 1),
        ("image", None, 1),
        ("mosaic", None, 1),
        ("image", None, 2),
        ("mosaic", None, 2),
        ("image", None, 3),
        ("mosaic", None, 3),
    ]

    chunk_events = [event for event in field.events if event["operation"] == "chunk"]
    assert [(event["step_count"], event["data_fraction"]) for event in chunk_events] == [
        (0, parset["generate_initial_skymodel_data_fraction"]),
        (1, parset["selfcal_data_fraction"]),
        (1, parset["final_data_fraction"]),
    ]
    assert field.events[-1]["operation"] == "report"


def test_run_pipeline_publishes_plot_artifacts_after_operations_and_report(monkeypatch):
    parset = _pipeline_parset(input_h5parm="input-solutions.h5")
    lifecycle = RecordingPipelineLifecycle(
        parset=parset,
        strategy_steps=[_image_step()],
    )

    def fake_publish_plot_artifacts(field, publish_index=True):
        field.events.append(
            {
                "operation": "publish_plot_artifacts",
                "publish_index": publish_index,
            }
        )
        return []

    monkeypatch.setattr(
        "rapthor.execution.pipeline.flow.publish_plot_artifacts_for_field",
        fake_publish_plot_artifacts,
    )

    field = run_pipeline(
        "input.parset",
        operation_factories=RECORDING_FACTORIES,
        lifecycle_hooks=lifecycle.hooks(),
    )

    assert [
        (event["operation"], event.get("publish_index"))
        for event in field.events
        if event["operation"] in {"image", "mosaic", "report", "publish_plot_artifacts"}
    ] == [
        ("image", None),
        ("publish_plot_artifacts", False),
        ("mosaic", None),
        ("publish_plot_artifacts", False),
        ("report", None),
        ("publish_plot_artifacts", True),
    ]


@pytest.mark.parametrize(
    "parset_updates, final_step_updates, expected_message",
    [
        pytest.param(
            {"input_h5parm": None},
            {},
            "no calibration solutions were provided",
            id="missing-input-h5parm",
        ),
        pytest.param(
            {"input_h5parm": "input-solutions.h5", "input_skymodel": None},
            {"peel_outliers": True},
            "sky model was provided",
            id="peel-outliers-without-input-skymodel",
        ),
        pytest.param(
            {"input_h5parm": "input-solutions.h5", "input_skymodel": None},
            {"peel_bright_sources": True},
            "sky model was provided",
            id="peel-bright-sources-without-input-skymodel",
        ),
    ],
)
def test_run_pipeline_final_only_validation_failures(
    parset_updates, final_step_updates, expected_message
):
    parset = _pipeline_parset(**parset_updates)
    strategy_steps = [_image_step(**final_step_updates)]

    with pytest.raises(ValueError, match=expected_message) as prefect_exc:
        _run_prefect_pipeline(
            copy.deepcopy(parset),
            copy.deepcopy(strategy_steps),
        )

    assert expected_message in str(prefect_exc.value)


def test_run_pipeline_skips_initial_skymodel_generation_without_calibration_step():
    parset = _pipeline_parset(generate_initial_skymodel=True)
    lifecycle = RecordingPipelineLifecycle(
        parset=parset,
        strategy_steps=[_image_step()],
    )

    field = run_pipeline(
        "input.parset",
        operation_factories=RECORDING_FACTORIES,
        lifecycle_hooks=lifecycle.hooks(),
    )

    assert field is lifecycle.field
    assert field.parset["generate_initial_skymodel"] is False
    assert not hasattr(field, "full_field_radius")
    assert [
        event["operation"]
        for event in field.events
        if event["operation"] in {"define_full_field_sector", "image_initial"}
    ] == []
    assert [
        event["operation"] for event in field.events if event["operation"] in {"image", "mosaic"}
    ] == ["image", "mosaic"]


def test_pipeline_flow_runs_no_selfcal_image_only_strategy():
    parset = _pipeline_parset(input_h5parm="input-solutions.h5")
    lifecycle = RecordingPipelineLifecycle(
        parset=parset,
        strategy_steps=[_image_step()],
    )

    with prefect_test_harness():
        field = pipeline_flow(
            "input.parset",
            operation_factories=RECORDING_FACTORIES,
            lifecycle_hooks=lifecycle.hooks(),
            execution_config=ExecutionConfig(task_runner="sync"),
        )

    assert field is lifecycle.field
    assert field.final_seen is True
    assert field.cycle_number == 1
    assert parset["generate_initial_skymodel"] is False
    assert parset["download_initial_skymodel"] is False
    assert [
        (event["operation"], event.get("mode"), event.get("index"))
        for event in field.events
        if event["operation"] in {"predict", "calibrate", "image", "mosaic"}
    ] == [
        ("image", None, 1),
        ("mosaic", None, 1),
    ]


def test_run_pipeline_rejects_image_only_final_without_input_h5parm():
    parset = _pipeline_parset(input_h5parm=None)
    lifecycle = RecordingPipelineLifecycle(
        parset=parset,
        strategy_steps=[_image_step()],
    )

    with pytest.raises(ValueError, match="no calibration solutions were provided"):
        run_pipeline(
            "input.parset",
            operation_factories=RECORDING_FACTORIES,
            lifecycle_hooks=lifecycle.hooks(),
        )


def test_pipeline_flow_rejects_image_only_peeling_without_input_skymodel():
    parset = _pipeline_parset(input_h5parm="input-solutions.h5", input_skymodel=None)
    lifecycle = RecordingPipelineLifecycle(
        parset=parset,
        strategy_steps=[_image_step(peel_outliers=True)],
    )

    with prefect_test_harness(), pytest.raises(ValueError, match="sky model was provided"):
        pipeline_flow(
            "input.parset",
            operation_factories=RECORDING_FACTORIES,
            lifecycle_hooks=lifecycle.hooks(),
            execution_config=ExecutionConfig(task_runner="sync"),
        )


def test_run_pipeline_final_hybrid_screens_skip_dd_predict_and_set_image_flags():
    parset = _pipeline_parset(
        input_h5parm="input-solutions.h5",
        input_skymodel="input.sky",
    )
    final_step = _image_step(do_predict=True, peel_outliers=True)
    lifecycle = RecordingPipelineLifecycle(
        parset=parset,
        strategy_steps=[final_step],
        field_updates={
            "dde_mode": "hybrid",
            "make_quv_images": True,
            "disable_iquv_clean": True,
            "save_image_cube": True,
            "image_cube_stokes_list": ["I", "Q", "U", "V", "XX"],
        },
    )

    field = run_pipeline(
        "input.parset",
        operation_factories=RECORDING_FACTORIES,
        lifecycle_hooks=lifecycle.hooks(),
        execution_config=ExecutionConfig(task_runner="sync"),
    )

    operation_events = [
        event for event in field.events if event["operation"] in {"predict", "image", "mosaic"}
    ]
    assert [(event["operation"], event["mode"]) for event in operation_events] == [
        ("image", None),
        ("mosaic", None),
    ]
    assert final_step["peel_outliers"] is False

    image_event = next(event for event in operation_events if event["operation"] == "image")
    assert image_event["image_pol"] == "IQUV"
    assert image_event["disable_clean"] is True
    assert image_event["make_image_cube"] is True
    assert image_event["image_cube_stokes_list"] == ["I", "Q", "U", "V"]
    assert image_event["generate_screens"] is True
    assert image_event["apply_screens"] is True
    assert image_event["skip_final_major_iteration"] is False
