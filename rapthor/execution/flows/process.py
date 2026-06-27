"""Top-level Prefect flow skeleton for Rapthor process orchestration."""

import logging
from dataclasses import dataclass
from typing import Callable, Optional

from prefect import flow

from rapthor.execution.artifacts import (
    publish_command_metrics_artifact_for_field,
    publish_fits_image_artifacts_for_field,
    publish_plot_artifacts_for_field,
)
from rapthor.execution.capabilities import preflight_execution
from rapthor.execution.config import ExecutionConfig
from rapthor.execution.flows.runtime import run_flow_with_task_runner
from rapthor.execution.prefect_logging import publish_python_logs_to_prefect
from rapthor.execution.process_plan import (
    SUPPORTED_PROCESS_FEATURES,
    calibration_mode_flags,
    collect_process_features,
)

log = logging.getLogger("rapthor")


@dataclass(frozen=True)
class ProcessOperationFactories:
    """Operation constructors used by the top-level process flow."""

    predict: Callable[[str, object, int], object]
    calibrate: Callable[[str, object, int], object]
    image: Callable[[object, int], object]
    mosaic: Callable[[object, int], object]
    image_normalize: Callable[[object, int], object]
    concatenate: Optional[Callable[[object, int], object]] = None
    image_initial: Optional[Callable[[object], object]] = None


@dataclass(frozen=True)
class ProcessLifecycleHooks:
    """Top-level process collaborators outside operation execution."""

    read_parset: Callable[[object], dict]
    set_logging_level: Callable[[str], None]
    build_field: Callable[[dict], object]
    set_strategy: Callable[[object], list[dict]]
    validate_strategy: Callable[[list[dict], dict], None]
    preflight_execution: Callable[[object, list[dict], ExecutionConfig, set[str]], None]
    chunk_observations: Callable[[object, list[dict], float], None]
    do_final_pass: Callable[[object, list[dict], dict], bool]
    make_report: Callable[[object], None]


def default_process_operation_factories() -> ProcessOperationFactories:
    """Return the production operation constructors."""
    from rapthor.operations.calibrate import Calibrate
    from rapthor.operations.concatenate import Concatenate
    from rapthor.operations.image import Image, ImageInitial, ImageNormalize
    from rapthor.operations.mosaic import Mosaic
    from rapthor.operations.predict import Predict

    return ProcessOperationFactories(
        predict=Predict,
        calibrate=Calibrate,
        image=Image,
        mosaic=Mosaic,
        image_normalize=ImageNormalize,
        concatenate=Concatenate,
        image_initial=ImageInitial,
    )


def default_process_lifecycle_hooks() -> ProcessLifecycleHooks:
    """Return the production process collaborators."""
    from rapthor import _logging
    from rapthor.lib.field import Field
    from rapthor.lib.parset import parset_read
    from rapthor.lib.strategy import set_strategy, validate_strategy
    from rapthor.process import chunk_observations, do_final_pass, make_report

    return ProcessLifecycleHooks(
        read_parset=parset_read,
        set_logging_level=_logging.set_level,
        build_field=Field,
        set_strategy=set_strategy,
        validate_strategy=validate_strategy,
        preflight_execution=run_process_preflight,
        chunk_observations=chunk_observations,
        do_final_pass=do_final_pass,
        make_report=make_report,
    )


def run_process_preflight(
    field: object,
    strategy_steps: list[dict],
    execution_config: ExecutionConfig,
    requested_features: set[str],
) -> None:
    """Run the default process-level preflight."""
    preflight_execution(
        execution_config,
        requested_features=requested_features,
        supported_features=SUPPORTED_PROCESS_FEATURES,
    )


def _do_calibrate_mode(strategy: dict) -> dict[str, bool]:
    return calibration_mode_flags(strategy)


def _run_operation(factory: Callable, *args) -> object:
    operation = factory(*args)
    operation.run()
    publish_plot_artifacts_for_field(getattr(operation, "field", None), publish_index=False)
    publish_fits_image_artifacts_for_field(getattr(operation, "field", None))
    publish_command_metrics_artifact_for_field(getattr(operation, "field", None))
    return operation


def _run_required_operation(factory: Optional[Callable], name: str, *args) -> object:
    if factory is None:
        raise ValueError(f"No operation factory configured for {name!r}")
    return _run_operation(factory, *args)


def run_process(
    parset_file: object,
    logging_level: str = "info",
    operation_factories: Optional[ProcessOperationFactories] = None,
    lifecycle_hooks: Optional[ProcessLifecycleHooks] = None,
    execution_config: Optional[ExecutionConfig] = None,
) -> Optional[object]:
    """Run the top-level Rapthor process lifecycle with injectable hooks.

    This implements the public ``rapthor.process.run`` lifecycle while keeping
    collaborators injectable for process-flow tests.
    """
    factories = operation_factories or default_process_operation_factories()
    hooks = lifecycle_hooks or default_process_lifecycle_hooks()

    parset = hooks.read_parset(parset_file)
    config = execution_config or ExecutionConfig.from_parset(parset)

    log.info("Setting log level to %s", logging_level.upper())
    hooks.set_logging_level(logging_level)

    field = hooks.build_field(parset)
    needs_concatenation = any(len(obs) > 1 for obs in field.epoch_observations)
    if needs_concatenation:
        hooks.preflight_execution(field, [], config, {"concatenate"})
        log.info(
            "MS files with different frequencies found for one or more epochs. "
            "Concatenation over frequency will be done."
        )
        _run_required_operation(factories.concatenate, "concatenate", field, 1)

    strategy_steps = hooks.set_strategy(field)
    if strategy_steps:
        selfcal_steps = strategy_steps[:-1]
        final_step = strategy_steps[-1]
    else:
        log.warning(
            "The strategy %r does not define any processing steps. No processing can be done.",
            parset["strategy"],
        )
        return None

    hooks.validate_strategy(strategy_steps, parset)
    requested_features = collect_process_features(field, strategy_steps, parset)
    if needs_concatenation:
        requested_features.add("concatenate")
    hooks.preflight_execution(field, strategy_steps, config, requested_features)

    if parset["generate_initial_skymodel"]:
        if not any(step["do_calibrate"] for step in strategy_steps):
            log.warning(
                "Generation of an initial sky model has been activated but "
                "the strategy %r does not contain any calibration steps. "
                "Skipping the initial skymodel generation...",
                parset["strategy"],
            )
            field.parset["generate_initial_skymodel"] = False
        else:
            field.define_full_field_sector(radius=parset["generate_initial_skymodel_radius"])
            log.info("Imaging full field to generate an initial sky model...")
            hooks.chunk_observations(field, [], parset["generate_initial_skymodel_data_fraction"])
            _run_required_operation(factories.image_initial, "image_initial", field)

    if selfcal_steps:
        log.info(
            "Starting self calibration with a data fraction of %.2f",
            parset["selfcal_data_fraction"],
        )
        hooks.chunk_observations(field, selfcal_steps, parset["selfcal_data_fraction"])
        run_process_steps(field, selfcal_steps, operation_factories=factories)

    field.do_final = hooks.do_final_pass(field, selfcal_steps, final_step)
    if field.do_final:
        for index in range(parset["ntimes_to_repeat_final_cycle"] + 1):
            if selfcal_steps:
                final_step["peel_outliers"] = selfcal_steps[0]["peel_outliers"]
                log.info(
                    "Starting final cycle with a data fraction of %.2f",
                    parset["final_data_fraction"],
                )
                field.cycle_number += 1
            else:
                if not final_step["do_calibrate"]:
                    if not parset["input_h5parm"]:
                        raise ValueError(
                            "The strategy indicates that no calibration is to be done "
                            "but no calibration solutions were provided. Please provide "
                            "the solutions with the input_h5parm parameter"
                        )
                    if (
                        final_step["peel_outliers"] or final_step["peel_bright_sources"]
                    ) and not parset["input_skymodel"]:
                        raise ValueError(
                            "Peeling of outliers or bright sources was activated but no "
                            "sky model was provided. Please provide a sky model with the "
                            "input_skymodel parameter"
                        )
                    field.parset["generate_initial_skymodel"] = False
                    field.parset["download_initial_skymodel"] = False
                log.info("Using a data fraction of %.2f", parset["final_data_fraction"])
                field.cycle_number = index + 1

            if field.make_quv_images:
                log.info("Stokes I, Q, U, and V images will be made")
            if field.dde_mode == "hybrid":
                log.info(
                    "Screens will be used for calibration and imaging (since dde_mode = "
                    "'hybrid' and this is the final iteration)"
                )
                field.generate_screens = True
                field.apply_screens = True
                if final_step["peel_outliers"]:
                    log.warning(
                        "Peeling of outliers is currently not supported when using "
                        "screens. Peeling will be skipped."
                    )
                    final_step["peel_outliers"] = False

            if index == 0:
                hooks.chunk_observations(field, [final_step], parset["final_data_fraction"])

            run_process_steps(field, [final_step], final=True, operation_factories=factories)

    hooks.make_report(field)
    publish_plot_artifacts_for_field(field)
    publish_fits_image_artifacts_for_field(field)
    publish_command_metrics_artifact_for_field(field)
    log.info("Rapthor has finished :)")
    return field


def run_process_steps(
    field: object,
    steps: list[dict],
    final: bool = False,
    operation_factories: Optional[ProcessOperationFactories] = None,
) -> object:
    """Run one group of process steps using injectable operation factories.

    This preserves the public ``rapthor.process.run_steps`` operation ordering
    for process-flow and compatibility tests.
    """
    factories = operation_factories or default_process_operation_factories()
    selfcal_state = None
    cycle_number = getattr(field, "cycle_number")

    for index, step in enumerate(steps):
        cycle_number = index + field.cycle_number
        field.update(step, cycle_number, final=final)

        if field.do_calibrate:
            field.generate_screens = (field.dde_mode == "hybrid") and final
            for mode, enabled in _do_calibrate_mode(field.calibration_strategy).items():
                if not enabled:
                    continue
                if mode == "di":
                    _run_operation(factories.predict, "di", field, cycle_number)
                _run_operation(factories.calibrate, mode, field, cycle_number)

        if field.do_predict and not field.generate_screens:
            _run_operation(factories.predict, "dd", field, cycle_number)

        if field.do_image:
            if field.do_normalize:
                field.define_normalize_sector()
                field.image_pol = "I"
                _run_operation(factories.image_normalize, field, cycle_number)

            field.image_pol = "IQUV" if (field.make_quv_images and final) else "I"
            field.disable_clean = field.image_pol == "IQUV" and field.disable_iquv_clean
            field.make_image_cube = field.save_image_cube and final
            field.image_cube_stokes_list = [
                pol for pol in field.image_cube_stokes_list if pol.upper() in field.image_pol
            ]
            field.apply_screens = (field.dde_mode == "hybrid") and final
            field.skip_final_major_iteration = (
                False if final else field.parset["imaging_specific"]["skip_final_major_iteration"]
            )

            _run_operation(factories.image, field, cycle_number)
            _run_operation(factories.mosaic, field, cycle_number)

        if field.do_check and not final:
            log.info("Checking selfcal convergence...")
            selfcal_state = field.check_selfcal_progress()
            if not any(selfcal_state):
                log.info(
                    "Improvement in image noise, dynamic range, and/or number "
                    "of sources exceeds that set by the convergence ratio of %.2f.",
                    field.convergence_ratio,
                )
                log.info("Continuing selfcal...")
            else:
                if selfcal_state.converged:
                    log.info(
                        "Selfcal has converged (improvement in image noise, "
                        "dynamic range, and number of sources does not exceed "
                        "that set by the convergence ratio of %.2f)",
                        field.convergence_ratio,
                    )
                if selfcal_state.diverged:
                    log.warning(
                        "Selfcal has diverged (ratio of current image noise "
                        "to previous value is > %.2f)",
                        field.divergence_ratio,
                    )
                if selfcal_state.failed:
                    log.warning(
                        "Selfcal has failed due to high noise (ratio of current "
                        "image noise to theoretical value is > %.2f)",
                        field.failure_ratio,
                    )
                log.info("Stopping selfcal at cycle %i of %i", cycle_number, len(steps))
                break
        else:
            selfcal_state = None

    field.selfcal_state = selfcal_state
    field.cycle_number = cycle_number
    return field


@flow(name="process_steps")
def _process_steps_flow(
    field: object,
    steps: list[dict],
    final: bool = False,
    operation_factories: Optional[ProcessOperationFactories] = None,
    execution_config: Optional[ExecutionConfig] = None,
):
    """Prefect implementation for top-level process-step orchestration."""
    with publish_python_logs_to_prefect():
        return run_process_steps(
            field,
            steps,
            final=final,
            operation_factories=operation_factories,
        )


def process_steps_flow(
    field: object,
    steps: list[dict],
    final: bool = False,
    operation_factories: Optional[ProcessOperationFactories] = None,
    execution_config: Optional[ExecutionConfig] = None,
):
    """Prefect entry point for top-level process-step orchestration."""
    return run_flow_with_task_runner(
        _process_steps_flow,
        field,
        steps,
        final=final,
        operation_factories=operation_factories,
        execution_config=execution_config,
    )


@flow(name="process")
def _process_flow(
    parset_file: object,
    logging_level: str = "info",
    operation_factories: Optional[ProcessOperationFactories] = None,
    lifecycle_hooks: Optional[ProcessLifecycleHooks] = None,
    execution_config: Optional[ExecutionConfig] = None,
):
    """Prefect implementation for top-level process orchestration."""
    with publish_python_logs_to_prefect(logging_level):
        return run_process(
            parset_file,
            logging_level=logging_level,
            operation_factories=operation_factories,
            lifecycle_hooks=lifecycle_hooks,
            execution_config=execution_config,
        )


def process_flow(
    parset_file: object,
    logging_level: str = "info",
    operation_factories: Optional[ProcessOperationFactories] = None,
    lifecycle_hooks: Optional[ProcessLifecycleHooks] = None,
    execution_config: Optional[ExecutionConfig] = None,
):
    """Prefect entry point for top-level process orchestration."""
    return run_flow_with_task_runner(
        _process_flow,
        parset_file,
        logging_level=logging_level,
        operation_factories=operation_factories,
        lifecycle_hooks=lifecycle_hooks,
        execution_config=execution_config,
    )
