"""Top-level Prefect flow for Rapthor pipeline orchestration."""

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
from rapthor.execution.pipeline.plan import (
    SUPPORTED_PIPELINE_FEATURES,
    calibration_mode_flags,
    collect_pipeline_features,
)
from rapthor.execution.prefect_logging import publish_python_logs_to_prefect
from rapthor.execution.task_runner import run_flow_with_task_runner

log = logging.getLogger("rapthor")


@dataclass(frozen=True)
class PipelineOperationFactories:
    """Operation constructors used by the top-level pipeline flow."""

    predict: Callable[[str, object, int], object]
    calibrate: Callable[[str, object, int], object]
    image: Callable[[object, int], object]
    mosaic: Callable[[object, int], object]
    image_normalize: Callable[[object, int], object]
    concatenate: Optional[Callable[[object, int], object]] = None
    image_initial: Optional[Callable[[object], object]] = None


@dataclass(frozen=True)
class PipelineLifecycleHooks:
    """Top-level pipeline collaborators outside operation execution."""

    read_parset: Callable[[object], dict]
    set_logging_level: Callable[[str], None]
    build_field: Callable[[dict], object]
    set_strategy: Callable[[object], list[dict]]
    validate_strategy: Callable[[list[dict], dict], None]
    preflight_execution: Callable[[object, list[dict], ExecutionConfig, set[str]], None]
    chunk_observations: Callable[[object, list[dict], float], None]
    do_final_pass: Callable[[object, list[dict], dict], bool]
    make_report: Callable[[object], None]


def default_pipeline_operation_factories() -> PipelineOperationFactories:
    """Return the production operation constructors."""
    from rapthor.operations.calibrate.base import Calibrate
    from rapthor.operations.concatenate import Concatenate
    from rapthor.operations.image.base import Image
    from rapthor.operations.image.initial import ImageInitial
    from rapthor.operations.image.normalize import ImageNormalize
    from rapthor.operations.mosaic import Mosaic
    from rapthor.operations.predict import Predict

    return PipelineOperationFactories(
        predict=Predict,
        calibrate=Calibrate,
        image=Image,
        mosaic=Mosaic,
        image_normalize=ImageNormalize,
        concatenate=Concatenate,
        image_initial=ImageInitial,
    )


def default_pipeline_lifecycle_hooks() -> PipelineLifecycleHooks:
    """Return the production pipeline collaborators."""
    from rapthor import _logging
    from rapthor.execution.pipeline.lifecycle import chunk_observations, do_final_pass, make_report
    from rapthor.lib.field import Field
    from rapthor.lib.parset import parset_read
    from rapthor.lib.strategy import set_strategy, validate_strategy

    return PipelineLifecycleHooks(
        read_parset=parset_read,
        set_logging_level=_logging.set_level,
        build_field=Field,
        set_strategy=set_strategy,
        validate_strategy=validate_strategy,
        preflight_execution=run_pipeline_preflight,
        chunk_observations=chunk_observations,
        do_final_pass=do_final_pass,
        make_report=make_report,
    )


def run_pipeline_preflight(
    field: object,
    strategy_steps: list[dict],
    execution_config: ExecutionConfig,
    requested_features: set[str],
) -> None:
    """Run the default pipeline-level preflight."""
    preflight_execution(
        execution_config,
        requested_features=requested_features,
        supported_features=SUPPORTED_PIPELINE_FEATURES,
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


def _sync_execution_config_to_parset(parset: dict, execution_config: ExecutionConfig) -> None:
    """Write effective runtime settings back into the operation parset."""
    cluster = parset.setdefault("cluster_specific", {})
    cluster.update(
        {
            "prefect_task_runner": execution_config.task_runner,
            "prefect_api_mode": execution_config.prefect_api_mode,
            "prefect_api_url": execution_config.prefect_api_url,
            "dask_scheduler": execution_config.dask_scheduler,
            "dask_dashboard_address": execution_config.dask_dashboard_address,
            "prefect_stream_output": execution_config.stream_output,
            "prefect_retries": execution_config.retries,
            "prefect_log_commands": execution_config.log_commands,
            "prefect_command_profile": execution_config.command_profile,
            "prefect_publish_fits_previews": execution_config.publish_fits_previews,
            "prefect_publish_postage_stamp_previews": (
                execution_config.publish_postage_stamp_previews
            ),
            "prefect_postage_stamp_preview_count": execution_config.postage_stamp_preview_count,
            "prefect_postage_stamp_preview_size_px": (
                execution_config.postage_stamp_preview_size_px
            ),
            "prefect_fits_preview_clip_percentile": (execution_config.fits_preview_clip_percentile),
            "batch_system": execution_config.batch_system,
            "max_nodes": execution_config.max_nodes,
            "local_dask_workers": execution_config.local_dask_workers,
            "cpus_per_task": execution_config.cpus_per_task,
            "mem_per_node_gb": execution_config.mem_per_node_gb,
            "use_container": execution_config.use_container,
            "container_type": execution_config.container_type,
            "local_scratch_dir": execution_config.local_scratch_dir,
            "global_scratch_dir": execution_config.global_scratch_dir,
        }
    )


def run_pipeline(
    parset_file: object,
    logging_level: str = "info",
    operation_factories: Optional[PipelineOperationFactories] = None,
    lifecycle_hooks: Optional[PipelineLifecycleHooks] = None,
    execution_config: Optional[ExecutionConfig] = None,
) -> Optional[object]:
    """Run the top-level Rapthor pipeline lifecycle with injectable hooks.

    This implements the CLI pipeline lifecycle while keeping collaborators
    injectable for pipeline-flow tests.
    """
    factories = operation_factories or default_pipeline_operation_factories()
    hooks = lifecycle_hooks or default_pipeline_lifecycle_hooks()

    parset = hooks.read_parset(parset_file)
    config = execution_config or ExecutionConfig.from_parset(parset)
    _sync_execution_config_to_parset(parset, config)

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
    requested_features = collect_pipeline_features(field, strategy_steps, parset)
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
        run_pipeline_steps(field, selfcal_steps, operation_factories=factories)

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

            run_pipeline_steps(field, [final_step], final=True, operation_factories=factories)

    hooks.make_report(field)
    publish_plot_artifacts_for_field(field)
    publish_fits_image_artifacts_for_field(field)
    publish_command_metrics_artifact_for_field(field)
    log.info("Rapthor has finished :)")
    return field


def run_pipeline_steps(
    field: object,
    steps: list[dict],
    final: bool = False,
    operation_factories: Optional[PipelineOperationFactories] = None,
) -> object:
    """Run one group of pipeline steps using injectable operation factories.

    This preserves pipeline-step operation ordering for pipeline-flow tests.
    """
    factories = operation_factories or default_pipeline_operation_factories()
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
            field.make_residual_visibilities = field.save_residual_visibilities and final
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


@flow(name="pipeline")
def _pipeline_flow(
    parset_file: object,
    logging_level: str = "info",
    operation_factories: Optional[PipelineOperationFactories] = None,
    lifecycle_hooks: Optional[PipelineLifecycleHooks] = None,
    execution_config: Optional[ExecutionConfig] = None,
):
    """Prefect implementation for top-level pipeline orchestration."""
    with publish_python_logs_to_prefect(logging_level):
        return run_pipeline(
            parset_file,
            logging_level=logging_level,
            operation_factories=operation_factories,
            lifecycle_hooks=lifecycle_hooks,
            execution_config=execution_config,
        )


def pipeline_flow(
    parset_file: object,
    logging_level: str = "info",
    operation_factories: Optional[PipelineOperationFactories] = None,
    lifecycle_hooks: Optional[PipelineLifecycleHooks] = None,
    execution_config: Optional[ExecutionConfig] = None,
):
    """Prefect entry point for top-level pipeline orchestration."""
    return run_flow_with_task_runner(
        _pipeline_flow,
        parset_file,
        logging_level=logging_level,
        operation_factories=operation_factories,
        lifecycle_hooks=lifecycle_hooks,
        execution_config=execution_config,
    )
