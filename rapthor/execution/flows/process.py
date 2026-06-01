"""Top-level Prefect flow skeleton for Rapthor process orchestration."""

import logging
from dataclasses import dataclass
from typing import Callable, Optional

from prefect import flow

from rapthor.execution.config import ExecutionConfig
from rapthor.execution.flows.runtime import run_flow_with_task_runner
from rapthor.operations.calibrate import Calibrate
from rapthor.operations.image import Image, ImageNormalize
from rapthor.operations.mosaic import Mosaic
from rapthor.operations.predict import Predict
from rapthor.process import _do_calibrate_mode

log = logging.getLogger("rapthor")


@dataclass(frozen=True)
class ProcessOperationFactories:
    """Operation constructors used by the top-level process flow."""

    predict: Callable[[str, object, int], object]
    calibrate: Callable[[str, object, int], object]
    image: Callable[[object, int], object]
    mosaic: Callable[[object, int], object]
    image_normalize: Callable[[object, int], object]


def default_process_operation_factories() -> ProcessOperationFactories:
    """Return the production operation constructors."""
    return ProcessOperationFactories(
        predict=Predict,
        calibrate=Calibrate,
        image=Image,
        mosaic=Mosaic,
        image_normalize=ImageNormalize,
    )


def _run_operation(factory: Callable, *args) -> object:
    operation = factory(*args)
    operation.run()
    return operation


def run_process_steps(
    field: object,
    steps: list[dict],
    final: bool = False,
    operation_factories: Optional[ProcessOperationFactories] = None,
) -> object:
    """Run one group of process steps using injectable operation factories.

    This mirrors ``rapthor.process.run_steps`` while the migration branch builds
    the Prefect top-level flow. Production CLI routing remains with
    ``process.run()`` until the final cut-over.
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
