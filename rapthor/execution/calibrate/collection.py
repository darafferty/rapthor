"""Calibration solution collection, plotting, and combination helpers."""

import glob
import os
from typing import Mapping, Optional

from rapthor.execution.artifacts import publish_plot_file_records
from rapthor.execution.calibrate.commands import (
    build_collect_h5parms_command,
    build_plot_solutions_command,
)
from rapthor.execution.calibrate.gain_processing import process_gain_solutions
from rapthor.execution.calibrate.h5parm_combination import combine_h5parms
from rapthor.execution.calibrate.h5parm_sources import adjust_h5parm_source_coordinates
from rapthor.execution.calibrate.screen_h5parms import collect_screen_h5parms
from rapthor.execution.config import ExecutionConfig
from rapthor.execution.outputs import require_file
from rapthor.execution.shell import run_external_command
from rapthor.lib.records import file_record, validate_output_record

PHASE_SOLVE_TYPES = {"fast_phase", "medium_phase"}


def process_plot_and_combine_collected_products(
    payload: Mapping[str, object],
    collected_products: list[Mapping[str, object]],
    execution_config: ExecutionConfig,
    shell_operation_cls=None,
) -> dict:
    """Process, plot, combine, and validate already collected solve h5parms."""
    processed_products = [
        process_collected_solve_product(payload, product) for product in collected_products
    ]
    plot_products = [
        plot_processed_solve_product(
            payload,
            product,
            execution_config,
            shell_operation_cls=shell_operation_cls,
        )
        for product in processed_products
    ]
    if needs_solution_combination(payload):
        active_solution = combine_processed_solution_products(payload, processed_products)
    else:
        active_solution = processed_products[active_solution_product_index(payload)]
    return finalize_processed_solution_products(
        payload,
        processed_products,
        plot_products,
        active_solution,
    )


def collect_plot_and_combine(
    payload: Mapping[str, object],
    solve_records: list[dict],
    execution_config: ExecutionConfig,
    shell_operation_cls=None,
) -> dict:
    """Collect, plot, process, and combine calibration solution products."""
    return _collect_strategy_solve_products(
        payload,
        solve_records,
        execution_config,
        shell_operation_cls=shell_operation_cls,
    )


def collect_screen_solutions(
    payload: Mapping[str, object],
    screen_records: list[dict],
) -> dict:
    """Collect screen-generation h5parm outputs into the final solution file."""
    combined_record = _run_collect_screen_h5parms(
        screen_records,
        payload["combined_h5parm"],
    )
    result = {"combined_solutions": combined_record}
    validate_output_record(result["combined_solutions"])
    return result


def collect_strategy_solve_h5parm(
    payload: Mapping[str, object],
    solve_records: list[dict],
    solve_slot: Mapping[str, object],
    execution_config: ExecutionConfig,
    shell_operation_cls=None,
) -> dict:
    """Collect one solve slot's per-chunk h5parm outputs into one h5parm file."""
    pipeline_working_dir = str(payload["pipeline_working_dir"])
    solve_key = _solve_key(solve_slot)
    collected_h5parms = payload["collected_h5parms"]
    collected_record = _run_collect_h5parm(
        [record[solve_key] for record in solve_records],
        collected_h5parms[solve_key],
        pipeline_working_dir,
        execution_config,
        f"Collected {_mode_label(payload)} {solve_key} {_solve_type(solve_slot)} h5parm",
        shell_operation_cls=shell_operation_cls,
    )
    return {
        "solve_key": solve_key,
        "solve_slot": dict(solve_slot),
        "collected_record": collected_record,
    }


def process_collected_solve_product(
    payload: Mapping[str, object],
    collected_product: Mapping[str, object],
) -> dict:
    """Prepare a collected solve product for plotting and cross-solve combining."""
    solve_slot = dict(collected_product["solve_slot"])
    collected_record = collected_product["collected_record"]
    solve_type = _solve_type(solve_slot)

    if solve_type == "slow_gains":
        combine_record = _run_process_gains(collected_record, payload)
    elif solve_type == "full_jones":
        combine_record = _run_process_fulljones_gains(collected_record, payload)
    else:
        combine_record = collected_record

    return {
        "solve_key": str(collected_product["solve_key"]),
        "solve_slot": solve_slot,
        "solution_record": collected_record,
        "combine_record": combine_record,
    }


def plot_processed_solve_product(
    payload: Mapping[str, object],
    processed_product: Mapping[str, object],
    execution_config: ExecutionConfig,
    shell_operation_cls=None,
) -> dict:
    """Plot one processed solve product and return the generated plot records."""
    solve_slot = processed_product["solve_slot"]
    solve_type = _solve_type(solve_slot)
    combine_record = processed_product["combine_record"]
    pipeline_working_dir = str(payload["pipeline_working_dir"])
    plot_first_dir = _plot_first_direction(payload)
    plots = {}

    if solve_type == "slow_gains":
        for soltype in ("phase", "amplitude"):
            root = "slow_amplitude_" if soltype == "amplitude" else "slow_phase_"
            plots[_plot_output_key(solve_slot, soltype)] = run_plot_solutions(
                combine_record,
                soltype,
                pipeline_working_dir,
                execution_config,
                root=root,
                first_dir=plot_first_dir,
                shell_operation_cls=shell_operation_cls,
            )
    else:
        plots[_plot_output_key(solve_slot, "phase")] = run_plot_solutions(
            combine_record,
            "phase",
            pipeline_working_dir,
            execution_config,
            root=_phase_plot_root(solve_slot),
            first_dir=plot_first_dir,
            shell_operation_cls=shell_operation_cls,
        )

    return {"solve_key": str(processed_product["solve_key"]), "plots": plots}


def needs_solution_combination(payload: Mapping[str, object]) -> bool:
    """Return whether processed solve products need an h5parm combine step."""
    solve_slots = list(payload["chunks"][0]["solve_slots"])
    phase_slots = [slot for slot in solve_slots if _is_phase_solve(slot)]
    slow_slots = [slot for slot in solve_slots if _solve_type(slot) == "slow_gains"]
    return len(phase_slots) > 1 or (bool(phase_slots) and bool(slow_slots))


def active_solution_product_index(payload: Mapping[str, object]) -> int:
    """Return the active processed-product index when no combine step is needed."""
    solve_slots = list(payload["chunks"][0]["solve_slots"])
    solve_priority = (
        _is_phase_solve,
        lambda slot: _solve_type(slot) == "slow_gains",
        lambda slot: _solve_type(slot) == "full_jones",
    )
    for is_match in solve_priority:
        for index, solve_slot in enumerate(solve_slots):
            if is_match(solve_slot):
                return index
    raise ValueError("Calibration produced no active solution product")


def combine_processed_solution_products(
    payload: Mapping[str, object],
    processed_products: list[Mapping[str, object]],
) -> dict:
    """Combine processed solve products into the active h5parm."""
    solve_slots = list(payload["chunks"][0]["solve_slots"])
    combine_records = {
        str(product["solve_key"]): product["combine_record"] for product in processed_products
    }
    phase_slots = [slot for slot in solve_slots if _is_phase_solve(slot)]
    slow_slots = [slot for slot in solve_slots if _solve_type(slot) == "slow_gains"]
    fulljones_slots = [slot for slot in solve_slots if _solve_type(slot) == "full_jones"]

    phase_record = None
    if phase_slots:
        phase_record = combine_records[_solve_key(phase_slots[0])]
        for output_record, solve_slot in zip(
            _combine_phase_outputs(
                payload,
                len(phase_slots),
                has_slow_gain=bool(slow_slots),
            ),
            phase_slots[1:],
        ):
            phase_record = _run_combine_h5parms(
                phase_record,
                combine_records[_solve_key(solve_slot)],
                output_record,
                "p1p2_scalar",
                payload,
                f"Combined {_mode_label(payload)} phase h5parm",
            )

    if slow_slots:
        if len(slow_slots) > 1:
            raise ValueError("A calibration cycle can contain at most one slow_gains solve")
        slow_record = combine_records[_solve_key(slow_slots[0])]
        if phase_record is None:
            return slow_record
        final_output = payload["combined_h5parms"].get("final")
        if final_output is None:
            raise ValueError("Calibration final combination output is missing")
        return _run_combine_h5parms(
            phase_record,
            slow_record,
            final_output,
            _final_phase_plus_slow_gain_combine_mode(payload),
            payload,
            f"Combined {_mode_label(payload)} phase and slow-gain h5parm",
        )

    if phase_record is not None:
        return phase_record

    if fulljones_slots:
        return combine_records[_solve_key(fulljones_slots[0])]

    raise ValueError("Calibration produced no active solution product")


def finalize_processed_solution_products(
    payload: Mapping[str, object],
    processed_products: list[Mapping[str, object]],
    plot_products: list[Mapping[str, object]],
    active_solution: Mapping[str, object],
) -> dict:
    """Collect solution, plot, and active h5parm records into the flow output map."""
    result = {}
    for product in processed_products:
        result[_solution_output_key(product["solve_slot"])] = product["solution_record"]

    for plot_product in plot_products:
        result.update(plot_product["plots"])

    active_record = _active_solution_record(active_solution)
    phase_slots = [slot for slot in payload["chunks"][0]["solve_slots"] if _is_phase_solve(slot)]
    if _should_adjust_dd_sources(payload) and len(phase_slots) > 1:
        active_record = adjust_h5parm_sources(
            active_record,
            payload,
            f"Adjusted {_mode_label(payload)} combined h5parm",
        )

    result["combined_solutions"] = active_record
    for value in result.values():
        validate_output_record(value)
    return result


def run_plot_solutions(
    h5parm_record: Mapping[str, str],
    soltype: str,
    pipeline_working_dir: str,
    execution_config: ExecutionConfig,
    root: Optional[str] = None,
    first_dir: bool = False,
    shell_operation_cls=None,
) -> list[dict]:
    """Run solution plotting and return only newly created plot records."""
    before_plots = set(glob.glob(os.path.join(pipeline_working_dir, "*.png")))
    plot_command = build_plot_solutions_command(
        h5parm_record["path"], soltype, root=root, first_dir=first_dir
    )
    run_external_command(
        plot_command,
        pipeline_working_dir,
        execution_config,
        shell_operation_cls=shell_operation_cls,
    )
    after_plots = set(glob.glob(os.path.join(pipeline_working_dir, "*.png")))
    plot_records = [file_record(path) for path in sorted(after_plots - before_plots)]
    publish_plot_file_records(plot_records, pipeline_working_dir)
    return plot_records


def adjust_h5parm_sources(
    h5parm_record: Mapping[str, str],
    payload: Mapping[str, object],
    label: str,
) -> dict:
    """Adjust DD h5parm source names to match the calibration skymodel."""
    skymodel = payload.get("sourcedb")
    if not skymodel:
        raise ValueError("DD source adjustment requires calibration_skymodel_file")
    adjust_h5parm_source_coordinates(str(skymodel), h5parm_record["path"])
    return require_file(h5parm_record["path"], label)


def _run_collect_h5parm(
    input_records: list[dict],
    output: Mapping[str, object],
    pipeline_working_dir: str,
    execution_config: ExecutionConfig,
    label: str,
    shell_operation_cls=None,
) -> dict:
    collect_command = build_collect_h5parms_command(
        [record["path"] for record in input_records],
        str(output["filename"]),
    )
    run_external_command(
        collect_command,
        pipeline_working_dir,
        execution_config,
        shell_operation_cls=shell_operation_cls,
    )
    return require_file(str(output["path"]), label)


def _run_collect_screen_h5parms(
    input_records: list[dict],
    output: Mapping[str, object],
) -> dict:
    collect_screen_h5parms(
        [record["path"] for record in input_records],
        str(output["path"]),
        overwrite=True,
    )
    return require_file(str(output["path"]), "Combined screen h5parm")


def _plot_first_direction(payload: Mapping[str, object]) -> bool:
    return str(payload.get("mode")) == "di"


def _mode_label(payload: Mapping[str, object]) -> str:
    return str(payload.get("mode", "calibration")).upper()


def _run_combine_h5parms(
    input_record1: Mapping[str, str],
    input_record2: Mapping[str, str],
    output: Mapping[str, object],
    mode: str,
    payload: Mapping[str, object],
    label: str,
) -> dict:
    combine_h5parms(
        input_record1["path"],
        input_record2["path"],
        str(output["path"]),
        mode,
        reweight=False,
        cal_names=list(payload["calibrator_patch_names"]),
        cal_fluxes=list(payload["calibrator_fluxes"]),
    )
    return require_file(str(output["path"]), label)


def _run_process_gains(
    h5parm_record: Mapping[str, str],
    payload: Mapping[str, object],
) -> dict:
    process_gain_solutions(
        h5parm_record["path"],
        normalize=True,
        flag=True,
        smooth=True,
        max_station_delta=float(payload["max_normalization_delta"]),
        scale_delta_with_dist=payload["scale_normalization_delta"],
        phase_center=(payload["phase_center_ra"], payload["phase_center_dec"]),
    )
    return require_file(h5parm_record["path"], "Processed slow-gain h5parm")


def _run_process_fulljones_gains(
    h5parm_record: Mapping[str, str],
    payload: Mapping[str, object],
) -> dict:
    process_gain_solutions(
        h5parm_record["path"],
        normalize=True,
        flag=False,
        smooth=False,
        max_station_delta=float(payload["max_normalization_delta"]),
        scale_delta_with_dist=False,
        phase_center=(0.0, 0.0),
    )
    return require_file(h5parm_record["path"], "Processed full-Jones h5parm")


def _should_adjust_dd_sources(payload: Mapping[str, object]) -> bool:
    return str(payload.get("mode")) == "dd" and len(payload.get("calibrator_patch_names", [])) > 1


def _final_phase_plus_slow_gain_combine_mode(payload: Mapping[str, object]) -> str:
    """Return the combine mode for the final phase-plus-slow-gain h5parm."""
    if str(payload.get("mode")) == "dd":
        return str(payload.get("solution_combine_mode") or "p1p2a2_scalar")
    return "p1a2"


def _solve_key(solve_slot: Mapping[str, object]) -> str:
    return f"solve{solve_slot['slot']}"


def _solve_type(solve_slot: Mapping[str, object]) -> str:
    return str(solve_slot["solve_type"])


def _is_phase_solve(solve_slot: Mapping[str, object]) -> bool:
    return _solve_type(solve_slot) in PHASE_SOLVE_TYPES


def _solution_label(solve_slot: Mapping[str, object]) -> str:
    return str(solve_slot["solution_label"])


def _solution_output_key(solve_slot: Mapping[str, object]) -> str:
    solve_type = _solve_type(solve_slot)
    if solve_type == "fast_phase":
        return "fast_phase_solutions"
    if solve_type == "medium_phase":
        return f"{_solution_label(solve_slot)}_phase_solutions"
    if solve_type == "slow_gains":
        return "slow_gain_solutions"
    if solve_type == "full_jones":
        return "fulljones_solutions"
    return f"{_solve_key(solve_slot)}_solutions"


def _phase_plot_root(solve_slot: Mapping[str, object]) -> Optional[str]:
    if _solve_type(solve_slot) == "medium_phase":
        return f"{_solution_label(solve_slot)}_phase_"
    if _solve_type(solve_slot) == "full_jones":
        return "fulljones_phase_"
    return None


def _plot_output_key(solve_slot: Mapping[str, object], soltype: str) -> str:
    solve_type = _solve_type(solve_slot)
    if solve_type == "fast_phase":
        return "fast_phase_plots"
    if solve_type == "medium_phase":
        return f"{_solution_label(solve_slot)}_phase_plots"
    if solve_type == "slow_gains":
        return "slow_amp_plots" if soltype == "amplitude" else "slow_phase_plots"
    if solve_type == "full_jones":
        return "fulljones_phase_plots"
    return f"{_solve_key(solve_slot)}_{soltype}_plots"


def _combine_phase_outputs(
    payload: Mapping[str, object],
    phase_count: int,
    *,
    has_slow_gain: bool,
) -> list[Mapping[str, object]]:
    combined_h5parms = payload["combined_h5parms"]
    outputs = []
    for key in ("phase_1_2", "phase_1_2_3"):
        if key in combined_h5parms:
            outputs.append(combined_h5parms[key])
    if not has_slow_gain and "final" in combined_h5parms:
        outputs.append(combined_h5parms["final"])

    required = max(0, phase_count - 1)
    if len(outputs) < required:
        raise ValueError("Calibration phase combination outputs are incomplete")
    return outputs[:required]


def _collect_strategy_solve_products(
    payload: Mapping[str, object],
    solve_records: list[dict],
    execution_config: ExecutionConfig,
    shell_operation_cls=None,
) -> dict:
    solve_slots = list(payload["chunks"][0]["solve_slots"])
    collected_products = [
        collect_strategy_solve_h5parm(
            payload,
            solve_records,
            solve_slot,
            execution_config,
            shell_operation_cls=shell_operation_cls,
        )
        for solve_slot in solve_slots
    ]
    return process_plot_and_combine_collected_products(
        payload,
        collected_products,
        execution_config,
        shell_operation_cls=shell_operation_cls,
    )


def _active_solution_record(active_solution: Mapping[str, object]) -> dict:
    if "combine_record" in active_solution:
        return active_solution["combine_record"]
    return dict(active_solution)
