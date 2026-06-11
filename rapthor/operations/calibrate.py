"""
Module that holds the Calibrate classes
"""

import glob
import os
import shutil
from dataclasses import dataclass

import lsmtool
import numpy as np

from rapthor.execution.config import ExecutionConfig
from rapthor.execution.flows.calibrate import calibrate_flow, calibrate_payload_from_inputs
from rapthor.lib import miscellaneous as misc
from rapthor.lib.cwl import CWLDir, CWLFile
from rapthor.lib.operation import Operation

FIELD_PREFIX_BY_SOLVE = {
    "fast_phase": "fast",
    "medium_phase": "medium",
    "slow_gains": "slow",
    "full_jones": "fulljones",
}

MODE_BY_SOLVE = {
    "fast_phase": "scalarphase",
    "medium_phase": "scalarphase",
    "slow_gains": "diagonal",
    "full_jones": "fulljones",
}

INTERVAL_KEYS_BY_SOLVE = {
    "fast_phase": ("solint_fast_timestep", "solint_fast_freqstep"),
    "medium_phase": ("solint_medium_timestep", "solint_medium_freqstep"),
    "slow_gains": ("solint_slow_timestep", "solint_slow_freqstep"),
    "full_jones": ("solint_fulljones_timestep", "solint_fulljones_freqstep"),
}


@dataclass(frozen=True)
class CalibrationSolve:
    """Resolved mapping from a strategy solve to a DP3 solve slot."""

    solve_type: str
    slot: int
    mode: str
    output_prefix: str
    collected_h5parm: str
    timestep_key: str
    freqstep_key: str
    field_prefix: str

    @property
    def step(self):
        return f"solve{self.slot}"

    def output_h5parms(self, ntimechunks):
        return [f"{self.output_prefix}_{index}.h5parm" for index in range(ntimechunks)]


class Calibrate(Operation):
    """
    Class for performing the calibration operation, which runs the CWL workflow template for calibration.
    This class is used for both direction-dependent (DD) and direction-independent (DI) calibration, with
    the mode specified by the "mode" parameter in the constructor.
    """

    def __init__(self, mode, field, index):
        if mode not in ["di", "dd"]:
            raise ValueError(f"Only di and dd mode are supported, chosen: {mode}")
        super().__init__(
            field,
            index=index,
            name="calibrate" if mode == "dd" else "calibrate_di",
            rootname="calibrate",
        )
        self.mode = mode

    def uses_python_flow(self):
        """
        Calibrate operations are executed through the Prefect/Dask Python flow.
        """
        return True

    def set_parset_parameters(self):
        """
        Define parameters needed for the CWL workflow template
        """
        if self.batch_system.startswith("slurm"):
            # For some reason, setting coresMax ResourceRequirement hints does
            # not work with SLURM
            max_cores = None
        else:
            max_cores = self.parset["cluster_specific"]["max_cores"]

        # Base parameters (shared by both DD and DI)
        self.parset_parms = {
            "rapthor_pipeline_dir": self.rapthor_pipeline_dir,
            "max_cores": max_cores,
            "mode": self.mode,
        }

        # Add DD-specific parameters only when needed
        if self.mode == "dd":
            # Set whether image-based prediction is used. Note that generation
            # of screens (IDGCal) requires image-based prediction.
            self.use_image_based_predict = (
                self.field.generate_screens or self.field.use_image_based_predict
            )

            self.parset_parms.update(
                {
                    "use_image_based_predict": self.use_image_based_predict,
                    "generate_screens": self.field.generate_screens,
                }
            )

    def set_input_parameters(self):
        """
        Define the CWL workflow inputs
        """
        # First set the calibration parameters for each observation
        field = self.field
        field.set_obs_parameters()
        # Get the start times and number of times for the time chunks (fast and slow
        # calibration)
        starttime = field.get_obs_parameters("starttime")
        ntimes = field.get_obs_parameters("ntimes")
        calibration_skymodel_file = field.calibration_skymodel_file

        if self.mode == "dd":
            # --- output h5parm configuration ---
            # Define various output filenames for the solution tables. We save some
            # as attributes since they are needed in finalize()
            self.fast_h5parm = "fast_phases.h5parm"
            self.medium1_h5parm = "medium1_phases.h5parm"
            self.medium2_h5parm = "medium2_phases.h5parm"
            self.slow_h5parm = "slow_gains.h5parm"
            self.combined_h5parms = "combined_solutions.h5"

            # --- Sky model configuration ---
            # Define the input sky model
            num_spectral_terms = misc.get_max_spectral_terms(calibration_skymodel_file)
            (
                model_image_frequency_bandwidth,
                model_image_ra_dec,
                model_image_imsize,
                model_image_cellsize,
            ) = self._get_model_image_parameters()

            # --- Set the constraints used in the calibrations ---
            # Smoothness constraints
            smoothness_dd_factors = {}
            smoothness_constraints = {}
            for field_key, input_key in zip(
                ("fast", "medium", "slow", "medium"), ("solve1", "solve2", "solve3", "solve4")
            ):
                dd_factor_key_field = f"{field_key}_smoothness_dd_factors"
                dd_factor_key_inputs = f"{input_key}_smoothness_dd_factors"
                constraint_key = f"{input_key}_smoothnessconstraint"
                dd_factor = smoothness_dd_factors[dd_factor_key_inputs] = field.get_obs_parameters(
                    dd_factor_key_field
                )

                factor_constraint_key = f"{field_key}_smoothnessconstraint"
                smoothness_constraints[constraint_key] = getattr(
                    field, factor_constraint_key
                ) / np.min(dd_factor)
            # Antenna constraints
            core_stations = self._get_core_stations()
            fast_antennaconstraint = f"[[{','.join(core_stations)}]]" if core_stations else "[]"
            medium_antennaconstraint = fast_antennaconstraint  # ???

            solve_plan = self._build_solve_plan()
            self.solve_plan = solve_plan
            applycal_inputs = self._build_applycal(field)

            # --- DP3 pipeline steps ---
            dp3_steps = self._build_dp3_steps(
                field.calibrate_bda_timebase,
                field.calibrate_bda_frequencybase,
                solve_steps=[solve.step for solve in solve_plan],
                preapply_solutions=applycal_inputs["applycal_steps"] is not None,
            )
            # --- Build final CWL input dict ---
            self.input_parms = {
                # File inputs / basic run configuration
                # Get the filenames of the input files for each time chunk
                "timechunk_filename": CWLDir(
                    field.get_obs_parameters("timechunk_filename")
                ).to_json(),
                "data_colname": field.data_colname,
                "modeldatacolumn": None,
                "generate_screens": field.generate_screens,
                "starttime": starttime,
                "ntimes": ntimes,
                # Solution interval configuration (time + frequency)
                # Get the solution intervals for the calibrations
                "do_slowgain_solve": field.do_slowgain_solve,
                "solint_fast_timestep": field.get_obs_parameters("solint_fast_timestep"),
                "solint_medium_timestep": field.get_obs_parameters("solint_medium_timestep"),
                "solint_slow_timestep": field.get_obs_parameters("solint_slow_timestep"),
                "solint_fast_freqstep": field.get_obs_parameters("solint_fast_freqstep"),
                "solint_medium_freqstep": field.get_obs_parameters("solint_medium_freqstep"),
                "solint_slow_freqstep": field.get_obs_parameters("solint_slow_freqstep"),
                # Solutions per direction
                "solve1_solutions_per_direction": field.get_obs_parameters(
                    "fast_solutions_per_direction"
                ),
                "solve2_solutions_per_direction": field.get_obs_parameters(
                    "medium_solutions_per_direction"
                ),
                "solve4_solutions_per_direction": field.get_obs_parameters(
                    "medium_solutions_per_direction"
                ),
                "solve3_solutions_per_direction": field.get_obs_parameters(
                    "slow_solutions_per_direction"
                ),
                # Calibration outputs (H5parm products)
                "calibrator_patch_names": field.calibrator_patch_names,
                "solve_directions": field.calibrator_patch_names,
                "calibrator_fluxes": field.calibrator_fluxes,
                "output_solve1_h5parm": [
                    f"fast_phase_{i}.h5parm" for i in range(field.ntimechunks)
                ],
                "collected_solve1_h5parm": self.fast_h5parm,
                "output_solve2_h5parm": [
                    f"medium1_phase_{i}.h5parm" for i in range(field.ntimechunks)
                ],
                "output_solve4_h5parm": [
                    f"medium2_phase_{i}.h5parm" for i in range(field.ntimechunks)
                ],
                "collected_solve2_h5parm": self.medium1_h5parm,
                "collected_solve4_h5parm": self.medium2_h5parm,
                "combined_solve1_solve2_h5parm": "combined_fast_medium1_phases.h5parm",
                "combined_solve1_solve2_solve4_h5parm": "combined_fast_medium1_medium2_phases.h5parm",
                "output_solve3_h5parm": [f"slow_gain_{i}.h5parm" for i in range(field.ntimechunks)],
                "collected_solve3_h5parm": self.slow_h5parm,
                # Sky model configuration
                "calibration_skymodel_file": CWLFile(calibration_skymodel_file).to_json(),
                "model_image_root": "calibration_model",
                "model_image_ra_dec": model_image_ra_dec,
                "model_image_imsize": model_image_imsize,
                "model_image_cellsize": model_image_cellsize,
                "model_image_frequency_bandwidth": model_image_frequency_bandwidth,
                "num_spectral_terms": num_spectral_terms,
                # Geometry / field setup
                "ra_mid": field.ra,
                "dec_mid": field.dec,
                "phase_center_ra": field.ra,
                "phase_center_dec": field.dec,
                "facet_region_width_ra": (
                    facet_region_width := max(model_image_imsize) * model_image_cellsize * 1.2
                    # deg
                ),
                "facet_region_width_dec": facet_region_width,
                "facet_region_file": "field_facets_ds9.reg",
                # Smoothness / regularisation constraints
                **smoothness_dd_factors,
                **smoothness_constraints,
                "solve1_smoothnessreffrequency": field.get_obs_parameters(
                    "fast_smoothnessreffrequency"
                ),
                "solve2_smoothnessreffrequency": field.get_obs_parameters(
                    "medium_smoothnessreffrequency"
                ),
                "solve4_smoothnessreffrequency": field.get_obs_parameters(
                    "medium_smoothnessreffrequency"
                ),
                "solve1_smoothnessrefdistance": field.fast_smoothnessrefdistance,
                "solve2_smoothnessrefdistance": field.medium_smoothnessrefdistance,
                "solve4_smoothnessrefdistance": field.medium_smoothnessrefdistance,
                # Applycal / DP3 control flow
                "dp3_steps": f"[{','.join(dp3_steps)}]",
                # --- Applycal + H5parm inputs ---
                # Set the DP3 applycal steps and input H5parm files depending on what
                # solutions need to be applied. Note: applycal steps are needed for
                # both the case in which applycal is part of the DDECal solve step and
                # the case in which it is a separate step that preceeds the DDECal step.
                # The latter is used when image-based predict is done
                **applycal_inputs,
                # Get the BDA (baseline-dependent averaging) parameters
                "bda_maxinterval": field.get_obs_parameters("bda_maxinterval"),
                "bda_minchannels": field.get_obs_parameters("bda_minchannels"),
                "bda_timebase": field.calibrate_bda_timebase,
                "bda_frequencybase": field.calibrate_bda_frequencybase,
                # Normalisation / scaling
                "max_normalization_delta": field.max_normalization_delta,
                "scale_normalization_delta": str(field.scale_normalization_delta),
                # Initial solutions (H5parm inputs)
                "fast_initialsolutions_h5parm": self._to_cwl_json_if_exists(
                    field.fast_phases_h5parm_filename
                ),
                "medium1_initialsolutions_h5parm": self._to_cwl_json_if_exists(
                    field.medium1_phases_h5parm_filename
                ),
                "solve4_initialsolutions_h5parm": self._to_cwl_json_if_exists(
                    field.medium2_phases_h5parm_filename
                ),
                "solve3_initialsolutions_h5parm": self._to_cwl_json_if_exists(
                    field.slow_gains_h5parm_filename
                ),
                # Get various DDECal solver parameters. Most of these are the same for both fast
                # and slow solves
                # ------------------------------------
                "llssolver": field.llssolver,
                "maxiter": field.maxiter,
                "propagatesolutions": field.propagatesolutions,
                "solveralgorithm": field.solveralgorithm,
                "onebeamperpatch": field.onebeamperpatch,
                "stepsize": field.stepsize,
                "stepsigma": field.stepsigma,
                "tolerance": field.tolerance,
                "uvlambdamin": field.solve_min_uv_lambda,
                "parallelbaselines": field.parallelbaselines,
                "sagecalpredict": field.sagecalpredict,
                "solve1_datause": field.fast_datause,
                "solve2_datause": field.medium_datause,
                "solve3_datause": field.slow_datause,
                "solve4_datause": field.medium_datause,
                "solverlbfgs_dof": field.solverlbfgs_dof,
                "solverlbfgs_iter": field.solverlbfgs_iter,
                "solverlbfgs_minibatches": field.solverlbfgs_minibatches,
                "solve1_mode": "scalarphase",
                "solve2_mode": "scalarphase",
                "solve3_mode": "diagonal",
                "solve4_mode": "scalarphase",
                "solint_solve1_timestep": field.get_obs_parameters("solint_fast_timestep"),
                "solint_solve2_timestep": field.get_obs_parameters("solint_medium_timestep"),
                "solint_solve3_timestep": field.get_obs_parameters("solint_slow_timestep"),
                "solint_solve4_timestep": field.get_obs_parameters("solint_medium_timestep"),
                "solint_solve1_freqstep": field.get_obs_parameters("solint_fast_freqstep"),
                "solint_solve2_freqstep": field.get_obs_parameters("solint_medium_freqstep"),
                "solint_solve3_freqstep": field.get_obs_parameters("solint_slow_freqstep"),
                "solint_solve4_freqstep": field.get_obs_parameters("solint_medium_freqstep"),
                # ------------------------------------
                # Get the size of the imaging area (for use in making the a-term images)
                "sector_bounds_deg": str(field.sector_bounds_deg),
                "sector_bounds_mid_deg": str(field.sector_bounds_mid_deg),
                "combined_h5parms": self.combined_h5parms,
                "solve1_antennaconstraint": fast_antennaconstraint,
                "solve2_antennaconstraint": medium_antennaconstraint,
                "solve4_antennaconstraint": medium_antennaconstraint,
                "solve3_antennaconstraint": "[]",
                "idgcal_antennaconstraint": (
                    "[]"  # TODO: set different constraints for phase and gain solves
                ),
                "output_idgcal_h5parm": [f"idgcal_{i}" for i in range(field.ntimechunks)],
                "solution_combine_mode": (
                    "p1p2a2_diagonal" if field.apply_diagonal_solutions else "p1p2a2_scalar"
                ),
                "correctfreqsmearing": field.correct_smearing_in_calibration,
                "correcttimesmearing": field.correct_smearing_in_calibration,
                "max_threads": self.parset["cluster_specific"]["max_threads"],
            }
            self._apply_solve_plan_inputs(solve_plan, dp3_steps=dp3_steps)
        elif self.mode == "di":
            solve_plan = self._build_solve_plan()
            self.solve_plan = solve_plan

            # Define various output filenames for the solution tables. We save some
            # as attributes since they are needed in finalize()
            self.collected_h5parm_fulljones = "fulljones_solutions.h5"

            # Set the constraints used in the calibrations
            self.input_parms = {
                # Get the filenames of the input files for each time chunk. These are the
                # output of the predict_di pipeline done before this calibration
                "timechunk_filename": CWLDir(
                    field.get_obs_parameters("predict_di_output_filename")
                ).to_json(),
                "data_colname": "DATA",
                "calibration_skymodel_file": None,
                "starttime": starttime,
                "ntimes": ntimes,
                # Get the BDA (baseline-dependent averaging) parameters
                "bda_maxinterval": field.get_obs_parameters("bda_maxinterval"),
                "bda_minchannels": field.get_obs_parameters("bda_minchannels"),
                "bda_timebase": field.calibrate_bda_timebase,
                "bda_frequencybase": field.calibrate_bda_frequencybase,
                "onebeamperpatch": field.onebeamperpatch,
                "parallelbaselines": field.parallelbaselines,
                "sagecalpredict": field.sagecalpredict,
                "do_slowgain_solve": field.do_slowgain_solve,
                "normalize_h5parm": None,
                "ddecal_applycal_steps": None,
                "applycal_steps": None,
                "applycal_h5parm": None,
                "fulljones_h5parm": None,
                # Get the solution intervals for the calibrations
                "solint_fast_timestep": field.get_obs_parameters("solint_fulljones_timestep"),
                "solint_fast_freqstep": field.get_obs_parameters("solint_fulljones_freqstep"),
                "solint_slow_timestep": field.get_obs_parameters("solint_slow_timestep"),
                "solint_slow_freqstep": field.get_obs_parameters("solint_slow_freqstep"),
                "solint_solve1_timestep": field.get_obs_parameters("solint_fulljones_timestep"),
                "solint_solve1_freqstep": field.get_obs_parameters("solint_fulljones_freqstep"),
                "fast_initialsolutions_h5parm": None,
                "medium1_initialsolutions_h5parm": None,
                "solve3_initialsolutions_h5parm": None,
                "solve4_initialsolutions_h5parm": None,
                "solve1_solutions_per_direction": [None for _ in range(field.ntimechunks)],
                "solve1_smoothness_dd_factors": [None for _ in range(field.ntimechunks)],
                "solve1_smoothnessreffrequency": [0] * field.ntimechunks,
                "solve1_smoothnessrefdistance": None,
                "solve2_solutions_per_direction": [None for _ in range(field.ntimechunks)],
                "solve2_smoothness_dd_factors": [None for _ in range(field.ntimechunks)],
                "solve2_smoothnessreffrequency": [0] * field.ntimechunks,
                "solve2_smoothnessrefdistance": None,
                "solve3_smoothness_dd_factors": [None for _ in range(field.ntimechunks)],
                "solve3_smoothnessreffrequency": [0] * field.ntimechunks,
                "solve2_smoothnessconstraint": 0,
                "solve3_solutions_per_direction": [None for _ in range(field.ntimechunks)],
                "solve3_smoothnessconstraint": 0,
                "solve4_smoothness_dd_factors": [None for _ in range(field.ntimechunks)],
                "solve4_smoothnessreffrequency": [0] * field.ntimechunks,
                "solve4_smoothnessrefdistance": None,
                "solve4_smoothnessconstraint": 0,
                "solve4_solutions_per_direction": [None for _ in range(field.ntimechunks)],
                "output_solve2_h5parm": [
                    f"medium1_phase_{i}.h5parm" for i in range(field.ntimechunks)
                ],
                "output_solve3_h5parm": [f"slow_gain_{i}.h5parm" for i in range(field.ntimechunks)],
                "output_solve4_h5parm": [
                    f"medium2_phase_{i}.h5parm" for i in range(field.ntimechunks)
                ],
                "collected_solve2_h5parm": "unused",
                "collected_solve3_h5parm": "unused",
                "collected_solve4_h5parm": "unused",
                "combined_solve1_solve2_h5parm": "unused",
                "combined_solve1_solve2_solve4_h5parm": "unused",
                "combined_h5parms": "unused",
                "solint_solve2_timestep": field.get_obs_parameters("solint_medium_timestep"),
                "solint_solve3_timestep": field.get_obs_parameters("solint_slow_timestep"),
                "solint_solve4_timestep": field.get_obs_parameters("solint_medium_timestep"),
                "solint_solve2_freqstep": field.get_obs_parameters("solint_medium_freqstep"),
                "solint_solve3_freqstep": field.get_obs_parameters("solint_slow_freqstep"),
                "solint_solve4_freqstep": field.get_obs_parameters("solint_medium_freqstep"),
                "solve1_smoothnessconstraint": field.smoothnessconstraint_fulljones,
                # ------------------------------------
                # Get the size of the imaging area (for use in making the a-term images)
                "sector_bounds_deg": str(field.sector_bounds_deg),
                "sector_bounds_mid_deg": str(field.sector_bounds_mid_deg),
                "solve1_mode": "fulljones",
                "solve2_mode": "null",
                "solve3_mode": "null",
                "solve4_mode": "null",
                "modeldatacolumn": "[MODEL_DATA]",
                "calibrator_patch_names": [],
                "solve_directions": None,
                "calibrator_fluxes": [],
                "dp3_steps": "[solve1]",
                "output_solve1_h5parm": [
                    f"fulljones_gain_{i}.h5parm" for i in range(field.ntimechunks)
                ],
                "collected_solve1_h5parm": self.collected_h5parm_fulljones,
                "smoothnessconstraint_fulljones": field.smoothnessconstraint_fulljones,
                "max_normalization_delta": field.max_normalization_delta,
                "scale_normalization_delta": str(field.scale_normalization_delta),
                "phase_center_ra": field.ra,
                "phase_center_dec": field.dec,
                # Get various DDECal solver parameters. Most of these are the same for both fast
                # and slow solves
                # ------------------------------------
                "llssolver": field.llssolver,
                "maxiter": field.maxiter,
                "propagatesolutions": field.propagatesolutions,
                "solveralgorithm": field.solveralgorithm,
                "stepsize": field.stepsize,
                "stepsigma": field.stepsigma,
                "tolerance": field.tolerance,
                "uvlambdamin": field.solve_min_uv_lambda,
                "solverlbfgs_dof": field.solverlbfgs_dof,
                "solverlbfgs_iter": field.solverlbfgs_iter,
                "solverlbfgs_minibatches": field.solverlbfgs_minibatches,
                "solve1_datause": None,
                "solve2_datause": None,
                "solve3_datause": None,
                "solve4_datause": None,
                "solve1_antennaconstraint": "[]",
                "solve2_antennaconstraint": "[]",
                "solve3_antennaconstraint": "[]",
                "solve4_antennaconstraint": "[]",
                "solution_combine_mode": "p1p2a2_scalar",
                # ---------------------------------
                "correctfreqsmearing": field.correct_smearing_in_calibration,
                "correcttimesmearing": field.correct_smearing_in_calibration,
                "max_threads": self.parset["cluster_specific"]["max_threads"],
            }
            self._apply_solve_plan_inputs(solve_plan)

            #
            #    "solve1_datause": field.fast_datause ,
            #    "solve2_datause": field.medium_datause,
            #    "solve3_datause": field.slow_datause,
            #    "solve4_datause": field.medium_datause,

    def _requested_calibration_solves(self):
        strategy = getattr(self.field, "calibration_strategy", None)
        if strategy is not None and self.mode in strategy:
            return list(strategy.get(self.mode) or []), getattr(
                self.field, "_calibration_strategy_defaulted", False
            )

        if self.mode == "dd":
            solves = ["fast_phase", "medium_phase"]
            if self.field.do_slowgain_solve:
                solves.append("slow_gains")
            return solves, True

        return ["full_jones"], True

    def _build_solve_plan(self):
        requested_solves, defaulted_strategy = self._requested_calibration_solves()
        expanded_solves = list(requested_solves)

        if (
            self.mode == "dd"
            and defaulted_strategy
            and expanded_solves == ["fast_phase", "medium_phase", "slow_gains"]
        ):
            expanded_solves.append("medium_phase")

        if len(expanded_solves) > 4:
            raise ValueError("A calibration cycle can contain at most four solve slots")

        medium_count = 0
        solve_plan = []
        for slot, solve_type in enumerate(expanded_solves, start=1):
            if solve_type == "medium_phase":
                medium_count += 1
            solve_plan.append(self._build_solve_slot(solve_type, slot, medium_count))

        return solve_plan

    def _build_solve_slot(self, solve_type, slot, medium_count):
        output_prefix, collected_h5parm = self._solve_output_names(solve_type, medium_count)
        timestep_key, freqstep_key = INTERVAL_KEYS_BY_SOLVE[solve_type]

        return CalibrationSolve(
            solve_type=solve_type,
            slot=slot,
            mode=(
                "scalarphase"
                if solve_type == "slow_gains" and slot == 1
                else MODE_BY_SOLVE[solve_type]
            ),
            output_prefix=output_prefix,
            collected_h5parm=collected_h5parm,
            timestep_key=timestep_key,
            freqstep_key=freqstep_key,
            field_prefix=FIELD_PREFIX_BY_SOLVE[solve_type],
        )

    def _solve_output_names(self, solve_type, medium_count):
        if solve_type == "fast_phase":
            suffix = "_di" if self.mode == "di" else ""
            return f"fast_phase{suffix}", f"fast_phases{suffix}.h5parm"
        if solve_type == "medium_phase":
            medium_name = "medium2" if medium_count > 1 else "medium1"
            suffix = "_di" if self.mode == "di" else ""
            return f"{medium_name}_phase{suffix}", f"{medium_name}_phases{suffix}.h5parm"
        if solve_type == "slow_gains":
            if self.mode == "di":
                return "slow_gains_di", "slow_gains_di.h5parm"
            return "slow_gain", "slow_gains.h5parm"
        if solve_type == "full_jones":
            return "fulljones_gain", "fulljones_solutions.h5"

        raise ValueError(f"Unsupported solve type: {solve_type}")

    def _apply_solve_plan_inputs(self, solve_plan, dp3_steps=None):
        field = self.field
        if dp3_steps is None:
            dp3_steps = [solve.step for solve in solve_plan]
        self.input_parms["dp3_steps"] = f"[{','.join(dp3_steps)}]"
        self.input_parms["do_slowgain_solve"] = any(
            solve.solve_type == "slow_gains" and solve.slot == 3 for solve in solve_plan
        )

        slot_map = {solve.slot: solve for solve in solve_plan}
        for slot in range(1, 5):
            solve = slot_map.get(slot)
            output_key = f"output_solve{slot}_h5parm"
            collected_key = f"collected_solve{slot}_h5parm"
            mode_key = f"solve{slot}_mode"
            timestep_key = f"solint_solve{slot}_timestep"
            freqstep_key = f"solint_solve{slot}_freqstep"

            if solve is None:
                self.input_parms[output_key] = [
                    f"unused_solve{slot}_{index}.h5parm" for index in range(field.ntimechunks)
                ]
                self.input_parms[collected_key] = "unused"
                self.input_parms[mode_key] = "null"
                self._clear_solve_slot_inputs(slot)
                continue

            self.input_parms[output_key] = solve.output_h5parms(field.ntimechunks)
            self.input_parms[collected_key] = solve.collected_h5parm
            self.input_parms[mode_key] = solve.mode
            self.input_parms[timestep_key] = field.get_obs_parameters(solve.timestep_key)
            self.input_parms[freqstep_key] = field.get_obs_parameters(solve.freqstep_key)
            self._apply_solve_slot_inputs(solve)

        if self.mode == "di":
            self.input_parms["combined_solve1_solve2_h5parm"] = "combined_solve1_solve2_di.h5parm"
            self.input_parms["combined_solve1_solve2_solve4_h5parm"] = (
                "combined_solve1_solve2_solve4_di.h5parm"
            )
            self.input_parms["combined_h5parms"] = "combined_di_solutions.h5parm"

    def _clear_solve_slot_inputs(self, slot):
        ntimechunks = self.field.ntimechunks
        default_values = {
            f"solve{slot}_datause": None,
            f"solve{slot}_solutions_per_direction": [None] * ntimechunks,
            f"solve{slot}_smoothness_dd_factors": [None] * ntimechunks,
            f"solve{slot}_smoothnessreffrequency": [0] * ntimechunks,
            f"solve{slot}_smoothnessconstraint": 0,
            f"solve{slot}_antennaconstraint": "[]",
            f"solve{slot}_smoothnessrefdistance": None,
        }
        for key, value in default_values.items():
            if key in self.input_parms:
                self.input_parms[key] = value

    def _apply_solve_slot_inputs(self, solve):
        slot = solve.slot
        if self.mode == "di" or solve.solve_type == "full_jones":
            self._clear_solve_slot_inputs(slot)
            self.input_parms[f"solve{slot}_mode"] = solve.mode
            if solve.solve_type == "full_jones":
                self.input_parms[f"solve{slot}_smoothnessconstraint"] = (
                    self.field.smoothnessconstraint_fulljones
                )
            return

        field_prefix = solve.field_prefix
        dd_factors = self.field.get_obs_parameters(f"{field_prefix}_smoothness_dd_factors")
        self.input_parms[f"solve{slot}_datause"] = getattr(self.field, f"{field_prefix}_datause")
        self.input_parms[f"solve{slot}_solutions_per_direction"] = self.field.get_obs_parameters(
            f"{field_prefix}_solutions_per_direction"
        )
        self.input_parms[f"solve{slot}_smoothness_dd_factors"] = dd_factors
        if f"solve{slot}_smoothnessreffrequency" in self.input_parms:
            if field_prefix in {"fast", "medium"}:
                self.input_parms[f"solve{slot}_smoothnessreffrequency"] = (
                    self.field.get_obs_parameters(f"{field_prefix}_smoothnessreffrequency")
                )
            else:
                self.input_parms[f"solve{slot}_smoothnessreffrequency"] = [
                    0
                ] * self.field.ntimechunks
        self.input_parms[f"solve{slot}_smoothnessconstraint"] = getattr(
            self.field, f"{field_prefix}_smoothnessconstraint"
        ) / np.min(dd_factors)
        self.input_parms[f"solve{slot}_antennaconstraint"] = (
            self.input_parms.get("solve1_antennaconstraint", "[]")
            if field_prefix in {"fast", "medium"}
            else "[]"
        )
        if f"solve{slot}_smoothnessrefdistance" in self.input_parms:
            self.input_parms[f"solve{slot}_smoothnessrefdistance"] = getattr(
                self.field, f"{field_prefix}_smoothnessrefdistance", None
            )

    def _build_dp3_steps(
        self,
        bda_timebase,
        bda_frequencybase,
        solve_steps=None,
        preapply_solutions=False,
    ):
        """
        Set the DDECal steps depending on whether baseline-dependent averaging is
        activated (and supported) or not. If BDA is used, a "null" step is also added to
        prevent the writing of the BDA data

        TODO: image-based predict doesn't yet work with BDA; once it does,
        the restriction on this mode should be removed
        """

        # Check whether all observations have regular channelization
        all_regular = all([obs.channels_are_regular for obs in self.field.observations])

        # Base solve chain depending on BDA + solver configuration
        if solve_steps is None:
            if self.field.do_slowgain_solve:
                common_steps = ["solve1", "solve2", "solve3", "solve4"]
            else:
                common_steps = ["solve1", "solve2"]
        else:
            common_steps = list(solve_steps)

        if (
            (bda_timebase > 0 or bda_frequencybase > 0)
            and all_regular
            and not self.field.use_image_based_predict
        ):
            common_steps = ["avg", *common_steps, "null"]

        if preapply_solutions and not self.field.use_image_based_predict:
            common_steps = ["applycal", *common_steps]

        # Optional image-based predict prefix
        if self.field.use_image_based_predict:
            # Add a predict, applybeam, and applycal steps to the beginning
            preprocessing_steps = (
                ["predict", "applybeam", "applycal"]
                if preapply_solutions
                else ["predict", "applybeam"]
            )
            dp3_steps = preprocessing_steps + common_steps
        else:
            dp3_steps = common_steps

        return dp3_steps

    def _build_applycal(self, field):
        """
        Prepare DP3 pre-apply steps used before DD calibration solves.

        The scalar DI h5parm is applied as DP3's ``fastphase`` step because it
        contains a single ``phase000`` soltab. When DI fast and medium phase
        solves were both run, ``di_h5parm_filename`` points at their combined
        scalar product, so there is no separate ``mediumphase`` pre-apply step
        here. Explicit ``mediumphase`` application belongs to the imaging
        prepare-data stage, where separate final calibration products may be
        applied to imaging visibilities.
        """
        steps = []
        applycal_h5parm = None
        fulljones_h5parm = None

        di_h5parm = getattr(field, "di_h5parm_filename", None)
        applycal_h5parm = self._to_cwl_json_if_exists(di_h5parm)
        if self.mode == "dd" and applycal_h5parm is not None:
            steps.append("fastphase")
            di_strategy = (getattr(field, "calibration_strategy", None) or {}).get("di", [])
            di_has_phase_solves = any(
                solve in {"fast_phase", "medium_phase"} for solve in di_strategy
            )
            if field.apply_amplitudes and not di_has_phase_solves:
                steps.append("slowgain")

        fulljones_h5parm = self._to_cwl_json_if_exists(field.fulljones_h5parm_filename)
        if self.mode == "dd" and fulljones_h5parm is not None:
            steps.append("fulljones")

        if field.apply_normalizations:
            steps.append("normalization")

        applycal_steps = f"[{','.join(steps)}]" if steps else None
        return {
            "normalize_h5parm": self._to_cwl_json_if_exists(field.normalize_h5parm),
            "ddecal_applycal_steps": applycal_steps,
            "applycal_steps": applycal_steps,
            "applycal_h5parm": applycal_h5parm,
            "fulljones_h5parm": fulljones_h5parm,
        }

    def _get_baselines_core(self):
        """
        Returns DPPP string of baseline selection for core calibration

        Returns
        -------
        baselines : str
            Baseline selection string
        """
        cs = self._get_core_stations()
        non_core = [a for a in self.field.stations if a not in cs]

        return f"[CR]*&&;!{';!'.join(non_core)}"

    def _get_superterp_stations(self):
        """
        Returns list of superterp station names

        Returns
        -------
        stations : list
            Station names
        """
        if self.field.antenna == "HBA":
            all_st = [
                "CS002HBA0",
                "CS003HBA0",
                "CS004HBA0",
                "CS005HBA0",
                "CS006HBA0",
                "CS007HBA0",
                "CS002HBA1",
                "CS003HBA1",
                "CS004HBA1",
                "CS005HBA1",
                "CS006HBA1",
                "CS007HBA1",
            ]
        elif self.field.antenna == "LBA":
            all_st = ["CS002LBA", "CS003LBA", "CS004LBA", "CS005LBA", "CS006LBA", "CS007LBA"]

        return [a for a in all_st if a in self.field.stations]

    def _get_core_stations(self, include_nearest_remote=True):
        """
        Returns list of station names for core calibration

        Parameters
        ----------
        include_nearest_remote : bool, optional
            If True, include the remote stations nearest to the core

        Returns
        -------
        stations : list
            Station names
        """
        if self.field.antenna == "HBA":
            all_core = [
                "CS001HBA0",
                "CS002HBA0",
                "CS003HBA0",
                "CS004HBA0",
                "CS005HBA0",
                "CS006HBA0",
                "CS007HBA0",
                "CS011HBA0",
                "CS013HBA0",
                "CS017HBA0",
                "CS021HBA0",
                "CS024HBA0",
                "CS026HBA0",
                "CS028HBA0",
                "CS030HBA0",
                "CS031HBA0",
                "CS032HBA0",
                "CS101HBA0",
                "CS103HBA0",
                "CS201HBA0",
                "CS301HBA0",
                "CS302HBA0",
                "CS401HBA0",
                "CS501HBA0",
                "CS001HBA1",
                "CS002HBA1",
                "CS003HBA1",
                "CS004HBA1",
                "CS005HBA1",
                "CS006HBA1",
                "CS007HBA1",
                "CS011HBA1",
                "CS013HBA1",
                "CS017HBA1",
                "CS021HBA1",
                "CS024HBA1",
                "CS026HBA1",
                "CS028HBA1",
                "CS030HBA1",
                "CS031HBA1",
                "CS032HBA1",
                "CS101HBA1",
                "CS103HBA1",
                "CS201HBA1",
                "CS301HBA1",
                "CS302HBA1",
                "CS401HBA1",
                "CS501HBA1",
            ]
            if include_nearest_remote:
                all_core.extend(
                    [
                        "RS106HBA0",
                        "RS205HBA0",
                        "RS305HBA0",
                        "RS306HBA0",
                        "RS503HBA0",
                        "RS106HBA1",
                        "RS205HBA1",
                        "RS305HBA1",
                        "RS306HBA1",
                        "RS503HBA1",
                    ]
                )
        elif self.field.antenna == "LBA":
            all_core = [
                "CS001LBA",
                "CS002LBA",
                "CS003LBA",
                "CS004LBA",
                "CS005LBA",
                "CS006LBA",
                "CS007LBA",
                "CS011LBA",
                "CS013LBA",
                "CS017LBA",
                "CS021LBA",
                "CS024LBA",
                "CS026LBA",
                "CS028LBA",
                "CS030LBA",
                "CS031LBA",
                "CS032LBA",
                "CS101LBA",
                "CS103LBA",
                "CS201LBA",
                "CS301LBA",
                "CS302LBA",
                "CS401LBA",
                "CS501LBA",
            ]
            if include_nearest_remote:
                all_core.extend(["RS106LBA", "RS205LBA", "RS305LBA", "RS306LBA", "RS503LBA"])

        return [a for a in all_core if a in self.field.stations]

    def _get_model_image_parameters(self):
        """
        Returns parameters needed for image-based predict

        Returns
        -------
        frequency_bandwidth : [float, float]
            Central frequency and bandwidth as [frequency, bandwidth] of model image in Hz
        center_coords : [str, str]
            Center of the image as [HHMMSS.S, DDMMSS.S] strings
        size : [int, int]
            Size of image as [RA, Dec] in pixels
        cellsize : float
            Size of image cell (pixel) in degrees/pixel
        """
        # Set frequency parameters. For the central frequency, we use the reference
        # frequency of the sky model (i.e., the frequency to which the fluxes are
        # referenced). For the bandwidth, we use 1 MHz as it is appropriate for images at
        # LOFAR frequencies, but the exact value is not important since the bandwidth does
        # not have any effect on the processing done in Rapthor
        skymodel = lsmtool.load(self.field.calibration_skymodel_file)
        if "ReferenceFrequency" in skymodel.getColNames():
            # Each source can have its own reference frequency, so use the median over all
            # sources
            ref_freq = np.median(skymodel.getColValues("ReferenceFrequency"))  # Hz
        else:
            ref_freq = skymodel.table.meta["ReferenceFrequency"]  # Hz
        frequency_bandwidth = [ref_freq, 1e6]  # Hz

        # Set the image coordinates, size, and cellsize
        if self.index == 1:
            # For initial cycle, assume center is the field center
            center_coords = [self.field.ra, self.field.dec]
            if hasattr(self.field, "full_field_sector"):
                # Sky model generated in initial image step
                cellsize = self.field.full_field_sector.cellsize_deg  # deg/pixel
                size = self.field.full_field_sector.imsize  # [xsize, ysize] in pixels
            else:
                # Sky model generated externally. Use the cellsize defined for imaging and
                # analyze the sky model to find its extent
                cellsize = self.parset["imaging_specific"]["cellsize_arcsec"] / 3600  # deg/pixel
                source_dict = {
                    name: [ra, dec]
                    for name, ra, dec in zip(
                        skymodel.getColValues("Name"),
                        skymodel.getColValues("RA"),
                        skymodel.getColValues("Dec"),
                    )
                }
                _, source_distances = self.field.get_source_distances(source_dict)  # deg
                radius = int(np.max(source_distances) / cellsize)  # pixels
                size = [radius * 2, radius * 2]  # pixels
        else:
            # Sky model generated in previous cycle's imaging step. Use the center and size
            # of the bounding box of all imaging sectors (note that this bounding box
            # includes a 20% padding, so it should include all model components, even
            # those on the very edge of a sector)
            cellsize = self.parset["imaging_specific"]["cellsize_arcsec"] / 3600  # deg/pixel
            center_coords = [
                self.field.sector_bounds_mid_ra,
                self.field.sector_bounds_mid_dec,
            ]  # deg
            size = [
                int(self.field.sector_bounds_width_ra / cellsize),
                int(self.field.sector_bounds_width_dec / cellsize),
            ]  # pixels

        # Convert RA and Dec to strings (required by WSClean)
        center_coords = lsmtool.utils.format_coordinates(*center_coords)

        return frequency_bandwidth, center_coords, size, cellsize

    @staticmethod
    def _to_cwl_json_if_exists(filepath):
        if filepath is not None and os.path.exists(filepath):
            return CWLFile(filepath).to_json()
        return None

    def execute_workflow(self):
        """
        Execute calibration through the Prefect flow and return operation outputs.
        """
        payload = calibrate_payload_from_inputs(
            self.mode,
            self.input_parms,
            self.pipeline_working_dir,
        )
        outputs = calibrate_flow(
            payload,
            execution_config=ExecutionConfig.from_parset(self.parset),
        )
        return True, outputs

    def finalize(self):
        """
        Finalize this operation
        """
        field = self.field
        # set up directories for copying solutions and plots
        workdir = self.parset["dir_working"]
        plot_filenames = glob.glob(os.path.join(self.pipeline_working_dir, "*.png"))
        sol_dir = os.path.join(workdir, "solutions", self.name)
        plot_dir = os.path.join(workdir, "plots", self.name)

        # Create the directories if they don't exist
        os.makedirs(sol_dir, exist_ok=True)

        self._copy_solutions(self.mode, sol_dir)
        field.scan_h5parms()

        if self.mode == "dd":  # uses solsetname + diagnostics
            solsetname = "coefficients000" if field.generate_screens else "sol000"
            flagged_frac = misc.get_flagged_solution_fraction(
                field.h5parm_filename,
                solsetname=solsetname,
            )

            field.calibration_diagnostics.append(
                {
                    "cycle_number": self.index,
                    "solution_flagged_fraction": flagged_frac,
                }
            )
        elif getattr(field, "fulljones_h5parm_filename", None) is not None:
            flagged_frac = misc.get_flagged_solution_fraction(field.fulljones_h5parm_filename)
        else:
            flagged_frac = misc.get_flagged_solution_fraction(field.h5parm_filename)
        # Log the fraction of flagged solutions.
        self.log.info("Fraction of solutions that are flagged = %.2f", flagged_frac)

        # Copy plots
        self._copy_plots(plot_filenames, plot_dir)

        # Finalize parent
        super().finalize()

    # finalize helper functions
    def _copy_solutions(self, mode, dst_dir):
        """
        Copy calibration solutions into destination directory.
        mode: "dd" or "di"
        """
        field = self.field
        if mode == "dd":
            self._copy_dd_solutions(dst_dir)

        elif mode == "di":
            solve_plan = getattr(self, "solve_plan", None) or self._build_solve_plan()
            fulljones_solve = next(
                (solve for solve in solve_plan if solve.solve_type == "full_jones"), None
            )
            scalar_solves = [solve for solve in solve_plan if solve.solve_type != "full_jones"]

            if fulljones_solve is not None:
                field.fulljones_h5parm_filename = os.path.join(dst_dir, "fulljones-solutions.h5")
                field.fulljones_h5parm_cycle_number = self.index
                if os.path.exists(field.fulljones_h5parm_filename):
                    os.remove(field.fulljones_h5parm_filename)
                collected_fulljones = getattr(
                    self, "collected_h5parm_fulljones", fulljones_solve.collected_h5parm
                )
                shutil.copy(
                    os.path.join(self.pipeline_working_dir, collected_fulljones),
                    field.fulljones_h5parm_filename,
                )

            if scalar_solves:
                field.di_h5parm_filename = os.path.join(dst_dir, "di-solutions.h5")
                field.h5parm_filename = field.di_h5parm_filename
                field.di_h5parm_cycle_number = self.index
                field.h5parm_cycle_number = self.index
                if os.path.exists(field.di_h5parm_filename):
                    os.remove(field.di_h5parm_filename)
                shutil.copy(
                    os.path.join(
                        self.pipeline_working_dir,
                        self._di_combined_solution(scalar_solves),
                    ),
                    field.di_h5parm_filename,
                )

                for solve in scalar_solves:
                    dst_filename = self._di_solution_destination(field, dst_dir, solve)
                    if os.path.exists(dst_filename):
                        os.remove(dst_filename)
                    shutil.copy(
                        os.path.join(self.pipeline_working_dir, solve.collected_h5parm),
                        dst_filename,
                    )

    def _copy_dd_solutions(self, dst_dir):
        field = self.field
        solve_plan = getattr(self, "solve_plan", None) or self._build_solve_plan()
        _, defaulted_strategy = self._requested_calibration_solves()

        field.h5parm_filename = os.path.join(dst_dir, "field-solutions.h5")
        field.dd_h5parm_filename = field.h5parm_filename
        field.h5parm_cycle_number = self.index
        field.dd_h5parm_cycle_number = self.index
        field.fast_phases_h5parm_filename = os.path.join(dst_dir, "field-solutions-fast-phase.h5")
        field.medium1_phases_h5parm_filename = os.path.join(
            dst_dir, "field-solutions-medium1-phase.h5"
        )
        field.medium2_phases_h5parm_filename = os.path.join(
            dst_dir, "field-solutions-medium2-phase.h5"
        )
        field.slow_gains_h5parm_filename = os.path.join(dst_dir, "field-solutions-slow-gain.h5")

        if os.path.exists(field.h5parm_filename):
            os.remove(field.h5parm_filename)

        active_solution = self._dd_active_solution(solve_plan)
        shutil.copy(
            os.path.join(self.pipeline_working_dir, active_solution),
            field.h5parm_filename,
        )

        if field.generate_screens:
            return

        if defaulted_strategy and not any(solve.solve_type == "slow_gains" for solve in solve_plan):
            solves_to_copy = solve_plan[:1]
        else:
            solves_to_copy = solve_plan

        for solve in solves_to_copy:
            dst_filename = self._dd_solution_destination(field, solve)
            if dst_filename is None:
                continue
            if os.path.exists(dst_filename):
                os.remove(dst_filename)
            shutil.copy(
                os.path.join(self.pipeline_working_dir, self._dd_collected_h5parm(solve)),
                dst_filename,
            )

    def _dd_active_solution(self, solve_plan):
        if self.field.generate_screens:
            return self.combined_h5parms
        if any(solve.solve_type == "slow_gains" and solve.slot == 3 for solve in solve_plan):
            return self.combined_h5parms
        return self._dd_collected_h5parm(solve_plan[0])

    def _dd_collected_h5parm(self, solve):
        if solve.solve_type == "fast_phase":
            return getattr(self, "fast_h5parm", solve.collected_h5parm)
        if solve.solve_type == "medium_phase":
            if solve.output_prefix.startswith("medium2"):
                return getattr(self, "medium2_h5parm", solve.collected_h5parm)
            return getattr(self, "medium1_h5parm", solve.collected_h5parm)
        if solve.solve_type == "slow_gains":
            return getattr(self, "slow_h5parm", solve.collected_h5parm)
        if solve.solve_type == "full_jones":
            return getattr(self, "collected_h5parm_fulljones", solve.collected_h5parm)
        raise ValueError(f"Unsupported DD solve type: {solve.solve_type}")

    @staticmethod
    def _dd_solution_destination(field, solve):
        if solve.solve_type == "fast_phase":
            return field.fast_phases_h5parm_filename
        if solve.solve_type == "medium_phase":
            if solve.output_prefix.startswith("medium2"):
                return field.medium2_phases_h5parm_filename
            return field.medium1_phases_h5parm_filename
        if solve.solve_type == "slow_gains":
            return field.slow_gains_h5parm_filename
        if solve.solve_type == "full_jones":
            return None
        raise ValueError(f"Unsupported DD solve type: {solve.solve_type}")

    def _di_combined_solution(self, scalar_solves):
        if any(solve.solve_type == "slow_gains" and solve.slot == 3 for solve in scalar_solves):
            return "combined_di_solutions.h5parm"
        if len(scalar_solves) > 1:
            return "combined_solve1_solve2_di.h5parm"
        return scalar_solves[0].collected_h5parm

    @staticmethod
    def _di_solution_destination(field, dst_dir, solve):
        if solve.solve_type == "fast_phase":
            field.di_fast_phases_h5parm_filename = os.path.join(
                dst_dir, "di-solutions-fast-phase.h5"
            )
            return field.di_fast_phases_h5parm_filename
        if solve.solve_type == "medium_phase":
            attr_name = (
                "di_medium2_phases_h5parm_filename"
                if solve.output_prefix.startswith("medium2")
                else "di_medium1_phases_h5parm_filename"
            )
            filename = (
                "di-solutions-medium2-phase.h5"
                if solve.output_prefix.startswith("medium2")
                else "di-solutions-medium1-phase.h5"
            )
            setattr(field, attr_name, os.path.join(dst_dir, filename))
            return getattr(field, attr_name)
        if solve.solve_type == "slow_gains":
            field.di_slow_gains_h5parm_filename = os.path.join(dst_dir, "di-solutions-slow-gain.h5")
            return field.di_slow_gains_h5parm_filename

        raise ValueError(f"Unsupported DI scalar solve type: {solve.solve_type}")

    @staticmethod
    def _copy_plots(plot_filenames, dst_dir):
        """
        Copy plots into destination directory.
        """
        os.makedirs(dst_dir, exist_ok=True)
        for plot_filename in plot_filenames:
            dst_filename = os.path.join(dst_dir, os.path.basename(plot_filename))
            if os.path.exists(dst_filename):
                os.remove(dst_filename)
            shutil.copy(plot_filename, dst_filename)
