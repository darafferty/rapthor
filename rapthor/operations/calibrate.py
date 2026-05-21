"""
Module that holds the Calibrate classes
"""

import glob
import logging
import os
import shutil

import lsmtool
import numpy as np

from rapthor.lib import miscellaneous as misc
from rapthor.lib.cwl import CWLDir, CWLFile
from rapthor.lib.operation import Operation

log = logging.getLogger("rapthor:calibrate")

DI_SOLVE_TYPE_ORDER = ["fast_phase", "medium_phase", "slow_gains", "full_jones"]

DI_SOLVE_TYPE_CONFIG = {
    "fast_phase": {
        "mode": "scalarphase",
        "solint_key": "solint_fast_timestep",
        "nchan_key": "solint_fast_freqstep",
        "chunk_output_stem": "fast_phase_di",
        "collected_h5parm": "fast_phase_di.h5parm",
        "field_attr": "fast_phases_h5parm_filename",
        "final_filename": "field-solutions-fast-phase.h5",
        "initial_h5_key": "fast_phases_h5parm_filename",
        "initial_soltab": "[phase000]",
        "plot_soltypes": ["phase"],
        "needs_process_gains": False,
    },
    "medium_phase": {
        "mode": "scalarphase",
        "solint_key": "solint_medium_timestep",
        "nchan_key": "solint_medium_freqstep",
        "chunk_output_stem": "medium1_phase_di",
        "collected_h5parm": "medium1_phase_di.h5parm",
        "field_attr": "medium1_phases_h5parm_filename",
        "final_filename": "field-solutions-medium1-phase.h5",
        "initial_h5_key": "medium1_phases_h5parm_filename",
        "initial_soltab": "[phase000]",
        "plot_soltypes": ["phase"],
        "needs_process_gains": False,
    },
    "slow_gains": {
        "mode": "diagonal",
        "solint_key": "solint_slow_timestep",
        "nchan_key": "solint_slow_freqstep",
        "chunk_output_stem": "slow_gains_di",
        "collected_h5parm": "slow_gains_di.h5parm",
        "processed_h5parm": "slow_gains_di.h5parm",
        "field_attr": "slow_gains_h5parm_filename",
        "final_filename": "field-solutions-slow-gain.h5",
        "initial_h5_key": "slow_gains_h5parm_filename",
        "initial_soltab": "[phase000,amplitude000]",
        "plot_soltypes": ["phase", "amplitude"],
        "needs_process_gains": True,
    },
    "full_jones": {
        "mode": "fulljones",
        "solint_key": "solint_fulljones_timestep",
        "nchan_key": "solint_fulljones_freqstep",
        "chunk_output_stem": "fulljones_gain",
        "collected_h5parm": "fulljones_gains.h5",
        "processed_h5parm": "fulljones_gains.h5",
        "field_attr": "fulljones_h5parm_filename",
        "final_filename": "fulljones-solutions.h5",
        "initial_h5_key": None,
        "initial_soltab": "[phase000,amplitude000]",
        "plot_soltypes": ["phase", "amplitude"],
        "needs_process_gains": True,
    },
}


class Calibrate(Operation):
    """
    Class for performing the calibration operation, which runs the CWL workflow template for calibration.
    This class is used for both direction-dependent (DD) and direction-independent (DI) calibration, with
    the mode specified by the "mode" parameter in the constructor.
    """

    def __init__(self, mode, field, index):
        if mode not in ["di", "dd"]:
            raise ValueError(f"Only di and dd mode are supported, chosen: {mode}")
        super().__init__(field, index=index, name="calibrate_di" if mode == "di" else "calibrate")

        self.mode = mode

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
        elif self.mode == "di":
            self.di_solves = self._get_di_solves()
            self.parset_parms.update(
                {
                    "di_solves": self.di_solves,
                    "nr_di_solves": len(self.di_solves),
                    "has_slow_gains": "slow_gains" in self.di_solves,
                    "is_full_jones": self.di_solves == ["full_jones"],
                    "needs_combine_fast_medium": self.di_solves
                    in (
                        ["fast_phase", "medium_phase"],
                        ["fast_phase", "medium_phase", "slow_gains"],
                    ),
                    "needs_combine_slow": self.di_solves
                    == ["fast_phase", "medium_phase", "slow_gains"],
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
            calibration_skymodel_file = field.calibration_skymodel_file
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
            for key in ("fast", "medium", "slow"):
                dd_factor_key = f"{key}_smoothness_dd_factors"
                constraint_key = f"{key}_smoothnessconstraint"
                dd_factor = smoothness_dd_factors[dd_factor_key] = field.get_obs_parameters(
                    dd_factor_key
                )
                smoothness_constraints[constraint_key] = getattr(field, constraint_key) / np.min(
                    dd_factor
                )
            # Antenna constraints
            core_stations = self._get_core_stations()
            fast_antennaconstraint = f"[[{','.join(core_stations)}]]" if core_stations else "[]"
            medium_antennaconstraint = fast_antennaconstraint  # ???

            # --- DP3 pipeline steps ---
            dp3_steps = self._build_dp3_steps(
                field.calibrate_bda_timebase, field.calibrate_bda_frequencybase
            )

            # --- Build final CWL input dict ---
            self.input_parms = {
                # File inputs / basic run configuration
                # Get the filenames of the input files for each time chunk
                "timechunk_filename": CWLDir(
                    field.get_obs_parameters("timechunk_filename")
                ).to_json(),
                "data_colname": field.data_colname,
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
                "fast_solutions_per_direction": field.get_obs_parameters(
                    "fast_solutions_per_direction"
                ),
                "medium_solutions_per_direction": field.get_obs_parameters(
                    "medium_solutions_per_direction"
                ),
                "slow_solutions_per_direction": field.get_obs_parameters(
                    "slow_solutions_per_direction"
                ),
                # Calibration outputs (H5parm products)
                "calibrator_patch_names": field.calibrator_patch_names,
                "calibrator_fluxes": field.calibrator_fluxes,
                "output_fast_h5parm": [f"fast_phase_{i}.h5parm" for i in range(field.ntimechunks)],
                "collected_fast_h5parm": self.fast_h5parm,
                "output_medium1_h5parm": [
                    f"medium1_phase_{i}.h5parm" for i in range(field.ntimechunks)
                ],
                "output_medium2_h5parm": [
                    f"medium2_phase_{i}.h5parm" for i in range(field.ntimechunks)
                ],
                "collected_medium1_h5parm": self.medium1_h5parm,
                "collected_medium2_h5parm": self.medium2_h5parm,
                "combined_fast_medium1_h5parm": "combined_fast_medium1_phases.h5parm",
                "combined_fast_medium1_medium2_h5parm": "combined_fast_medium1_medium2_phases.h5parm",
                "output_slow_h5parm": [f"slow_gain_{i}.h5parm" for i in range(field.ntimechunks)],
                "collected_slow_h5parm": self.slow_h5parm,
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
                "fast_smoothnessreffrequency": field.get_obs_parameters(
                    "fast_smoothnessreffrequency"
                ),
                "medium_smoothnessreffrequency": field.get_obs_parameters(
                    "medium_smoothnessreffrequency"
                ),
                "fast_smoothnessrefdistance": field.fast_smoothnessrefdistance,
                "medium_smoothnessrefdistance": field.medium_smoothnessrefdistance,
                # Applycal / DP3 control flow
                "dp3_steps": f"[{','.join(dp3_steps)}]",
                # --- Applycal + H5parm inputs ---
                # Set the DP3 applycal steps and input H5parm files depending on what
                # solutions need to be applied. Note: applycal steps are needed for
                # both the case in which applycal is part of the DDECal solve step and
                # the case in which it is a separate step that preceeds the DDECal step.
                # The latter is used when image-based predict is done
                **self._build_applycal(field),
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
                "medium2_initialsolutions_h5parm": self._to_cwl_json_if_exists(
                    field.medium2_phases_h5parm_filename
                ),
                "slow_initialsolutions_h5parm": self._to_cwl_json_if_exists(
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
                "fast_datause": field.fast_datause,
                "medium_datause": field.medium_datause,
                "slow_datause": field.slow_datause,
                "solverlbfgs_dof": field.solverlbfgs_dof,
                "solverlbfgs_iter": field.solverlbfgs_iter,
                "solverlbfgs_minibatches": field.solverlbfgs_minibatches,
                # ------------------------------------
                # Get the size of the imaging area (for use in making the a-term images)
                "sector_bounds_deg": str(field.sector_bounds_deg),
                "sector_bounds_mid_deg": str(field.sector_bounds_mid_deg),
                "combined_h5parms": self.combined_h5parms,
                "fast_antennaconstraint": fast_antennaconstraint,
                "medium_antennaconstraint": medium_antennaconstraint,
                "slow_antennaconstraint": "[]",
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
        elif self.mode == "di":
            self.di_solves = self._get_di_solves()
            if not self.di_solves:
                self.input_parms = {}
                return

            self.input_parms = {
                "timechunk_filename": CWLDir(
                    field.get_obs_parameters("predict_di_output_filename")
                ).to_json(),
                "data_colname": "DATA",
                "starttime": starttime,
                "ntimes": ntimes,
                "steps": f"[{','.join(f'solve{i}' for i in range(1, len(self.di_solves) + 1))}]",
                "maxiter": field.maxiter,
                "llssolver": field.llssolver,
                "propagatesolutions": field.propagatesolutions,
                "solveralgorithm": field.solveralgorithm,
                "stepsize": field.stepsize,
                "stepsigma": field.stepsigma,
                "tolerance": field.tolerance,
                "uvlambdamin": field.solve_min_uv_lambda,
                "solverlbfgs_dof": field.solverlbfgs_dof,
                "solverlbfgs_iter": field.solverlbfgs_iter,
                "solverlbfgs_minibatches": field.solverlbfgs_minibatches,
                "correctfreqsmearing": field.correct_smearing_in_calibration,
                "correcttimesmearing": field.correct_smearing_in_calibration,
                "max_threads": self.parset["cluster_specific"]["max_threads"],
                "max_normalization_delta": field.max_normalization_delta,
                "scale_normalization_delta": str(field.scale_normalization_delta),
                "phase_center_ra": field.ra,
                "phase_center_dec": field.dec,
                "calibrator_names": field.calibrator_patch_names,
                "calibrator_fluxes": field.calibrator_fluxes,
                "final_h5parm": self._get_di_final_h5parm(),
            }

            for solve_index, solve_type in enumerate(self.di_solves, start=1):
                config = DI_SOLVE_TYPE_CONFIG[solve_type]
                prefix = f"solve{solve_index}"
                initial_h5_key = config["initial_h5_key"]
                initial_h5parm = (
                    None
                    if initial_h5_key is None
                    else self._to_cwl_json_if_exists(getattr(field, initial_h5_key, None))
                )

                self.input_parms.update(
                    {
                        f"{prefix}_mode": config["mode"],
                        f"{prefix}_h5parm": [
                            f"{config['chunk_output_stem']}_{i}.h5parm"
                            for i in range(field.ntimechunks)
                        ],
                        f"{prefix}_solint": field.get_obs_parameters(config["solint_key"]),
                        f"{prefix}_nchan": field.get_obs_parameters(config["nchan_key"]),
                        f"{prefix}_collected_h5parm": config["collected_h5parm"],
                        f"{prefix}_initialsolutions_h5parm": initial_h5parm,
                        f"{prefix}_initialsolutions_soltab": config["initial_soltab"],
                        f"{prefix}_llssolver": field.llssolver,
                        f"{prefix}_maxiter": field.maxiter,
                        f"{prefix}_propagatesolutions": field.propagatesolutions,
                        f"{prefix}_solveralgorithm": field.solveralgorithm,
                        f"{prefix}_solverlbfgs_dof": field.solverlbfgs_dof,
                        f"{prefix}_solverlbfgs_iter": field.solverlbfgs_iter,
                        f"{prefix}_solverlbfgs_minibatches": field.solverlbfgs_minibatches,
                        f"{prefix}_stepsize": field.stepsize,
                        f"{prefix}_stepsigma": field.stepsigma,
                        f"{prefix}_tolerance": field.tolerance,
                        f"{prefix}_uvlambdamin": field.solve_min_uv_lambda,
                    }
                )
                if config["needs_process_gains"]:
                    self.input_parms[f"{prefix}_processed_h5parm"] = config["processed_h5parm"]

            if len(self.di_solves) > 1:
                self.input_parms["solve1_keepmodel"] = "True"
                for solve_index in range(2, len(self.di_solves) + 1):
                    self.input_parms[f"solve{solve_index}_reusemodel"] = "[solve1.*]"

            combine_plan = self._get_di_combine_plan()
            if combine_plan:
                self.input_parms["combined_fast_medium_h5parm"] = "di_fast_medium.h5parm"
                self.input_parms["solution_combine_mode"] = combine_plan[-1]["mode"]
            if len(combine_plan) > 1:
                self.input_parms["combined_slow_h5parm"] = "di_solutions.h5"

    def _get_di_solves(self):
        strategy = getattr(self.field, "calibration_strategy", None)
        if strategy is None:
            return ["full_jones"]

        requested_solves = strategy.get("di", []) or []
        return [solve for solve in DI_SOLVE_TYPE_ORDER if solve in requested_solves]

    def _get_di_final_h5parm(self):
        di_solves = self._get_di_solves()
        if not di_solves:
            return None
        if di_solves == ["full_jones"]:
            return DI_SOLVE_TYPE_CONFIG["full_jones"]["processed_h5parm"]
        if len(di_solves) == 1:
            config = DI_SOLVE_TYPE_CONFIG[di_solves[0]]
            if config["needs_process_gains"]:
                return config["processed_h5parm"]
            return config["collected_h5parm"]
        if di_solves == ["fast_phase", "medium_phase"]:
            return "di_fast_medium.h5parm"
        if di_solves == ["fast_phase", "medium_phase", "slow_gains"]:
            return "di_solutions.h5"
        raise ValueError(f"Unsupported DI calibration solve combination: {di_solves}")

    def _get_di_combine_plan(self):
        di_solves = self._get_di_solves()
        if di_solves in (
            [],
            ["fast_phase"],
            ["medium_phase"],
            ["slow_gains"],
            ["full_jones"],
        ):
            return []
        if di_solves == ["fast_phase", "medium_phase"]:
            return [
                {
                    "mode": "p1p2_scalar",
                    "output_h5parm": "di_fast_medium.h5parm",
                }
            ]
        if di_solves == ["fast_phase", "medium_phase", "slow_gains"]:
            return [
                {
                    "mode": "p1p2_scalar",
                    "output_h5parm": "di_fast_medium.h5parm",
                },
                {
                    "mode": (
                        "p1p2a2_diagonal"
                        if self.field.apply_diagonal_solutions
                        else "p1p2a2_scalar"
                    ),
                    "output_h5parm": "di_solutions.h5",
                },
            ]
        raise ValueError(f"Unsupported DI calibration solve combination: {di_solves}")

    def _build_dp3_steps(self, bda_timebase, bda_frequencybase):
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
        if (
            (bda_timebase > 0 or bda_frequencybase > 0)
            and all_regular
            and not self.field.use_image_based_predict
        ):
            if self.field.do_slowgain_solve:
                common_steps = ["avg", "solve1", "solve2", "solve3", "solve4", "null"]
            else:
                common_steps = ["avg", "solve1", "solve2", "null"]
        else:
            if self.field.do_slowgain_solve:
                common_steps = ["solve1", "solve2", "solve3", "solve4"]
            else:
                common_steps = ["solve1", "solve2"]

        # Optional image-based predict prefix
        if self.field.use_image_based_predict:
            # Add a predict, applybeam, and applycal steps to the beginning
            preprocessing_steps = (
                ["predict", "applybeam", "applycal"]
                if self.field.apply_normalizations
                else ["predict", "applybeam"]
            )
            dp3_steps = preprocessing_steps + common_steps
        else:
            dp3_steps = common_steps

        return dp3_steps

    def _build_applycal(self, field):
        """
        Prepare DP3 applycal steps and normalization H5parm.
        """
        if field.apply_normalizations:
            return {
                "normalize_h5parm": self._to_cwl_json_if_exists(field.normalize_h5parm),
                "ddecal_applycal_steps": "[normalization]",
                "applycal_steps": "[normalization]",
            }
        return dict.fromkeys(["ddecal_applycal_steps", "normalize_h5parm", "applycal_steps"], None)

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
        else:
            if self._get_di_solves() == ["full_jones"]:
                flagged_h5parm = field.fulljones_h5parm_filename
            else:
                flagged_h5parm = field.h5parm_filename
            flagged_frac = misc.get_flagged_solution_fraction(flagged_h5parm)
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
            # dd solutions
            field.h5parm_filename = os.path.join(dst_dir, "field-solutions.h5")
            field.fast_phases_h5parm_filename = os.path.join(
                dst_dir, "field-solutions-fast-phase.h5"
            )
            field.medium1_phases_h5parm_filename = os.path.join(
                dst_dir, "field-solutions-medium1-phase.h5"
            )
            field.medium2_phases_h5parm_filename = os.path.join(
                dst_dir, "field-solutions-medium2-phase.h5"
            )
            field.slow_gains_h5parm_filename = os.path.join(dst_dir, "field-solutions-slow-gain.h5")

            if os.path.exists(field.h5parm_filename):
                os.remove(field.h5parm_filename)

            if field.generate_screens:
                shutil.copy(
                    os.path.join(self.pipeline_working_dir, self.combined_h5parms),
                    field.h5parm_filename,
                )

            elif field.do_slowgain_solve:
                shutil.copy(
                    os.path.join(self.pipeline_working_dir, self.combined_h5parms),
                    field.h5parm_filename,
                )

                shutil.copy(
                    os.path.join(self.pipeline_working_dir, self.slow_h5parm),
                    field.slow_gains_h5parm_filename,
                )

                shutil.copy(
                    os.path.join(self.pipeline_working_dir, self.medium1_h5parm),
                    field.medium1_phases_h5parm_filename,
                )

                shutil.copy(
                    os.path.join(self.pipeline_working_dir, self.medium2_h5parm),
                    field.medium2_phases_h5parm_filename,
                )

                shutil.copy(
                    os.path.join(self.pipeline_working_dir, self.fast_h5parm),
                    field.fast_phases_h5parm_filename,
                )

            else:
                shutil.copy(
                    os.path.join(self.pipeline_working_dir, self.fast_h5parm), field.h5parm_filename
                )

                shutil.copy(
                    os.path.join(self.pipeline_working_dir, self.fast_h5parm),
                    field.fast_phases_h5parm_filename,
                )

        elif mode == "di":
            di_solves = self._get_di_solves()
            if not di_solves:
                return

            def copy_solution(src_name, dst_filename):
                if os.path.exists(dst_filename):
                    os.remove(dst_filename)
                shutil.copy(os.path.join(self.pipeline_working_dir, src_name), dst_filename)

            if di_solves == ["full_jones"]:
                field.fulljones_h5parm_filename = os.path.join(
                    dst_dir, DI_SOLVE_TYPE_CONFIG["full_jones"]["final_filename"]
                )
                copy_solution(self._get_di_final_h5parm(), field.fulljones_h5parm_filename)
                return

            field.h5parm_filename = os.path.join(dst_dir, "field-solutions-di.h5")
            copy_solution(self._get_di_final_h5parm(), field.h5parm_filename)

            for solve_type in di_solves:
                config = DI_SOLVE_TYPE_CONFIG[solve_type]
                solve_h5parm = (
                    config["processed_h5parm"]
                    if config["needs_process_gains"]
                    else config["collected_h5parm"]
                )
                dst_filename = os.path.join(dst_dir, config["final_filename"])
                setattr(field, config["field_attr"], dst_filename)
                copy_solution(solve_h5parm, dst_filename)

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
