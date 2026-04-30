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
                    "do_slowgain_solve": self.field.do_slowgain_solve,
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
                dd_factor = smoothness_dd_factors[dd_factor_key] = field.field.get_obs_parameters(
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
            # Define various output filenames for the solution tables. We save some
            # as attributes since they are needed in finalize()
            self.collected_h5parm_fulljones = "fulljones_gains.h5"

            # Set the constraints used in the calibrations
            self.input_parms = {
                # Get the filenames of the input files for each time chunk. These are the
                # output of the predict_di pipeline done before this calibration
                "timechunk_filename_fulljones": CWLDir(
                    field.get_obs_parameters("predict_di_output_filename")
                ).to_json(),
                "data_colname": "DATA",
                "starttime_fulljones": starttime,
                "ntimes_fulljones": ntimes,
                # Get the solution intervals for the calibrations
                "solint_fulljones_timestep": field.get_obs_parameters("solint_fulljones_timestep"),
                "solint_fulljones_freqstep": field.get_obs_parameters("solint_fulljones_freqstep"),
                "output_h5parm_fulljones": [
                    f"fulljones_gain_{i}.h5parm" for i in range(field.ntimechunks)
                ],
                "collected_h5parm_fulljones": self.collected_h5parm_fulljones,
                "smoothnessconstraint_fulljones": field.smoothnessconstraint_fulljones,
                "max_normalization_delta": field.max_normalization_delta,
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
                # ---------------------------------
                "correctfreqsmearing": field.correct_smearing_in_calibration,
                "correcttimesmearing": field.correct_smearing_in_calibration,
                "max_threads": self.parset["cluster_specific"]["max_threads"],
            }

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

        return "[CR]*&&;!{}".format(";!".join(non_core))

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
        else:  # if self.mode == "di" uses fulljones file
            flagged_frac = misc.get_flagged_solution_fraction(field.fulljones_h5parm_filename)
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
            # di solutions
            field = self.field
            field.fulljones_h5parm_filename = os.path.join(dst_dir, "fulljones-solutions.h5")
            if os.path.exists(field.fulljones_h5parm_filename):
                os.remove(field.fulljones_h5parm_filename)
            shutil.copy(
                os.path.join(self.pipeline_working_dir, self.collected_h5parm_fulljones),
                field.fulljones_h5parm_filename,
            )

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


class CalibrateDI(Operation):
    """
    Operation to perform direction-independent (DI) calibration of the field
    """

    def __init__(self, field, index):
        super().__init__(field, index=index, name="calibrate_di")

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
        self.parset_parms = {
            "rapthor_pipeline_dir": self.rapthor_pipeline_dir,
            "max_cores": max_cores,
        }

    def set_input_parameters(self):
        """
        Define the CWL workflow inputs
        """
        # First set the calibration parameters for each observation
        self.field.set_obs_parameters()

        # Next, get the various parameters needed by the workflow
        #
        # Get the start times and number of times for the time chunks (fast and slow
        # calibration)
        starttime_fulljones = self.field.get_obs_parameters("starttime")
        ntimes_fulljones = self.field.get_obs_parameters("ntimes")

        # Get the filenames of the input files for each time chunk. These are the
        # output of the predict_di pipeline done before this calibration
        timechunk_filename_fulljones = self.field.get_obs_parameters("predict_di_output_filename")

        # Get the solution intervals for the calibrations
        solint_fulljones_timestep = self.field.get_obs_parameters("solint_fulljones_timestep")
        solint_fulljones_freqstep = self.field.get_obs_parameters("solint_fulljones_freqstep")

        # Define various output filenames for the solution tables. We save some
        # as attributes since they are needed in finalize()
        output_h5parm_fulljones = [
            "fulljones_gain_{}.h5parm".format(i) for i in range(self.field.ntimechunks)
        ]
        self.collected_h5parm_fulljones = "fulljones_gains.h5"

        # Set the constraints used in the calibrations
        smoothnessconstraint_fulljones = self.field.smoothnessconstraint_fulljones
        max_normalization_delta = self.field.max_normalization_delta

        # Get various DDECal solver parameters
        llssolver = self.field.llssolver
        maxiter = self.field.maxiter
        propagatesolutions = self.field.propagatesolutions
        solveralgorithm = self.field.solveralgorithm
        stepsize = self.field.stepsize
        stepsigma = self.field.stepsigma
        tolerance = self.field.tolerance
        uvlambdamin = self.field.solve_min_uv_lambda
        solverlbfgs_dof = self.field.solverlbfgs_dof
        solverlbfgs_iter = self.field.solverlbfgs_iter
        solverlbfgs_minibatches = self.field.solverlbfgs_minibatches

        self.input_parms = {
            "timechunk_filename_fulljones": CWLDir(timechunk_filename_fulljones).to_json(),
            "data_colname": "DATA",
            "starttime_fulljones": starttime_fulljones,
            "ntimes_fulljones": ntimes_fulljones,
            "solint_fulljones_timestep": solint_fulljones_timestep,
            "solint_fulljones_freqstep": solint_fulljones_freqstep,
            "output_h5parm_fulljones": output_h5parm_fulljones,
            "collected_h5parm_fulljones": self.collected_h5parm_fulljones,
            "smoothnessconstraint_fulljones": smoothnessconstraint_fulljones,
            "max_normalization_delta": max_normalization_delta,
            "llssolver": llssolver,
            "maxiter": maxiter,
            "propagatesolutions": propagatesolutions,
            "solveralgorithm": solveralgorithm,
            "stepsize": stepsize,
            "stepsigma": stepsigma,
            "tolerance": tolerance,
            "uvlambdamin": uvlambdamin,
            "solverlbfgs_dof": solverlbfgs_dof,
            "solverlbfgs_iter": solverlbfgs_iter,
            "solverlbfgs_minibatches": solverlbfgs_minibatches,
            "correctfreqsmearing": self.field.correct_smearing_in_calibration,
            "correcttimesmearing": self.field.correct_smearing_in_calibration,
            "max_threads": self.parset["cluster_specific"]["max_threads"],
        }

    def finalize(self):
        """
        Finalize this operation
        """
        # Copy the solutions (h5parm file) and report the flagged fraction
        dst_dir = os.path.join(
            self.parset["dir_working"], "solutions", "calibrate_di_{}".format(self.index)
        )
        os.makedirs(dst_dir, exist_ok=True)
        self.field.fulljones_h5parm_filename = os.path.join(dst_dir, "fulljones-solutions.h5")
        if os.path.exists(self.field.fulljones_h5parm_filename):
            os.remove(self.field.fulljones_h5parm_filename)
        shutil.copy(
            os.path.join(self.pipeline_working_dir, self.collected_h5parm_fulljones),
            os.path.join(dst_dir, self.field.fulljones_h5parm_filename),
        )
        self.field.scan_h5parms()  # verify h5parm and update flags for predict/image operations
        flagged_frac = misc.get_flagged_solution_fraction(self.field.fulljones_h5parm_filename)
        self.log.info("Fraction of solutions that are flagged = %.2f", flagged_frac)

        # Copy the plots (PNG files)
        dst_dir = os.path.join(
            self.parset["dir_working"], "plots", "calibrate_di_{}".format(self.index)
        )
        os.makedirs(dst_dir, exist_ok=True)
        plot_filenames = glob.glob(os.path.join(self.pipeline_working_dir, "*.png"))
        for plot_filename in plot_filenames:
            dst_filename = os.path.join(dst_dir, os.path.basename(plot_filename))
            if os.path.exists(dst_filename):
                os.remove(dst_filename)
            shutil.copy(plot_filename, dst_filename)

        # Finally call finalize() in the parent class
        super().finalize()
