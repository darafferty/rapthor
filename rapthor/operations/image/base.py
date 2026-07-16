"""
Base image operation adapter.
"""

import json
import logging
import os
from typing import List, Union

from rapthor.execution.image.builders import image_payload_from_inputs
from rapthor.execution.image.flow import image_flow
from rapthor.lib import miscellaneous as misc
from rapthor.lib.operation import Operation
from rapthor.lib.records import DirectoryRecord, FileRecord
from rapthor.operations.flow_execution import run_prefect_flow
from rapthor.operations.image.diagnostics import report_sector_diagnostics
from rapthor.operations.image.plan import (
    adjust_parallel_gridding_tasks,
    build_image_applycal_steps,
    build_image_facet_solution_controls,
    build_image_mpi_resource_controls,
    build_image_prepare_data_steps,
    build_image_screen_interval,
    build_image_wsclean_control_inputs,
    is_only_pol_I,
)

log = logging.getLogger("rapthor:image")

NON_FULLJONES_SOLVES = {"fast_phase", "medium_phase", "slow_gains"}


class Image(Operation):
    """
    Operation to image a field sector
    """

    def __init__(self, field, index, name="image"):
        super().__init__(field, index=index, name=name)

        # Initialize various parameters
        # Note:
        #   Parameters set to None will be set in the set_parset_parameters()
        #       method as needed for the given imaging mode
        #   Paramters set to True or False must be explicitly set by a subclass
        self.apply_amplitudes = None
        self.apply_fulljones = None
        self.apply_normalizations = None
        self.preapply_dd_solutions = None
        self.apply_screens = None
        self.dde_method = None
        self.use_facets = None
        self.save_source_list = None
        self.image_pol: Union[List[str], str, None] = None
        self.peel_bright_sources = None
        self.imaging_sectors = None
        self.imaging_parameters = None
        self.do_predict = None
        self.do_multiscale_clean = None
        self.pol_combine_method = None
        self.disable_clean = None
        # no solutions applied before or during imaging (ImageInitial only)
        self.apply_none = False
        self.make_image_cube = self.field.make_image_cube  # make an image cube
        self.make_residual_visibilities = getattr(self.field, "make_residual_visibilities", False)
        self.normalize_flux_scale = False  # derive flux scale normalizations (ImageNormalize only)
        self.compress_images = None
        self.image_cube_stokes_list = None
        self.allow_internet_access = True
        self.photometry_skymodel = None
        self.astrometry_skymodel = None

    def set_parset_parameters(self):
        """
        Define parameters needed by the image flow.
        """
        # Set parameters as needed
        if self.apply_screens is None:
            self.apply_screens = self.field.apply_screens  # set by process-step orchestration
        if self.dde_method is None:
            self.dde_method = self.field.dde_method
        if self.use_facets is None:
            self.use_facets = (
                self.dde_method == "full"
                and not self.apply_screens
                and self._has_dd_h5parm_for_facet_application()
            )
        if self.image_pol is None:
            self.image_pol = self.field.image_pol  # set by process-step orchestration
        if self.save_source_list is None:
            self.save_source_list = is_only_pol_I(self.image_pol)
        if self.preapply_dd_solutions is None:
            self.preapply_dd_solutions = self.dde_method == "single" and not self.apply_none
        if self.compress_images is None:
            self.compress_images = self.field.compress_images
        if self.image_cube_stokes_list is None:
            self.image_cube_stokes_list = self.field.image_cube_stokes_list
        if self.photometry_skymodel is None:
            self.photometry_skymodel = self.field.photometry_skymodel
        if self.astrometry_skymodel is None:
            self.astrometry_skymodel = self.field.astrometry_skymodel
        self.allow_internet_access = self.field.parset["cluster_specific"]["allow_internet_access"]
        self.parset_parms = self.flow_parset_parameters(
            include_pipeline_working_dir=True,
            apply_screens=self.apply_screens,
            make_image_cube=self.make_image_cube,
            make_residual_visibilities=self.make_residual_visibilities,
            normalize_flux_scale=self.normalize_flux_scale,
            use_facets=self.use_facets,
            save_source_list=self.save_source_list,
            preapply_dd_solutions=self.preapply_dd_solutions,
            use_mpi=self.field.use_mpi,
            compress_images=self.compress_images,
            image_cube_stokes_list=self.image_cube_stokes_list,
            photometry_skymodel=self.photometry_skymodel,
            astrometry_skymodel=self.astrometry_skymodel,
            allow_internet_access=self.allow_internet_access,
        )

    def _has_dd_h5parm_for_facet_application(self):
        if self._is_current_cycle_solution(
            getattr(self.field, "dd_h5parm_filename", None),
            cycle_attr="dd_h5parm_cycle_number",
        ):
            return True
        if getattr(self.field, "di_h5parm_filename", None) is not None:
            return False
        if self._fallback_h5parm_role() != "dd":
            return False
        return self._is_current_cycle_solution(
            self.field.h5parm_filename,
            cycle_attr="h5parm_cycle_number",
        )

    def _fallback_h5parm_role(self):
        """
        Return how to treat ``input_h5parm`` when it has no explicit DI/DD owner.

        A supplied non-full-Jones h5parm can only be applied as DD during
        imaging when a matching sky model or facet layout provides solution
        directions. If the current strategy asks for DI application and there
        is no DD direction model to pair with the h5parm, use DP3 pre-apply
        instead.
        """
        if self.field.h5parm_filename is None:
            return None
        strategy = getattr(self.field, "calibration_strategy", None) or {}
        has_di_non_fulljones = any(
            solve in NON_FULLJONES_SOLVES for solve in strategy.get("di", [])
        )
        has_dd_non_fulljones = any(
            solve in NON_FULLJONES_SOLVES for solve in strategy.get("dd", [])
        )
        has_dd_directions = bool(
            self.field.parset.get("input_skymodel") or self.field.parset.get("facet_layout")
        )
        if has_di_non_fulljones and (not has_dd_non_fulljones or not has_dd_directions):
            return "di"
        return "dd"

    def _can_use_carried_forward_solution(self, cycle_number):
        return (
            bool(getattr(self.field, "do_image", False))
            and not bool(getattr(self.field, "do_calibrate", True))
            and cycle_number < self.index
        )

    def _facet_skymodel_file(self):
        imaging_h5parm = getattr(self, "_selected_imaging_h5parm", None)
        if imaging_h5parm is None:
            return self.field.calibration_skymodel_file

        cycle_number = self.field.solution_cycle_number(
            imaging_h5parm,
            "dd_h5parm_cycle_number",
        )
        if not self._can_use_carried_forward_solution(cycle_number):
            return self.field.calibration_skymodel_file

        previous_skymodel = getattr(self.field, "calibration_skymodel_file_prev_cycle", None)
        if previous_skymodel is not None and os.path.exists(previous_skymodel):
            return previous_skymodel
        return self.field.calibration_skymodel_file

    def _resolve_non_fulljones_h5parms(self):
        dd_h5parm = getattr(self.field, "dd_h5parm_filename", None)
        di_h5parm = getattr(self.field, "di_h5parm_filename", None)
        fallback_h5parm = self.field.h5parm_filename
        if not self._is_current_cycle_solution(dd_h5parm, cycle_attr="dd_h5parm_cycle_number"):
            dd_h5parm = None
        if not self._is_current_cycle_solution(di_h5parm, cycle_attr="di_h5parm_cycle_number"):
            di_h5parm = None
        if not self._is_current_cycle_solution(
            fallback_h5parm,
            cycle_attr="h5parm_cycle_number",
        ):
            fallback_h5parm = None

        if fallback_h5parm is not None:
            fallback_role = self._fallback_h5parm_role()
            if fallback_role == "di" and di_h5parm is None:
                di_h5parm = fallback_h5parm
            elif fallback_role == "dd":
                if dd_h5parm is None:
                    dd_h5parm = fallback_h5parm
                if di_h5parm is None and fallback_h5parm != dd_h5parm:
                    di_h5parm = fallback_h5parm
        return dd_h5parm, di_h5parm

    def _resolve_fulljones_h5parm(self):
        fulljones_h5parm = getattr(self.field, "fulljones_h5parm_filename", None)
        if not self._is_current_cycle_solution(
            fulljones_h5parm,
            cycle_attr="fulljones_h5parm_cycle_number",
        ):
            return None
        return fulljones_h5parm

    def _is_current_cycle_solution(self, h5parm_filename, cycle_attr):
        if h5parm_filename is None:
            return False

        cycle_number = self.field.solution_cycle_number(h5parm_filename, cycle_attr)
        if (
            cycle_number is None
            or cycle_number == self.index
            or self._can_use_carried_forward_solution(cycle_number)
        ):
            return True

        log.warning(
            "Ignoring h5parm %r for image cycle %i because it was produced in cycle %i",
            h5parm_filename,
            self.index,
            cycle_number,
        )
        return False

    def _shared_facet_rw_enabled(self):
        return bool(
            self.use_facets
            and self.parset["imaging_specific"]["shared_facet_rw"]
            and self._facet_work_units() > 1
        )

    def _facet_work_units(self):
        return max(0, int(getattr(self.field, "num_patches", 0) or 0))

    def _parallel_gridding_tasks_for_sector(self, sector_index, channels_out):
        requested_tasks = self.field.parset["cluster_specific"]["parallel_gridding_tasks"]
        max_cores = self.field.parset["cluster_specific"]["max_cores"]
        channels_out_per_node = int(channels_out)
        if self.field.use_mpi:
            mpi_nnodes = int(self.input_parms["mpi_nnodes"][sector_index])
            channels_out_per_node = max(1, channels_out_per_node // mpi_nnodes)

        facet_work_units = self._facet_work_units()
        max_work_units = (
            facet_work_units if self.use_facets and facet_work_units > 1 else channels_out_per_node
        )
        return adjust_parallel_gridding_tasks(max_cores, requested_tasks, max_work_units)

    def _build_applycal_steps(self):
        """
        Build the DP3 applycal steps for the prepare-imaging-data stage.

        The steps are determined from the solve-type flags set on the
        field by the calibration strategy. Unlike DD calibration pre-apply,
        imaging preparation may apply separate final non-full-Jones products,
        so an explicit ``mediumphase`` step is valid here when a DD
        medium-phase solution is selected.
        """
        if self.apply_none:
            self._selected_applycal_h5parm = None
            self._selected_imaging_h5parm = None
            return None, None, None, None

        fulljones_h5parm = None
        input_normalize_h5parm = None
        dd_h5parm, di_h5parm = self._resolve_non_fulljones_h5parms()
        fulljones_h5parm_path = self._resolve_fulljones_h5parm()
        self._selected_imaging_h5parm = dd_h5parm if self.use_facets else None
        prepare_dd_h5parm = None if self.use_facets else dd_h5parm
        steps, self._selected_applycal_h5parm = build_image_applycal_steps(
            getattr(self.field, "calibration_strategy", None),
            dd_h5parm=prepare_dd_h5parm,
            di_h5parm=di_h5parm,
            has_fulljones_h5parm=fulljones_h5parm_path is not None,
            use_facets=self.use_facets,
            apply_amplitudes=self.apply_amplitudes,
            apply_normalizations=self.apply_normalizations,
            apply_none=self.apply_none,
            di_apply_amplitudes=(
                self.field.apply_amplitudes
                if di_h5parm == self.field.h5parm_filename
                and getattr(self.field, "di_h5parm_filename", None) is None
                else getattr(self.field, "di_apply_amplitudes", self.apply_amplitudes)
            ),
            dd_apply_amplitudes=self.apply_amplitudes,
        )

        if "fulljones" in steps:
            fulljones_h5parm = FileRecord(fulljones_h5parm_path).to_json()
        if "normalization" in steps:
            input_normalize_h5parm = FileRecord(self.field.normalize_h5parm).to_json()
        if len(steps) == 0:
            return None, None, fulljones_h5parm, input_normalize_h5parm

        formatted = f"[{','.join(steps)}]" if steps else None
        applycal_h5parm = (
            FileRecord(self._selected_applycal_h5parm).to_json()
            if self._selected_applycal_h5parm is not None
            else None
        )
        return formatted, applycal_h5parm, fulljones_h5parm, input_normalize_h5parm

    def _sector_observation_filenames(self, sector):
        if self.do_predict:
            return [obs.ms_imaging_filename for obs in sector.observations]
        return sector.get_obs_parameters("ms_filename")

    def _collect_sector_input_values(self):
        """Set per-sector imaging parameters and collect flow input lists."""
        values = {
            "obs_filename": [],
            "obs_original_filename": [],
            "prepare_filename": [],
            "concat_filename": [],
            "residual_filename": [],
            "previous_mask_filename": [],
            "mask_filename": [],
            "filtered_model_image_name": [],
            "starttime": [],
            "ntimes": [],
            "image_freqstep": [],
            "image_timestep": [],
            "image_maxinterval": [],
            "image_timebase": [],
            "image_minchannels": [],
            "image_frequencybase": [],
            "phasecenter": [],
            "image_name": [],
            "central_patch_name": [],
            "image_I_cube_name": [],
            "image_Q_cube_name": [],
            "image_U_cube_name": [],
            "image_V_cube_name": [],
            "output_source_catalog": [],
            "output_normalize_h5parm": [],
        }
        for sector in self.imaging_sectors:
            # IDG screen imaging requires square images; other modes keep the
            # existing size stable so cycles are easier to compare.
            sector.set_imaging_parameters(
                self.do_multiscale_clean,
                recalculate_imsize=self.apply_screens,
                imaging_parameters=self.imaging_parameters,
                preapply_dd_solutions=self.preapply_dd_solutions,
            )

            values["image_name"].append(sector.name)
            values["obs_filename"].append(self._sector_observation_filenames(sector))
            values["obs_original_filename"].append(sector.get_obs_parameters("ms_filename"))
            values["prepare_filename"].append(sector.get_obs_parameters("ms_prep_filename"))
            values["concat_filename"].append(f"{sector.name}_concat.ms")
            values["residual_filename"].append(f"{sector.name}_resid.ms")
            if self.field.parset["imaging_specific"]["use_clean_mask"] and sector.I_mask_file:
                values["previous_mask_filename"].append(sector.I_mask_file)
            else:
                values["previous_mask_filename"].append(None)
            values["mask_filename"].append(f"{sector.name}_mask.fits")
            values["filtered_model_image_name"].append(f"{sector.name}-MFS-filtered-model.fits.fz")
            values["image_freqstep"].append(sector.get_obs_parameters("image_freqstep"))
            values["image_timestep"].append(sector.get_obs_parameters("image_timestep"))
            values["image_maxinterval"].append(sector.get_obs_parameters("image_bda_maxinterval"))
            values["image_timebase"].append(self.field.image_bda_timebase)
            values["image_minchannels"].append(sector.get_obs_parameters("image_bda_minchannels"))
            values["image_frequencybase"].append(self.field.image_bda_frequencybase)
            values["starttime"].append(
                [misc.convert_mjd2mvt(obs.starttime) for obs in self.field.observations]
            )
            values["ntimes"].append([obs.numsamples for obs in self.field.observations])
            values["phasecenter"].append(f"'[{sector.ra}deg, {sector.dec}deg]'")
            if self.preapply_dd_solutions:
                values["central_patch_name"].append(sector.central_patch)
            if self.make_image_cube:
                values["image_I_cube_name"].append(f"{sector.name}_I_freq_cube.fits")
                values["image_Q_cube_name"].append(f"{sector.name}_Q_freq_cube.fits")
                values["image_U_cube_name"].append(f"{sector.name}_U_freq_cube.fits")
                values["image_V_cube_name"].append(f"{sector.name}_V_freq_cube.fits")
            if self.normalize_flux_scale:
                values["output_source_catalog"].append(f"{sector.name}_source_catalog.fits")
                values["output_normalize_h5parm"].append(f"{sector.name}_normalize.h5parm")
        return values

    def set_input_parameters(self):
        """
        Define inputs passed to the image flow.
        """
        # Set parameters as needed
        if self.imaging_sectors is None:
            self.imaging_sectors = self.field.imaging_sectors
        if self.imaging_parameters is None:
            self.imaging_parameters = self.field.parset["imaging_specific"].copy()
        if self.do_predict is None:
            self.do_predict = self.field.do_predict
        if self.do_multiscale_clean is None:
            self.do_multiscale_clean = self.field.do_multiscale_clean
        if self.pol_combine_method is None:
            self.pol_combine_method = self.field.pol_combine_method
        if self.apply_amplitudes is None:
            self.apply_amplitudes = self.field.apply_amplitudes  # set by CalibrateDD.finalize()
        if self.apply_fulljones is None:
            self.apply_fulljones = self.field.apply_fulljones  # set by CalibrateDI.finalize()
        if self.apply_normalizations is None:
            if self.normalize_flux_scale:
                self.apply_normalizations = False
            else:
                self.apply_normalizations = (
                    self.field.apply_normalizations
                )  # set by ImageNormalize.finalize()
        if self.peel_bright_sources is None:
            self.peel_bright_sources = self.field.peel_bright_sources
        nsectors = len(self.imaging_sectors)
        sector_inputs = self._collect_sector_input_values()

        link_polarizations, join_polarizations, wsclean_niter = build_image_wsclean_control_inputs(
            self.image_pol,
            self.pol_combine_method,
            [sector.wsclean_niter for sector in self.imaging_sectors],
            disable_clean=self.field.disable_clean,
        )

        # Set the DP3 steps and applycal steps depending on whether solutions
        # should be preapplied before imaging and on whether baseline-dependent
        # averaging is activated (and supported) or not
        (
            prepare_data_applycal_steps,
            prepare_data_h5parm,
            fulljones_h5parm,
            input_normalize_h5parm,
        ) = self._build_applycal_steps()
        all_regular = all(obs.channels_are_regular for obs in self.field.observations)
        prepare_data_steps = build_image_prepare_data_steps(
            preapply_solutions=prepare_data_applycal_steps is not None,
            average_visibilities=self.field.average_visibilities,
            image_bda_timebase=self.field.image_bda_timebase,
            image_bda_frequencybase=self.field.image_bda_frequencybase,
            all_channels_regular=all_regular,
            apply_screens=self.apply_screens,
        )
        prepare_data_steps = f"[{','.join(prepare_data_steps)}]"

        # Set the h5parm to use to apply the DD solutions as needed.
        imaging_h5parm = (
            self._selected_imaging_h5parm
            if getattr(self, "_selected_imaging_h5parm", None) is not None
            else self._selected_applycal_h5parm
        )
        h5parm = FileRecord(imaging_h5parm).to_json() if imaging_h5parm is not None else None
        first_observation = self.field.observations[0]
        interval = build_image_screen_interval(
            slow_timestep_sec=self.field.slow_timestep_sec,
            timepersample=first_observation.timepersample,
            numsamples=first_observation.numsamples,
        )
        # Set the parameters common to all modes
        self.input_parms = {
            "obs_filename": [
                DirectoryRecord(name).to_json() for name in sector_inputs["obs_filename"]
            ],
            "obs_original_filename": [
                DirectoryRecord(name).to_json() for name in sector_inputs["obs_original_filename"]
            ],
            "data_colname": self.field.data_colname,
            "prepare_filename": sector_inputs["prepare_filename"],
            "concat_filename": sector_inputs["concat_filename"],
            "residual_filename": sector_inputs["residual_filename"],
            "previous_mask_filename": [
                None if name is None else FileRecord(name).to_json()
                for name in sector_inputs["previous_mask_filename"]
            ],
            "mask_filename": sector_inputs["mask_filename"],
            "starttime": sector_inputs["starttime"],
            "ntimes": sector_inputs["ntimes"],
            "image_freqstep": sector_inputs["image_freqstep"],
            "image_timestep": sector_inputs["image_timestep"],
            "image_maxinterval": sector_inputs["image_maxinterval"],
            "image_timebase": sector_inputs["image_timebase"],
            "image_minchannels": sector_inputs["image_minchannels"],
            "image_frequencybase": sector_inputs["image_frequencybase"],
            "phasecenter": sector_inputs["phasecenter"],
            "image_name": sector_inputs["image_name"],
            "pol": self.image_pol,
            "save_source_list": self.save_source_list,
            "link_polarizations": link_polarizations,
            "join_polarizations": join_polarizations,
            "prepare_data_steps": prepare_data_steps,
            "prepare_data_applycal_steps": prepare_data_applycal_steps,
            "prepare_data_h5parm": prepare_data_h5parm,
            "h5parm": h5parm,
            "fulljones_h5parm": fulljones_h5parm,
            "input_normalize_h5parm": input_normalize_h5parm,
            "channels_out": [sector.wsclean_nchannels for sector in self.imaging_sectors],
            "deconvolution_channels": [
                sector.wsclean_deconvolution_channels for sector in self.imaging_sectors
            ],
            "fit_spectral_pol": [
                sector.wsclean_spectral_poly_order for sector in self.imaging_sectors
            ],
            "ra": [sector.ra for sector in self.imaging_sectors],
            "dec": [sector.dec for sector in self.imaging_sectors],
            "wsclean_imsize": [sector.imsize for sector in self.imaging_sectors],
            "vertices_file": [
                FileRecord(sector.vertices_file).to_json() for sector in self.imaging_sectors
            ],
            "region_file": [
                None if sector.region_file is None else FileRecord(sector.region_file).to_json()
                for sector in self.imaging_sectors
            ],
            "wsclean_niter": wsclean_niter,
            "wsclean_nmiter": [sector.wsclean_nmiter for sector in self.imaging_sectors],
            "skip_final_iteration": self.field.skip_final_major_iteration,
            "robust": [sector.robust for sector in self.imaging_sectors],
            "cellsize_deg": [sector.cellsize_deg for sector in self.imaging_sectors],
            "min_uv_lambda": [sector.min_uv_lambda for sector in self.imaging_sectors],
            "max_uv_lambda": [sector.max_uv_lambda for sector in self.imaging_sectors],
            "mgain": [sector.mgain for sector in self.imaging_sectors],
            "taper_arcsec": [sector.taper_arcsec for sector in self.imaging_sectors],
            "local_rms_strength": [sector.local_rms_strength for sector in self.imaging_sectors],
            "local_rms_window": [sector.local_rms_window for sector in self.imaging_sectors],
            "local_rms_method": [sector.local_rms_method for sector in self.imaging_sectors],
            "auto_mask": [sector.auto_mask for sector in self.imaging_sectors],
            "auto_mask_nmiter": [sector.auto_mask_nmiter for sector in self.imaging_sectors],
            "idg_mode": [sector.idg_mode for sector in self.imaging_sectors],
            "wsclean_mem": [sector.mem_limit_gb for sector in self.imaging_sectors],
            "threshisl": [sector.threshisl for sector in self.imaging_sectors],
            "threshpix": [sector.threshpix for sector in self.imaging_sectors],
            "filter_by_mask": self.imaging_parameters["filter_skymodel"],
            "source_finder": self.imaging_parameters["source_finder"],
            "do_multiscale": [sector.multiscale for sector in self.imaging_sectors],
            "dd_psf_grid": [sector.dd_psf_grid for sector in self.imaging_sectors],
            "apply_time_frequency_smearing": self.field.correct_smearing_in_imaging,
            "interval": interval,
            "max_threads": self.field.parset["cluster_specific"]["max_threads"],
            "filter_skymodel_ncores": self.field.parset["cluster_specific"].get(
                "filter_skymodel_ncores",
                self.field.parset["cluster_specific"]["max_threads"],
            ),
            "deconvolution_threads": self.field.parset["cluster_specific"]["deconvolution_threads"],
            "save_filtered_model_image": self.field.parset["imaging_specific"][
                "save_filtered_model_image"
            ],
            "make_residual_visibilities": self.make_residual_visibilities,
            "filtered_model_image_name": sector_inputs["filtered_model_image_name"],
            "allow_internet_access": self.allow_internet_access,
            "photometry_skymodel": (
                FileRecord(self.photometry_skymodel).to_json() if self.photometry_skymodel else None
            ),
            "astrometry_skymodel": (
                FileRecord(self.astrometry_skymodel).to_json() if self.astrometry_skymodel else None
            ),
            "peel_bright_sources": self.peel_bright_sources,
        }
        # Add parameters that depend on the set_parset parameters (set in set_parset_parameters())
        if self.peel_bright_sources:
            self.input_parms.update(
                {"bright_skymodel_pb": FileRecord(self.field.bright_source_skymodel_file).to_json()}
            )
        if self.field.use_mpi:
            self.input_parms.update(
                build_image_mpi_resource_controls(
                    nsectors=nsectors,
                    max_nodes=self.parset["cluster_specific"]["max_nodes"],
                    cpus_per_task=self.parset["cluster_specific"]["cpus_per_task"],
                    batch_system=self.batch_system,
                )
            )
        self.input_parms["shared_facet_rw"] = self._shared_facet_rw_enabled()
        self.input_parms["parallel_gridding_tasks"] = [
            self._parallel_gridding_tasks_for_sector(index, channels_out)
            for index, channels_out in enumerate(self.input_parms["channels_out"])
        ]
        if not self.apply_none and self.use_facets:
            # For faceting, we need inputs for making the ds9 facet region files
            self.input_parms.update({"skymodel": FileRecord(self._facet_skymodel_file()).to_json()})
            ra_mid = []
            dec_mid = []
            width_ra = []
            width_dec = []
            facet_region_file = []
            min_width = 2 * self.field.get_calibration_radius() * 1.2
            for sector in self.imaging_sectors:
                # Note: WSClean requires that all sources in the h5parm must have
                # corresponding regions in the facets region file. We ensure this
                # requirement is met by extending the regions to cover the larger of
                # the calibration region and the sector region, plus a 20% padding
                ra_mid.append(self.field.ra)
                dec_mid.append(self.field.dec)
                width_ra.append(max(min_width, sector.width_ra * 1.2))
                width_dec.append(max(min_width, sector.width_dec * 1.2))
                facet_region_file.append(f"{sector.name}_facets_ds9.reg")
            self.input_parms.update({"ra_mid": ra_mid})
            self.input_parms.update({"dec_mid": dec_mid})
            self.input_parms.update({"width_ra": width_ra})
            self.input_parms.update({"width_dec": width_dec})
            self.input_parms.update({"facet_region_file": facet_region_file})
            self.input_parms.update(
                build_image_facet_solution_controls(
                    self.image_pol,
                    apply_amplitudes=self.apply_amplitudes,
                    apply_diagonal_solutions=self.field.apply_diagonal_solutions,
                )
            )
        elif self.preapply_dd_solutions:
            self.input_parms.update({"central_patch_name": sector_inputs["central_patch_name"]})
        if self.make_image_cube:
            self.input_parms.update({"image_I_cube_name": sector_inputs["image_I_cube_name"]})
            self.input_parms.update({"image_Q_cube_name": sector_inputs["image_Q_cube_name"]})
            self.input_parms.update({"image_U_cube_name": sector_inputs["image_U_cube_name"]})
            self.input_parms.update({"image_V_cube_name": sector_inputs["image_V_cube_name"]})
        if self.normalize_flux_scale:
            self.input_parms.update(
                {"output_source_catalog": sector_inputs["output_source_catalog"]}
            )
            self.input_parms.update(
                {"output_normalize_h5parm": sector_inputs["output_normalize_h5parm"]}
            )

    def execute_workflow(self):
        """
        Execute imaging through the Prefect flow and return operation outputs.
        """
        payload = image_payload_from_inputs(
            self.input_parms,
            self.pipeline_working_dir,
            apply_screens=self.apply_screens,
            use_facets=self.use_facets,
            compress_images=self.compress_images,
            make_image_cube=self.make_image_cube,
            make_residual_visibilities=self.make_residual_visibilities,
            normalize_flux_scale=self.normalize_flux_scale,
            use_mpi=self.field.use_mpi,
        )
        outputs = run_prefect_flow(image_flow, payload, self.parset)
        return True, outputs

    def _record_sector_image_outputs(self, index, sector, leave_in_place):
        """Record image product paths on a sector and preserve files needed downstream."""
        file_list = [
            record["path"]
            for record in self.outputs["sector_I_images"][index]
            + self.outputs["sector_extra_images"][index]
        ]
        leave_in_place.update({"sector_I_images", "sector_extra_images"})
        if self.field.save_supplementary_images:
            file_list.append(self.outputs["source_filtering_mask"][index]["path"])
            leave_in_place.update({"source_filtering_mask"})
        if self.field.parset["imaging_specific"]["save_filtered_model_image"]:
            file_list.append(self.outputs["sector_skymodel_image_fits"][index]["path"])
            leave_in_place.update({"sector_skymodel_image_fits"})

        type_path_map = Image.find_in_file_list(file_list)
        for output_type, paths in type_path_map.items():
            if output_type not in ["filtering_mask_file", "filtered_model_file_apparent_sky"]:
                for path in paths:
                    pol = Image.derive_pol_from_filename(path)
                    setattr(sector, f"{pol}_{output_type}", path)
            else:
                for path in paths:
                    setattr(sector, output_type, path)

    def finalize(self):
        """
        Finalize this operation
        """
        # Save the output FITS image filenames, sky models, ds9 facet region file, and
        # visibilities (if desired) for each sector. Also read the image diagnostics (rms
        # noise, etc.) derived by PyBDSF and print them to the log. The images are not
        # copied to the final location here, as this is done after mosaicking (if needed)
        # by the mosaic operation
        self.field.lofar_to_true_flux_ratio = 1.0  # reset values for this cycle
        self.field.lofar_to_true_flux_std = 0.0

        leave_in_place = set()
        for index, sector in enumerate(self.field.imaging_sectors):
            self._record_sector_image_outputs(index, sector, leave_in_place)

            if self.field.parset["imaging_specific"]["use_clean_mask"]:
                # Copy the sector mask to save it for use in a subsequent imaging operation. Note
                # that, unlike the normal images above, this image is copied directly since
                # mosaiking is not needed.
                # The path to this file is saved in the sector's I_mask_file attribute
                dest_dir = os.path.join(
                    self.parset["dir_working"],
                    "images",
                    self.name,
                    sector.name,
                )
                self.copy_outputs_to(
                    dest_dir,
                    index=index,
                    include={"source_filtering_mask"},
                    move=False,
                )
                sector.I_mask_file = os.path.join(
                    dest_dir, os.path.basename(self.outputs["source_filtering_mask"][index]["path"])
                )
            else:
                sector.I_mask_file = None

            # Save the output image cubes. Note that, unlike the normal images above,
            # the cubes are copied directly since mosaicking of the cubes is not yet
            # supported
            image_cube_keys = {
                "sector_image_cubes",
                "sector_image_cube_beams",
                "sector_image_cube_frequencies",
            }
            if "sector_image_cubes" in self.outputs:
                self.copy_outputs_to(
                    os.path.join(self.parset["dir_working"], "images", self.name),
                    index=index,
                    include=image_cube_keys,
                    move=True,
                )

            # The output sky models. We also set the paths as attributes of the sector for later
            # use
            #
            # Note: these are not generated when QUV images are made (WSClean does not
            # currently support writing a source list in this mode)
            skymodel_dest_dir = os.path.join(
                self.parset["dir_working"], "skymodels", f"image_{self.index}"
            )
            if self.field.image_pol.lower() == "i":
                for skymodel_type in ["true_sky", "apparent_sky"]:
                    src_sector_skymodel = self.outputs[f"filtered_skymodel_{skymodel_type}"][index][
                        "path"
                    ]
                    sector_skymodel_file = os.path.join(
                        skymodel_dest_dir, os.path.basename(src_sector_skymodel)
                    )
                    setattr(sector, f"image_skymodel_file_{skymodel_type}", sector_skymodel_file)
                    self.copy_outputs_to(
                        skymodel_dest_dir,
                        index=index,
                        include={f"filtered_skymodel_{skymodel_type}"},
                        move=True,
                    )

            # The output PyBDSF source catalog
            self.copy_outputs_to(
                skymodel_dest_dir, index=index, include={"pybdsf_catalog"}, move=True
            )

            # The output ds9 region file, if made
            if self.use_facets:
                self.copy_outputs_to(
                    os.path.join(
                        self.parset["dir_working"],
                        "regions",
                        f"image_{self.index}",
                    ),
                    index=index,
                    include={"sector_region_file"},
                    move=True,
                )

            # The imaging visibilities
            if self.field.save_visibilities:
                self.copy_outputs_to(
                    os.path.join(
                        self.parset["dir_working"],
                        "visibilities",
                        f"image_{self.index}",
                        sector.name,
                    ),
                    index=index,
                    include={"visibilities"},
                    move=True,
                )
            if self.make_residual_visibilities:
                self.copy_outputs_to(
                    os.path.join(
                        self.parset["dir_working"],
                        "visibilities",
                        f"image_{self.index}",
                        sector.name,
                    ),
                    index=index,
                    include={"residual_visibilities"},
                    move=True,
                )

            # The astrometry and photometry plots and diagnostics file
            diagnostics_dest_dir = os.path.join(
                self.parset["dir_working"], "plots", f"image_{self.index}"
            )
            diagnotics = {"sector_diagnostics"}
            if self.outputs["sector_diagnostic_plots"][index]:
                diagnotics.update({"sector_diagnostic_plots"})
            self.copy_outputs_to(
                diagnostics_dest_dir,
                index=index,
                include=diagnotics,
                move=True,
            )

            # Read in the image diagnostics and log a summary of them
            diagnostics_file = os.path.join(
                diagnostics_dest_dir,
                os.path.basename(self.outputs["sector_diagnostics"][index]["path"]),
            )
            with open(diagnostics_file, "r") as f:
                diagnostics_dict = json.load(f)
            diagnostics_dict["cycle_number"] = self.index
            sector.diagnostics.append(diagnostics_dict)
            ratio, std = report_sector_diagnostics(sector.name, diagnostics_dict, self.log)
            if self.field.lofar_to_true_flux_std == 0.0 or std < self.field.lofar_to_true_flux_std:
                # Save the ratio with the lowest scatter for later use
                self.field.lofar_to_true_flux_ratio = ratio
                self.field.lofar_to_true_flux_std = std

        # Clean up other files
        self.clean_outputs(exclude=leave_in_place)

        # Finally call finalize() in the parent class
        super().finalize()

    @staticmethod
    def find_in_file_list(file_list):
        ext_mapping = {
            "image_file_true_sky": "image-pb.fits",
            "image_file_true_sky_astcorr": "image-pb-ast.fits",
            "image_file_apparent_sky": "image.fits",
            "model_file_true_sky": "model-pb.fits",
            "filtered_model_file_apparent_sky": "filtered-model.fits",
            "residual_file_apparent_sky": "residual.fits",
            "dirty_file_apparent_sky": "dirty.fits",
            "filtering_mask_file": "mask.fits",
        }
        type_path_map = {}
        for name, ext in ext_mapping.items():
            for filename in file_list:
                if filename.endswith(ext) or filename.endswith(ext + ".fz"):
                    if name in type_path_map:
                        type_path_map[name] += [filename]
                    else:
                        type_path_map[name] = [filename]
        return type_path_map

    @staticmethod
    def derive_pol_from_filename(filename):
        for pol in "IQUV":
            if f"-{pol}-" in filename:
                return pol
        return "I"  # default
