"""
Module that holds the Calibrate classes
"""
import glob
import logging
import lsmtool
import numpy as np
import os
from rapthor.lib.operation import Operation
from rapthor.lib import miscellaneous as misc
from rapthor.lib.cwl import CWLFile, CWLDir
import shutil

log = logging.getLogger('rapthor:calibrate')


class CalibrateDD(Operation):
    """
    Operation to perform direction-dependent (DD) calibration of the field
    """
    def __init__(self, field, index):
        super().__init__(field, index=index, name='calibrate')

    def set_parset_parameters(self):
        """
        Define parameters needed for the CWL workflow template
        """
        if self.batch_system == 'slurm':
            # For some reason, setting coresMax ResourceRequirement hints does
            # not work with SLURM
            max_cores = None
        else:
            max_cores = self.parset['cluster_specific']['max_cores']

        # Set whether image-based prediction is used. Note that generation of
        # screens (IDGCal) requires image-based prediction
        self.use_image_based_predict = self.field.generate_screens or self.field.use_image_based_predict

        self.parset_parms = {'rapthor_pipeline_dir': self.rapthor_pipeline_dir,
                             'use_image_based_predict': self.use_image_based_predict,
                             'generate_screens': self.field.generate_screens,
                             'do_slowgain_solve': self.field.do_slowgain_solve,
                             'max_cores': max_cores}

    def set_input_parameters(self):
        """
        Define the CWL workflow inputs
        """
        # First set the calibration parameters for each observation
        self.field.set_obs_parameters()

        # Next, get the various parameters needed by the workflow
        #
        # Get the filenames of the input files for each time chunk
        timechunk_filename = self.field.get_obs_parameters('timechunk_filename')

        # Get the start times and number of times for the time chunks (fast and slow
        # calibration)
        starttime = self.field.get_obs_parameters('starttime')
        ntimes = self.field.get_obs_parameters('ntimes')

        # Get the solution intervals for the calibrations
        solint_fast_timestep = self.field.get_obs_parameters('solint_fast_timestep')
        solint_slow_timestep = self.field.get_obs_parameters('solint_slow_timestep')
        solint_fast_freqstep = self.field.get_obs_parameters('solint_fast_freqstep')
        solint_slow_freqstep = self.field.get_obs_parameters('solint_slow_freqstep')

        # Get the number of solutions per direction
        fast_solutions_per_direction = self.field.get_obs_parameters('fast_solutions_per_direction')
        slow_solutions_per_direction = self.field.get_obs_parameters('slow_solutions_per_direction')

        # Get the BDA (baseline-dependent averaging) parameters
        bda_maxinterval = self.field.get_obs_parameters('bda_maxinterval')
        bda_minchannels = self.field.get_obs_parameters('bda_minchannels')
        bda_timebase = self.field.bda_timebase
        bda_frequencybase = self.field.bda_frequencybase

        # Define various output filenames for the solution tables. We save some
        # as attributes since they are needed in finalize()
        output_fast_h5parm = ['fast_phase_{}.h5parm'.format(i)
                              for i in range(self.field.ntimechunks)]
        self.combined_fast_h5parm = 'fast_phases.h5parm'
        self.combined_h5parms = 'combined_solutions.h5'
        output_idgcal_h5parm = ['idgcal_{}.h5parm'.format(i)  # TODO: chunk the solve over frequency as well as time?
                                for i in range(self.field.ntimechunks)]
        output_slow_h5parm = ['slow_gain_{}.h5parm'.format(i)
                              for i in range(self.field.ntimechunks)]
        self.combined_slow_h5parm = 'slow_gains.h5parm'
        if self.field.apply_diagonal_solutions:
            solution_combine_mode = 'p1p2a2_diagonal'
        else:
            solution_combine_mode = 'p1p2a2_scalar'

        # Define the input sky model
        if self.field.peel_non_calibrator_sources:
            calibration_skymodel_file = self.field.calibrators_only_skymodel_file
        else:
            calibration_skymodel_file = self.field.calibration_skymodel_file
        num_spectral_terms = misc.get_max_spectral_terms(calibration_skymodel_file)
        model_image_root = 'calibration_model'
        model_image_frequency_bandwidth, model_image_ra_dec, model_image_imsize, model_image_cellsize = self.get_model_image_parameters()
        facet_region_width = max(model_image_imsize) * model_image_cellsize * 1.2  # deg
        facet_region_file = 'field_facets_ds9.reg'

        # Get the calibrator names and fluxes
        calibrator_patch_names = self.field.calibrator_patch_names
        calibrator_fluxes = self.field.calibrator_fluxes

        # Set the constraints used in the calibrations
        fast_smoothness_dd_factors = self.field.get_obs_parameters('fast_smoothness_dd_factors')
        slow_smoothness_dd_factors = self.field.get_obs_parameters('slow_smoothness_dd_factors')
        fast_smoothnessconstraint = self.field.fast_smoothnessconstraint / np.min(fast_smoothness_dd_factors)
        fast_smoothnessreffrequency = self.field.get_obs_parameters('fast_smoothnessreffrequency')
        fast_smoothnessrefdistance = self.field.fast_smoothnessrefdistance
        slow_smoothnessconstraint = (self.field.slow_smoothnessconstraint / np.min(slow_smoothness_dd_factors))
        if self.field.do_slowgain_solve or self.field.antenna == 'LBA':
            # Use the core stationconstraint if the slow solves will be done or if
            # we have LBA data (which has lower sensitivity than HBA data)
            core_stations = self.get_core_stations()
            # In the case of SKA sets, we currently have not defined core stations yet. To let
            # SKA continue, we disable the antennaconstraint for now if the list of core
            # stations is empty.
            if core_stations:
                fast_antennaconstraint = '[[{}]]'.format(','.join(core_stations))
            else:
                fast_antennaconstraint = '[]'
        else:
            # For HBA data, if the slow solves will not be done, we remove the
            # stationconstraint to allow each station to get its own fast phase
            # corrections
            fast_antennaconstraint = '[]'
        idgcal_antennaconstraint = '[]'  # TODO: set different constraints for phase and gain solves
        slow_antennaconstraint = '[]'
        max_normalization_delta = self.field.max_normalization_delta
        scale_normalization_delta = '{}'.format(self.field.scale_normalization_delta)

        # Get various DDECal solver parameters. Most of these are the same for both fast
        # and slow solves
        llssolver = self.field.llssolver
        maxiter = self.field.maxiter
        propagatesolutions = self.field.propagatesolutions
        solveralgorithm = self.field.solveralgorithm
        onebeamperpatch = self.field.onebeamperpatch
        stepsize = self.field.stepsize
        stepsigma = self.field.stepsigma
        tolerance = self.field.tolerance
        uvlambdamin = self.field.solve_min_uv_lambda
        parallelbaselines = self.field.parallelbaselines
        sagecalpredict = self.field.sagecalpredict
        solverlbfgs_dof = self.field.solverlbfgs_dof
        solverlbfgs_iter = self.field.solverlbfgs_iter
        solverlbfgs_minibatches = self.field.solverlbfgs_minibatches
        fast_datause = self.field.fast_datause
        slow_datause = self.field.slow_datause

        # Get the size of the imaging area (for use in making the a-term images)
        sector_bounds_deg = '{}'.format(self.field.sector_bounds_deg)
        sector_bounds_mid_deg = '{}'.format(self.field.sector_bounds_mid_deg)

        # Set the DDECal steps depending on whether baseline-dependent averaging is
        # activated (and supported) or not. If BDA is used, a "null" step is also added to
        # prevent the writing of the BDA data
        #
        # TODO: image-based predict doesn't yet work with BDA; once it does,
        # the restriction on this mode should be removed
        all_regular = all([obs.channels_are_regular for obs in self.field.observations])
        if self.field.bda_timebase > 0 and all_regular and not self.field.use_image_based_predict:
            if self.field.do_slowgain_solve:
                dp3_steps = ['avg', 'solve1', 'solve2', 'null']
            else:
                dp3_steps = ['avg', 'solve1', 'null']
        else:
            if self.field.do_slowgain_solve:
                dp3_steps = ['solve1', 'solve2']
            else:
                dp3_steps = ['solve1']
        if self.field.use_image_based_predict:
            # Add a predict, applybeam, and applycal steps to the beginning
            dp3_steps = (['predict', 'applybeam', 'applycal'] if self.field.apply_normalizations else
                         ['predict', 'applybeam']) + dp3_steps

        # Set the DP3 applycal steps and input H5parm files depending on what
        # solutions need to be applied. Note: applycal steps are needed for
        # both the case in which applycal is part of the DDECal solve step and
        # the case in which it is a separate step that preceeds the DDECal step.
        # The latter is used when image-based predict is done
        if self.field.apply_normalizations:
            normalize_h5parm = CWLFile(self.field.normalize_h5parm).to_json()
            ddecal_applycal_steps = ['normalization']
            applycal_steps = ['normalization']

            # Convert the lists to strings, with square brackets as required by DP3
            ddecal_applycal_steps = f"[{','.join(ddecal_applycal_steps)}]"
            applycal_steps = f"[{','.join(applycal_steps)}]"
        else:
            normalize_h5parm = None
            ddecal_applycal_steps = None
            applycal_steps = None
        if (
            self.field.fast_phases_h5parm_filename is not None and
            os.path.exists(self.field.fast_phases_h5parm_filename)
        ):
            fast_initialsolutions_h5parm = CWLFile(self.field.fast_phases_h5parm_filename).to_json()
        else:
            fast_initialsolutions_h5parm = None
        if (
            self.field.slow_gains_h5parm_filename is not None and
            os.path.exists(self.field.slow_gains_h5parm_filename)
        ):
            slow_initialsolutions_h5parm = CWLFile(self.field.slow_gains_h5parm_filename).to_json()
        else:
            slow_initialsolutions_h5parm = None

        self.input_parms = {'timechunk_filename': CWLDir(timechunk_filename).to_json(),
                            'data_colname': self.field.data_colname,
                            'starttime': starttime,
                            'ntimes': ntimes,
                            'solint_fast_timestep': solint_fast_timestep,
                            'solint_slow_timestep': solint_slow_timestep,
                            'solint_fast_freqstep': solint_fast_freqstep,
                            'solint_slow_freqstep': solint_slow_freqstep,
                            'fast_solutions_per_direction': fast_solutions_per_direction,
                            'slow_solutions_per_direction': slow_solutions_per_direction,
                            'calibrator_patch_names': calibrator_patch_names,
                            'calibrator_fluxes': calibrator_fluxes,
                            'output_fast_h5parm': output_fast_h5parm,
                            'combined_fast_h5parm': self.combined_fast_h5parm,
                            'output_slow_h5parm': output_slow_h5parm,
                            'combined_slow_h5parm': self.combined_slow_h5parm,
                            'calibration_skymodel_file': CWLFile(calibration_skymodel_file).to_json(),
                            'model_image_root': model_image_root,
                            'model_image_ra_dec': model_image_ra_dec,
                            'model_image_imsize': model_image_imsize,
                            'model_image_cellsize': model_image_cellsize,
                            'model_image_frequency_bandwidth': model_image_frequency_bandwidth,
                            'num_spectral_terms': num_spectral_terms,
                            'ra_mid': self.field.ra,
                            'dec_mid': self.field.dec,
                            'facet_region_width_ra': facet_region_width,
                            'facet_region_width_dec': facet_region_width,
                            'facet_region_file': facet_region_file,
                            'fast_smoothness_dd_factors': fast_smoothness_dd_factors,
                            'slow_smoothness_dd_factors': slow_smoothness_dd_factors,
                            'fast_smoothnessconstraint': fast_smoothnessconstraint,
                            'slow_smoothnessconstraint': slow_smoothnessconstraint,
                            'fast_smoothnessreffrequency': fast_smoothnessreffrequency,
                            'fast_smoothnessrefdistance': fast_smoothnessrefdistance,
                            'dp3_steps': f"[{','.join(dp3_steps)}]",
                            'ddecal_applycal_steps': ddecal_applycal_steps,
                            'applycal_steps': applycal_steps,
                            'bda_maxinterval': bda_maxinterval,
                            'bda_timebase': bda_timebase,
                            'bda_minchannels': bda_minchannels,
                            'bda_frequencybase': bda_frequencybase,
                            'normalize_h5parm': normalize_h5parm,
                            'fast_initialsolutions_h5parm': fast_initialsolutions_h5parm,
                            'slow_initialsolutions_h5parm': slow_initialsolutions_h5parm,
                            'max_normalization_delta': max_normalization_delta,
                            'scale_normalization_delta': scale_normalization_delta,
                            'phase_center_ra': self.field.ra,
                            'phase_center_dec': self.field.dec,
                            'llssolver': llssolver,
                            'maxiter': maxiter,
                            'propagatesolutions': propagatesolutions,
                            'solveralgorithm': solveralgorithm,
                            'onebeamperpatch': onebeamperpatch,
                            'stepsize': stepsize,
                            'stepsigma': stepsigma,
                            'tolerance': tolerance,
                            'uvlambdamin': uvlambdamin,
                            'parallelbaselines': parallelbaselines,
                            'sagecalpredict': sagecalpredict,
                            'fast_datause': fast_datause,
                            'slow_datause': slow_datause,
                            'sector_bounds_deg': sector_bounds_deg,
                            'sector_bounds_mid_deg': sector_bounds_mid_deg,
                            'combined_h5parms': self.combined_h5parms,
                            'fast_antennaconstraint': fast_antennaconstraint,
                            'slow_antennaconstraint': slow_antennaconstraint,
                            'idgcal_antennaconstraint': idgcal_antennaconstraint,
                            'output_idgcal_h5parm': output_idgcal_h5parm,
                            'solution_combine_mode': solution_combine_mode,
                            'solverlbfgs_dof': solverlbfgs_dof,
                            'solverlbfgs_iter': solverlbfgs_iter,
                            'solverlbfgs_minibatches': solverlbfgs_minibatches,
                            'max_threads': self.parset['cluster_specific']['max_threads']}

    def get_baselines_core(self):
        """
        Returns DPPP string of baseline selection for core calibration

        Returns
        -------
        baselines : str
            Baseline selection string
        """
        cs = self.get_core_stations()
        non_core = [a for a in self.field.stations if a not in cs]

        return '[CR]*&&;!{}'.format(';!'.join(non_core))

    def get_superterp_stations(self):
        """
        Returns list of superterp station names

        Returns
        -------
        stations : list
            Station names
        """
        if self.field.antenna == 'HBA':
            all_st = ['CS002HBA0', 'CS003HBA0', 'CS004HBA0', 'CS005HBA0', 'CS006HBA0', 'CS007HBA0',
                      'CS002HBA1', 'CS003HBA1', 'CS004HBA1', 'CS005HBA1', 'CS006HBA1', 'CS007HBA1']
        elif self.field.antenna == 'LBA':
            all_st = ['CS002LBA', 'CS003LBA', 'CS004LBA', 'CS005LBA', 'CS006LBA', 'CS007LBA']

        return [a for a in all_st if a in self.field.stations]

    def get_core_stations(self, include_nearest_remote=True):
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
        if self.field.antenna == 'HBA':
            all_core = ['CS001HBA0', 'CS002HBA0', 'CS003HBA0', 'CS004HBA0', 'CS005HBA0', 'CS006HBA0',
                        'CS007HBA0', 'CS011HBA0', 'CS013HBA0', 'CS017HBA0', 'CS021HBA0', 'CS024HBA0',
                        'CS026HBA0', 'CS028HBA0', 'CS030HBA0', 'CS031HBA0', 'CS032HBA0', 'CS101HBA0',
                        'CS103HBA0', 'CS201HBA0', 'CS301HBA0', 'CS302HBA0', 'CS401HBA0', 'CS501HBA0',
                        'CS001HBA1', 'CS002HBA1', 'CS003HBA1', 'CS004HBA1', 'CS005HBA1', 'CS006HBA1',
                        'CS007HBA1', 'CS011HBA1', 'CS013HBA1', 'CS017HBA1', 'CS021HBA1', 'CS024HBA1',
                        'CS026HBA1', 'CS028HBA1', 'CS030HBA1', 'CS031HBA1', 'CS032HBA1', 'CS101HBA1',
                        'CS103HBA1', 'CS201HBA1', 'CS301HBA1', 'CS302HBA1', 'CS401HBA1', 'CS501HBA1']
            if include_nearest_remote:
                all_core.extend(['RS106HBA0', 'RS205HBA0', 'RS305HBA0', 'RS306HBA0', 'RS503HBA0',
                                 'RS106HBA1', 'RS205HBA1', 'RS305HBA1', 'RS306HBA1', 'RS503HBA1'])
        elif self.field.antenna == 'LBA':
            all_core = ['CS001LBA', 'CS002LBA', 'CS003LBA', 'CS004LBA', 'CS005LBA', 'CS006LBA',
                        'CS007LBA', 'CS011LBA', 'CS013LBA', 'CS017LBA', 'CS021LBA', 'CS024LBA',
                        'CS026LBA', 'CS028LBA', 'CS030LBA', 'CS031LBA', 'CS032LBA', 'CS101LBA',
                        'CS103LBA', 'CS201LBA', 'CS301LBA', 'CS302LBA', 'CS401LBA', 'CS501LBA']
            if include_nearest_remote:
                all_core.extend(['RS106LBA', 'RS205LBA', 'RS305LBA', 'RS306LBA', 'RS503LBA'])

        return [a for a in all_core if a in self.field.stations]

    def get_model_image_parameters(self):
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
        if 'ReferenceFrequency' in skymodel.getColNames():
            # Each source can have its own reference frequency, so use the median over all
            # sources
            ref_freq = np.median(skymodel.getColValues('ReferenceFrequency'))  # Hz
        else:
            ref_freq = skymodel.table.meta['ReferenceFrequency']  # Hz
        frequency_bandwidth = [ref_freq, 1e6]  # Hz

        # Set the image coordinates, size, and cellsize
        if self.index == 1:
            # For initial cycle, assume center is the field center
            center_coords = [self.field.ra, self.field.dec]
            if hasattr(self.field, 'full_field_sector'):
                # Sky model generated in initial image step
                cellsize = self.field.full_field_sector.cellsize_deg  # deg/pixel
                size = self.field.full_field_sector.imsize  # [xsize, ysize] in pixels
            else:
                # Sky model generated externally. Use the cellsize defined for imaging and
                # analyze the sky model to find its extent
                cellsize = self.parset['imaging_specific']['cellsize_arcsec'] / 3600  # deg/pixel
                source_dict = {name: [ra, dec] for name, ra, dec in
                               zip(skymodel.getColValues('Name'),
                                   skymodel.getColValues('RA'),
                                   skymodel.getColValues('Dec'))}
                _, source_distances = self.field.get_source_distances(source_dict)  # deg
                radius = int(np.max(source_distances) / cellsize)  # pixels
                size = [radius * 2, radius * 2]  # pixels
        else:
            # Sky model generated in previous cycle's imaging step. Use the center and size
            # of the bounding box of all imaging sectors (note that this bounding box
            # includes a 20% padding, so it should include all model components, even
            # those on the very edge of a sector)
            cellsize = self.parset['imaging_specific']['cellsize_arcsec'] / 3600  # deg/pixel
            center_coords = [self.field.sector_bounds_mid_ra, self.field.sector_bounds_mid_dec]  # deg
            size = [int(self.field.sector_bounds_width_ra / cellsize), int(self.field.sector_bounds_width_dec / cellsize)]  # pixels

        # Convert RA and Dec to strings (required by WSClean)
        center_coords = lsmtool.utils.format_coordinates(*center_coords)

        return frequency_bandwidth, center_coords, size, cellsize

    def finalize(self):
        """
        Finalize this operation
        """
        # Copy the solutions (h5parm files) and report the flagged fraction
        dst_dir = os.path.join(self.parset['dir_working'], 'solutions', 'calibrate_{}'.format(self.index))
        os.makedirs(dst_dir, exist_ok=True)
        self.field.h5parm_filename = os.path.join(dst_dir, 'field-solutions.h5')
        self.field.fast_phases_h5parm_filename = os.path.join(dst_dir, 'field-solutions-fast-phase.h5')
        self.field.slow_gains_h5parm_filename = os.path.join(dst_dir, 'field-solutions-slow-gain.h5')
        if os.path.exists(self.field.h5parm_filename):
            os.remove(self.field.h5parm_filename)
        if self.field.generate_screens:
            # IDGCal (screens) only gives a combined h5parm, regardless of the type of solve
            shutil.copy(os.path.join(self.pipeline_working_dir, self.combined_h5parms),
                        os.path.join(dst_dir, self.field.h5parm_filename))
        elif self.field.do_slowgain_solve:
            shutil.copy(os.path.join(self.pipeline_working_dir, self.combined_h5parms),
                        os.path.join(dst_dir, self.field.h5parm_filename))
            shutil.copy(os.path.join(self.pipeline_working_dir, self.combined_slow_h5parm),
                        os.path.join(dst_dir, self.field.slow_gains_h5parm_filename))
            shutil.copy(os.path.join(self.pipeline_working_dir, self.combined_fast_h5parm),
                        os.path.join(dst_dir, self.field.fast_phases_h5parm_filename))
        else:
            # The h5parm with the full, combined solutions is also the fast-phases
            # h5parm
            shutil.copy(os.path.join(self.pipeline_working_dir, self.combined_fast_h5parm),
                        os.path.join(dst_dir, self.field.h5parm_filename))
            shutil.copy(os.path.join(self.pipeline_working_dir, self.combined_fast_h5parm),
                        os.path.join(dst_dir, self.field.fast_phases_h5parm_filename))
        self.field.scan_h5parms()  # verify h5parm and update flags for predict/image operations
        solsetname = 'coefficients000' if self.field.generate_screens else 'sol000'
        flagged_frac = misc.get_flagged_solution_fraction(self.field.h5parm_filename, solsetname=solsetname)
        self.log.info('Fraction of solutions that are flagged = {0:.2f}'.format(flagged_frac))
        self.field.calibration_diagnostics.append({'cycle_number': self.index,
                                                   'solution_flagged_fraction': flagged_frac})

        # Copy the plots (PNG files)
        dst_dir = os.path.join(self.parset['dir_working'], 'plots', 'calibrate_{}'.format(self.index))
        os.makedirs(dst_dir, exist_ok=True)
        plot_filenames = glob.glob(os.path.join(self.pipeline_working_dir, '*.png'))
        for plot_filename in plot_filenames:
            dst_filename = os.path.join(dst_dir, os.path.basename(plot_filename))
            if os.path.exists(dst_filename):
                os.remove(dst_filename)
            shutil.copy(plot_filename, dst_filename)

        # Finally call finalize() in the parent class
        super().finalize()


class CalibrateDI(Operation):
    """
    Operation to perform direction-independent (DI) calibration of the field
    """
    def __init__(self, field, index):
        super().__init__(field, index=index, name='calibrate_di')

    def set_parset_parameters(self):
        """
        Define parameters needed for the CWL workflow template
        """
        if self.batch_system == 'slurm':
            # For some reason, setting coresMax ResourceRequirement hints does
            # not work with SLURM
            max_cores = None
        else:
            max_cores = self.parset['cluster_specific']['max_cores']
        self.parset_parms = {'rapthor_pipeline_dir': self.rapthor_pipeline_dir,
                             'max_cores': max_cores}

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
        starttime_fulljones = self.field.get_obs_parameters('starttime')
        ntimes_fulljones = self.field.get_obs_parameters('ntimes')

        # Get the filenames of the input files for each time chunk
        timechunk_filename_fulljones = self.field.get_obs_parameters('timechunk_filename')

        # Get the solution intervals for the calibrations
        solint_fulljones_timestep = self.field.get_obs_parameters('solint_slow_timestep_fulljones')
        solint_fulljones_freqstep = self.field.get_obs_parameters('solint_slow_freqstep_fulljones')

        # Define various output filenames for the solution tables. We save some
        # as attributes since they are needed in finalize()
        output_h5parm_fulljones = ['fulljones_gain_{}.h5parm'.format(i)
                                   for i in range(self.field.ntimechunks)]
        self.combined_h5parm_fulljones = 'fulljones_gains.h5'

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

        self.input_parms = {'timechunk_filename_fulljones': CWLDir(timechunk_filename_fulljones).to_json(),
                            'data_colname': 'DATA',
                            'starttime_fulljones': starttime_fulljones,
                            'ntimes_fulljones': ntimes_fulljones,
                            'solint_fulljones_timestep': solint_fulljones_timestep,
                            'solint_fulljones_freqstep': solint_fulljones_freqstep,
                            'output_h5parm_fulljones': output_h5parm_fulljones,
                            'combined_h5parm_fulljones': self.combined_h5parm_fulljones,
                            'smoothnessconstraint_fulljones': smoothnessconstraint_fulljones,
                            'max_normalization_delta': max_normalization_delta,
                            'llssolver': llssolver,
                            'maxiter': maxiter,
                            'propagatesolutions': propagatesolutions,
                            'solveralgorithm': solveralgorithm,
                            'stepsize': stepsize,
                            'stepsigma': stepsigma,
                            'tolerance': tolerance,
                            'uvlambdamin': uvlambdamin,
                            'solverlbfgs_dof': solverlbfgs_dof,
                            'solverlbfgs_iter': solverlbfgs_iter,
                            'solverlbfgs_minibatches': solverlbfgs_minibatches,
                            'max_threads': self.parset['cluster_specific']['max_threads']}

    def finalize(self):
        """
        Finalize this operation
        """
        # Copy the solutions (h5parm file) and report the flagged fraction
        dst_dir = os.path.join(self.parset['dir_working'], 'solutions', 'calibrate_di_{}'.format(self.index))
        os.makedirs(dst_dir, exist_ok=True)
        self.field.fulljones_h5parm_filename = os.path.join(dst_dir, 'fulljones-solutions.h5')
        if os.path.exists(self.field.fulljones_h5parm_filename):
            os.remove(self.field.fulljones_h5parm_filename)
        shutil.copy(os.path.join(self.pipeline_working_dir, self.combined_h5parm_fulljones),
                    os.path.join(dst_dir, self.field.fulljones_h5parm_filename))
        self.field.scan_h5parms()  # verify h5parm and update flags for predict/image operations
        flagged_frac = misc.get_flagged_solution_fraction(self.field.fulljones_h5parm_filename)
        self.log.info('Fraction of solutions that are flagged = {0:.2f}'.format(flagged_frac))

        # Copy the plots (PNG files)
        dst_dir = os.path.join(self.parset['dir_working'], 'plots', 'calibrate_di_{}'.format(self.index))
        os.makedirs(dst_dir, exist_ok=True)
        plot_filenames = glob.glob(os.path.join(self.pipeline_working_dir, '*.png'))
        for plot_filename in plot_filenames:
            dst_filename = os.path.join(dst_dir, os.path.basename(plot_filename))
            if os.path.exists(dst_filename):
                os.remove(dst_filename)
            shutil.copy(plot_filename, dst_filename)

        # Finally call finalize() in the parent class
        super().finalize()
