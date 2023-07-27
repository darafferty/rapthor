"""
Module that holds the Calibrate class
"""
import os
import logging
import shutil
import glob
from rapthor.lib.operation import Operation
from rapthor.lib import miscellaneous as misc
from rapthor.lib.cwl import CWLFile, CWLDir

log = logging.getLogger('rapthor:calibrate')


class Calibrate(Operation):
    """
    Operation to calibrate the field
    """
    def __init__(self, field, index):
        super(Calibrate, self).__init__(field, name='calibrate', index=index)

    def set_parset_parameters(self):
        """
        Define parameters needed for the CWL workflow template
        """
        if self.batch_system == 'slurm':
            # For some reason, setting coresMax ResourceRequirement hints does
            # not work with SLURM
            max_cores = None
        else:
            max_cores = self.field.parset['cluster_specific']['max_cores']
        if self.field.parset['calibration_specific']['slow_timestep_joint_sec'] > 0:
            do_joint_solve = True
        else:
            do_joint_solve = False
        if self.field.dde_method == 'facets':
            use_facets = True
        else:
            use_facets = False
        self.parset_parms = {'rapthor_pipeline_dir': self.rapthor_pipeline_dir,
                             'use_screens': self.field.use_screens,
                             'use_facets': use_facets,
                             'do_slowgain_solve': self.field.do_slowgain_solve,
                             'do_joint_solve': do_joint_solve,
                             'use_scalarphase': self.field.use_scalarphase,
                             'apply_diagonal_solutions': self.field.apply_diagonal_solutions,
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
        slow_starttime_joint = self.field.get_obs_parameters('slow_starttime_joint')
        slow_ntimes_joint = self.field.get_obs_parameters('slow_ntimes_joint')
        slow_starttime_separate = self.field.get_obs_parameters('slow_starttime_separate')
        slow_ntimes_separate = self.field.get_obs_parameters('slow_ntimes_separate')

        # Get the filenames of the input files for each frequency chunk
        freqchunk_filename_joint = self.field.get_obs_parameters('freqchunk_filename_joint')
        freqchunk_filename_separate = self.field.get_obs_parameters('freqchunk_filename_separate')

        # Get the start channel and number of channels for the frequency chunks
        startchan_joint = self.field.get_obs_parameters('startchan_joint')
        nchan_joint = self.field.get_obs_parameters('nchan_joint')
        startchan_separate = self.field.get_obs_parameters('startchan_separate')
        nchan_separate = self.field.get_obs_parameters('nchan_separate')

        # Get the solution intervals for the calibrations
        solint_fast_timestep = self.field.get_obs_parameters('solint_fast_timestep')
        solint_slow_timestep_joint = self.field.get_obs_parameters('solint_slow_timestep_joint')
        solint_slow_timestep_separate = self.field.get_obs_parameters('solint_slow_timestep_separate')
        solint_fast_freqstep = self.field.get_obs_parameters('solint_fast_freqstep')
        solint_slow_freqstep_joint = self.field.get_obs_parameters('solint_slow_freqstep_joint')
        solint_slow_freqstep_separate = self.field.get_obs_parameters('solint_slow_freqstep_separate')

        # Define various output filenames for the solution tables. We save some
        # as attributes since they are needed in finalize()
        output_fast_h5parm = ['fast_phase_{}.h5parm'.format(i)
                              for i in range(self.field.ntimechunks)]
        self.combined_fast_h5parm = 'fast_phases.h5parm'
        output_slow_h5parm_joint = ['slow_gain_joint_{}.h5parm'.format(i)
                                    for i in range(self.field.nfreqchunks_joint)]
        self.combined_h5parms = 'combined_solutions.h5'
        output_slow_h5parm_separate = ['slow_gain_separate_{}.h5parm'.format(i)
                                       for i in range(self.field.nfreqchunks_separate)]
        combined_slow_h5parm_joint = 'slow_gains_joint.h5parm'
        combined_slow_h5parm_separate = 'slow_gains_separate.h5parm'
        combined_h5parms_fast_slow_joint = 'combined_solutions_fast_slow_joint.h5'
        combined_h5parms_slow_joint_separate = 'combined_solutions_slow_joint_separate.h5'

        # Define the input sky model
        calibration_skymodel_file = self.field.calibration_skymodel_file

        # Get the calibrator names and fluxes (used in screen fitting)
        calibrator_patch_names = self.field.calibrator_patch_names
        calibrator_fluxes = self.field.calibrator_fluxes

        # Set the constraints used in the calibrations
        fast_smoothnessconstraint = self.field.fast_smoothnessconstraint
        fast_smoothnessreffrequency = self.field.get_obs_parameters('fast_smoothnessreffrequency')
        fast_smoothnessrefdistance = self.field.fast_smoothnessrefdistance
        slow_smoothnessconstraint_joint = self.field.slow_smoothnessconstraint_joint
        slow_smoothnessconstraint_separate = self.field.slow_smoothnessconstraint_separate
        if self.field.do_slowgain_solve or self.field.antenna == 'LBA':
            # Use the core stationconstraint if the slow solves will be done or if
            # we have LBA data (which has lower sensitivity than HBA data)
            fast_antennaconstraint = '[[{}]]'.format(','.join(self.get_core_stations()))
        else:
            # For HBA data, if the slow solves will not be done, we remove the
            # stationconstraint to allow each station to get its own fast phase
            # corrections
            fast_antennaconstraint = '[]'
        slow_antennaconstraint = '[[{}]]'.format(','.join(self.field.stations))
        max_normalization_delta = self.field.max_normalization_delta
        scale_normalization_delta = '{}'.format(self.field.scale_normalization_delta)

        # Get various DDECal solver parameters
        llssolver = self.field.llssolver
        maxiter = self.field.maxiter
        propagatesolutions = self.field.propagatesolutions
        solveralgorithm = self.field.solveralgorithm
        onebeamperpatch = self.field.onebeamperpatch
        stepsize = self.field.stepsize
        tolerance = self.field.tolerance
        uvlambdamin = self.field.solve_min_uv_lambda
        parallelbaselines = self.field.parallelbaselines
        sagecalpredict = self.field.sagecalpredict
        solverlbfgs_dof = self.field.solverlbfgs_dof
        solverlbfgs_iter = self.field.solverlbfgs_iter
        solverlbfgs_minibatches = self.field.solverlbfgs_minibatches

        # Get the size of the imaging area (for use in making the a-term images)
        sector_bounds_deg = '{}'.format(self.field.sector_bounds_deg)
        sector_bounds_mid_deg = '{}'.format(self.field.sector_bounds_mid_deg)

        # Set the number of chunks to split the solution tables into and define
        # the associated filenames
        if self.field.do_slowgain_solve:
            nsplit = sum(self.field.get_obs_parameters('nsplit_slow'))
        else:
            nsplit = sum(self.field.get_obs_parameters('nsplit_fast'))
        split_outh5parm = ['split_solutions_{}.h5'.format(i) for i in
                           range(nsplit)]

        # Set the root filenames for the a-term images. We save it to an attribute
        # since it is needed in finalize()
        self.output_aterms_root = ['diagonal_aterms_{}'.format(i) for i in
                                   range(len(split_outh5parm))]

        # Set the type of screen to make
        screen_type = self.field.screen_type

        self.input_parms = {'timechunk_filename': CWLDir(timechunk_filename).to_json(),
                            'freqchunk_filename_joint': CWLDir(freqchunk_filename_joint).to_json(),
                            'freqchunk_filename_separate': CWLDir(freqchunk_filename_separate).to_json(),
                            'starttime': starttime,
                            'ntimes': ntimes,
                            'slow_starttime_joint': slow_starttime_joint,
                            'slow_starttime_separate': slow_starttime_separate,
                            'slow_ntimes_joint': slow_ntimes_joint,
                            'slow_ntimes_separate': slow_ntimes_separate,
                            'startchan_joint': startchan_joint,
                            'startchan_separate': startchan_separate,
                            'nchan_joint': nchan_joint,
                            'nchan_separate': nchan_separate,
                            'solint_fast_timestep': solint_fast_timestep,
                            'solint_slow_timestep_joint': solint_slow_timestep_joint,
                            'solint_slow_timestep_separate': solint_slow_timestep_separate,
                            'solint_fast_freqstep': solint_fast_freqstep,
                            'solint_slow_freqstep_joint': solint_slow_freqstep_joint,
                            'solint_slow_freqstep_separate': solint_slow_freqstep_separate,
                            'calibrator_patch_names': calibrator_patch_names,
                            'calibrator_fluxes': calibrator_fluxes,
                            'output_fast_h5parm': output_fast_h5parm,
                            'combined_fast_h5parm': self.combined_fast_h5parm,
                            'output_slow_h5parm_joint': output_slow_h5parm_joint,
                            'output_slow_h5parm_separate': output_slow_h5parm_separate,
                            'calibration_skymodel_file': CWLFile(calibration_skymodel_file).to_json(),
                            'fast_smoothnessconstraint': fast_smoothnessconstraint,
                            'fast_smoothnessreffrequency': fast_smoothnessreffrequency,
                            'fast_smoothnessrefdistance': fast_smoothnessrefdistance,
                            'slow_smoothnessconstraint_joint': slow_smoothnessconstraint_joint,
                            'slow_smoothnessconstraint_separate': slow_smoothnessconstraint_separate,
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
                            'tolerance': tolerance,
                            'uvlambdamin': uvlambdamin,
                            'parallelbaselines': parallelbaselines,
                            'sagecalpredict': sagecalpredict,
                            'sector_bounds_deg': sector_bounds_deg,
                            'sector_bounds_mid_deg': sector_bounds_mid_deg,
                            'split_outh5parm': split_outh5parm,
                            'output_aterms_root': self.output_aterms_root,
                            'screen_type': screen_type,
                            'combined_h5parms': self.combined_h5parms,
                            'fast_antennaconstraint': fast_antennaconstraint,
                            'slow_antennaconstraint': slow_antennaconstraint,
                            'combined_slow_h5parm_joint': combined_slow_h5parm_joint,
                            'combined_slow_h5parm_separate': combined_slow_h5parm_separate,
                            'combined_h5parms_fast_slow_joint': combined_h5parms_fast_slow_joint,
                            'combined_h5parms_slow_joint_separate': combined_h5parms_slow_joint_separate,
                            'solverlbfgs_dof': solverlbfgs_dof,
                            'solverlbfgs_iter': solverlbfgs_iter,
                            'solverlbfgs_minibatches': solverlbfgs_minibatches,
                            'max_threads': self.field.parset['cluster_specific']['max_threads']}

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

    def finalize(self):
        """
        Finalize this operation
        """
        # Get the filenames of the aterm images (for use in the image operation). The files
        # were written by the 'make_aterms' step and the number of them can vary, depending
        # on the node memory, etc.
        if self.field.use_screens:
            self.field.aterm_image_filenames = []
            for aterms_root in self.output_aterms_root:
                with open(os.path.join(self.pipeline_working_dir, aterms_root+'.txt'), 'r') as f:
                    self.field.aterm_image_filenames.extend(f.readlines())
            self.field.aterm_image_filenames = [os.path.join(self.pipeline_working_dir, af.strip())
                                                for af in self.field.aterm_image_filenames]

        # Copy the solutions (h5parm files) and report the flagged fraction
        dst_dir = os.path.join(self.parset['dir_working'], 'solutions', 'calibrate_{}'.format(self.index))
        misc.create_directory(dst_dir)
        self.field.h5parm_filename = os.path.join(dst_dir, 'field-solutions.h5')
        if os.path.exists(self.field.h5parm_filename):
            os.remove(self.field.h5parm_filename)
        if self.field.do_slowgain_solve:
            shutil.copy(os.path.join(self.pipeline_working_dir, self.combined_h5parms),
                        os.path.join(dst_dir, self.field.h5parm_filename))
        else:
            shutil.copy(os.path.join(self.pipeline_working_dir, self.combined_fast_h5parm),
                        os.path.join(dst_dir, self.field.h5parm_filename))
        flagged_frac = misc.get_flagged_solution_fraction(self.field.h5parm_filename)
        self.log.info('Fraction of solutions that are flagged = {0:.2f}'.format(flagged_frac))

        # Copy the plots (PNG files)
        dst_dir = os.path.join(self.parset['dir_working'], 'plots', 'calibrate_{}'.format(self.index))
        misc.create_directory(dst_dir)
        plot_filenames = glob.glob(os.path.join(self.pipeline_working_dir, '*.png'))
        for plot_filename in plot_filenames:
            dst_filename = os.path.join(dst_dir, os.path.basename(plot_filename))
            if os.path.exists(dst_filename):
                os.remove(dst_filename)
            shutil.copy(plot_filename, dst_filename)

        # Finally call finalize() in the parent class
        super().finalize()
