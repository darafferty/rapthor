"""
Definition of the Field class
"""
import os
import sys
import logging
import numpy as np
import lsmtool
import lsmtool.skymodel
from rapthor.lib import miscellaneous as misc
from rapthor.lib.observation import Observation
from rapthor.lib.sector import Sector
from rapthor.lib.image import FITSImage
from shapely.geometry import Point, Polygon, MultiPolygon
from astropy.table import vstack
import rtree.index
import glob


class Field(object):
    """
    The Field object stores parameters needed for processing of the field

    Parameters
    ----------
    parset : dict
        Parset with processing parameters
    minimal : bool
        If True, only initialize the minimal set of required parameters
    """
    def __init__(self, parset, mininmal=False):
        # Initialize basic attributes. These can be overridden later by the strategy
        # values and/or the operations
        self.name = 'field'
        self.log = logging.getLogger('rapthor:{}'.format(self.name))
        self.parset = parset.copy()
        self.working_dir = self.parset['dir_working']
        self.ms_filenames = self.parset['mss']
        self.numMS = len(self.ms_filenames)
        self.data_colname = 'DATA'
        self.flag_abstime = self.parset['flag_abstime']
        self.flag_baseline = self.parset['flag_baseline']
        self.flag_freqrange = self.parset['flag_freqrange']
        self.flag_expr = self.parset['flag_expr']
        self.input_h5parm = self.parset['input_h5parm']
        self.target_flux = self.parset['calibration_specific']['patch_target_flux_jy']
        self.target_number = self.parset['calibration_specific']['patch_target_number']
        self.solve_min_uv_lambda = self.parset['calibration_specific']['solve_min_uv_lambda']
        self.fast_smoothnessconstraint = self.parset['calibration_specific']['fast_smoothnessconstraint']
        self.slow_smoothnessconstraint = self.parset['calibration_specific']['slow_smoothnessconstraint']
        self.propagatesolutions = self.parset['calibration_specific']['propagatesolutions']
        self.solveralgorithm = self.parset['calibration_specific']['solveralgorithm']
        self.onebeamperpatch = self.parset['calibration_specific']['onebeamperpatch']
        self.llssolver = self.parset['calibration_specific']['llssolver']
        self.maxiter = self.parset['calibration_specific']['maxiter']
        self.stepsize = self.parset['calibration_specific']['stepsize']
        self.tolerance = self.parset['calibration_specific']['tolerance']
        self.llsstarttolerance = self.parset['calibration_specific']['llsstarttolerance']
        self.llstolerance = self.parset['calibration_specific']['llstolerance']
        self.use_screens = self.parset['imaging_specific']['use_screens']
        self.screen_type = self.parset['imaging_specific']['screen_type']
        self.use_mpi = self.parset['imaging_specific']['use_mpi']
        self.use_idg_predict = self.parset['calibration_specific']['use_idg_predict']
        self.reweight = self.parset['imaging_specific']['reweight']
        self.debug = self.parset['calibration_specific']['debug']
        self.do_multiscale_clean = self.parset['imaging_specific']['do_multiscale_clean']
        self.convergence_ratio = 0.95
        self.divergence_ratio = 1.1
        self.peel_outliers = False
        self.imaged_sources_only = False
        self.peel_bright_sources = False
        self.use_scalarphase = True
        self.field_image_filename_prev = None
        self.field_image_filename = None

        if not mininmal:
            # Scan MS files to get observation info
            self.scan_observations()

            # Set up imaging sectors
            self.makeWCS()
            self.define_imaging_sectors()

            # Chunk the observations into smaller parts if needed
            self.chunk_observations(self.parset['data_fraction'])

    def scan_observations(self):
        """
        Checks input MS files and initializes the associated Observation objects
        """
        if len(self.ms_filenames) > 1:
            infix = 's'
        else:
            infix = ''
        self.log.debug('Scanning input MS file{}...'.format(infix))
        self.observations = []
        for ms_filename in self.ms_filenames:
            self.observations.append(Observation(ms_filename))

        # Define a reference observation for the comparisons below
        obs0 = self.observations[0]

        # Check that all observations have the same antenna type
        self.antenna = obs0.antenna
        for obs in self.observations:
            if self.antenna != obs.antenna:
                self.log.critical('Antenna type for MS {0} differs from the one for MS {1}! '
                                  'Exiting!'.format(self.obs.ms_filename, self.obs0.ms_filename))
                sys.exit(1)

        # Check that all observations have the same frequency axis
        # NOTE: this may not be necessary and is disabled for now
        enforce_uniform_frequency_structure = False
        if enforce_uniform_frequency_structure:
            for obs in self.observations:
                if (obs0.numchannels != obs.numchannels or
                        obs0.startfreq != obs.startfreq or
                        obs0.endfreq != obs.endfreq or
                        obs0.channelwidth != obs.channelwidth):
                    self.log.critical('Frequency axis for MS {0} differs from the one for MS {1}! '
                                      'Exiting!'.format(self.obs.ms_filename, self.obs0.ms_filename))
                    sys.exit(1)

        # Check that all observations have the same pointing
        self.ra = obs0.ra
        self.dec = obs0.dec
        for obs in self.observations:
            if self.ra != obs.ra or self.dec != obs.dec:
                self.log.critical('Pointing for MS {0} differs from the one for MS {1}! '
                                  'Exiting!'.format(self.obs.ms_filename, self.obs0.ms_filename))
                sys.exit(1)

        # Check that all observations have the same station diameter
        self.diam = obs0.diam
        for obs in self.observations:
            if self.diam != obs.diam:
                self.log.critical('Station diameter for MS {0} differs from the one for MS {1}! '
                                  'Exiting!'.format(self.obs.ms_filename, self.obs0.ms_filename))
                sys.exit(1)

        # Check that all observations have the same stations
        self.stations = obs0.stations
        for obs in self.observations:
            if self.stations != obs.stations:
                self.log.critical('Stations in MS {0} differ from those in MS {1}! '
                                  'Exiting!'.format(self.obs.ms_filename, self.obs0.ms_filename))
                sys.exit(1)

        # Find mean elevation and FOV over all observations
        el_rad_list = []
        ref_freq_list = []
        for obs in self.observations:
            el_rad_list.append(obs.mean_el_rad)
            ref_freq_list.append(obs.referencefreq)
        sec_el = 1.0 / np.sin(np.mean(el_rad_list))
        self.mean_el_rad = np.mean(el_rad_list)
        self.fwhm_deg = 1.1 * ((3.0e8 / np.mean(ref_freq_list)) /
                               self.diam) * 180. / np.pi * sec_el
        self.fwhm_ra_deg = self.fwhm_deg / sec_el
        self.fwhm_dec_deg = self.fwhm_deg

        # Warning if parset pointing is different from observation pointing
        parra = self.parset['imaging_specific']['grid_center_ra']
        pardec = self.parset['imaging_specific']['grid_center_dec']
        if (parra is not None and np.abs(parra - self.ra) > self.fwhm_ra_deg / 2.0):
            self.log.warning('Grid_center_ra requested in parset is different from the value in the observation.')
        if (pardec is not None and np.abs(pardec - self.dec) > self.fwhm_dec_deg / 2.0):
            self.log.warning('Grid_center_dec requested in parset is different from the value in the observation.')

        # Set the MS file to use for beam model in sky model correction.
        # This should be the observation that best matches the weighted average
        # beam, so we use that closest to the mid point
        times = [(obs.endtime+obs.starttime)/2.0 for obs in self.observations]
        weights = [(obs.endtime-obs.starttime) for obs in self.observations]
        mid_time = np.average(times, weights=weights)
        mid_index = np.argmin(np.abs(np.array(times)-mid_time))
        self.beam_ms_filename = self.observations[mid_index].ms_filename

    def chunk_observations(self, data_fraction=1.0):
        """
        Break observations into smaller time chunks if desired

        Chunking is done if:

        - the specified data_fraction < 1 (so part of an observation is to be processed)
        - nobs * nsectors < nnodes (so all nodes can be used efficiently. In particular,
          the predict pipeline parallelizes over sectors and observations, so we need
          enough observations to allow all nodes to be occupied.)

        Parameters
        ----------
        data_fraction : float, optional
            Fraction of data to use during processing
        """
        if data_fraction < 1.0:
            if not hasattr(self, 'full_observations'):
                self.full_observations = self.observations[:]
            self.observations = []
            for obs in self.full_observations:
                mintime = self.parset['calibration_specific']['slow_timestep_sec']
                tottime = obs.endtime - obs.starttime
                if data_fraction < min(1.0, mintime/tottime):
                    obs.log.warning('The specified value of data_fraction ({0:0.3f}) results in a '
                                    'total time for this observation that is less than the '
                                    'slow-gain timestep. The data fraction will be increased '
                                    'to {1:0.3f} to ensure the slow-gain timestep requirement is '
                                    'met.'.format(data_fraction, min(1.0, mintime/tottime)))
                nchunks = int(np.ceil(data_fraction / (mintime / tottime)))
                if nchunks == 1:
                    # Center the chunk around the midpoint (which is generally the most
                    # sensitive, near transit)
                    midpoint = obs.starttime + tottime / 2
                    chunktime = min(tottime, max(mintime, data_fraction*tottime))
                    if chunktime < tottime:
                        self.observations.append(Observation(obs.ms_filename,
                                                             starttime=midpoint-chunktime/2,
                                                             endtime=midpoint+chunktime/2))
                    else:
                        self.observations.append(obs)
                else:
                    steptime = mintime * (tottime / mintime - nchunks) / nchunks + mintime
                    starttimes = np.arange(obs.starttime, obs.endtime, steptime)
                    endtimes = np.arange(obs.starttime+mintime, obs.endtime+mintime, steptime)
                    for starttime, endtime in zip(starttimes, endtimes):
                        if endtime > obs.endtime:
                            starttime = obs.endtime - mintime
                            endtime = obs.endtime
                        self.observations.append(Observation(obs.ms_filename, starttime=starttime,
                                                             endtime=endtime))
        minnobs = self.parset['cluster_specific']['max_nodes']
        if len(self.imaging_sectors)*len(self.observations) < minnobs:
            # To ensure all the nodes are used efficiently, try to divide up the observations
            # to reach at least minnobs in total. For simplicity, we try to divide each existing
            # observation into minnobs number of new observations (since there is no
            # drawback to having more than minnobs observations in total)
            #
            # Note: Due to a limitation in Dysco, we make sure to have at least
            # 2 time slots per observation, otherwise the output MS cannot be
            # written with compression
            mintime = self.parset['calibration_specific']['slow_timestep_sec']
            prev_observations = self.observations[:]
            self.observations = []
            for obs in prev_observations:
                tottime = obs.endtime - obs.starttime
                nchunks = min(minnobs, max(1, int(tottime / mintime)))
                if nchunks > 1:
                    steptime = tottime / nchunks
                    steptime -= steptime % mintime
                    numsamples = int(np.ceil(steptime / obs.timepersample))
                    if numsamples < 2:
                        steptime += mintime
                    starttimes = np.arange(obs.starttime, obs.endtime, steptime)
                    endtimes = [st+steptime for st in starttimes]
                    for nc, (starttime, endtime) in enumerate(zip(starttimes, endtimes)):
                        if nc == len(endtimes)-1:
                            endtime = obs.endtime
                        self.observations.append(Observation(obs.ms_filename, starttime=starttime,
                                                             endtime=endtime))
                else:
                    self.observations.append(obs)

        # Update the copies stored in the imaging sectors
        for sector in self.imaging_sectors:
            sector.observations = []
            for obs in self.observations:
                cobs = obs.copy()
                sector.observations.append(cobs)

    def set_obs_parameters(self):
        """
        Sets parameters for all observations from current parset and sky model
        """
        ntimechunks = 0
        nfreqchunks = 0
        for obs in self.observations:
            obs.set_calibration_parameters(self.parset, self.num_patches, len(self.observations))
            ntimechunks += obs.ntimechunks
            nfreqchunks += obs.nfreqchunks
        self.ntimechunks = ntimechunks
        self.nfreqchunks = nfreqchunks

    def get_obs_parameters(self, parameter):
        """
        Returns list of parameters for all observations

        Parameters
        ----------
        parameter : str
            Name of parameter to return

        Returns
        -------
        parameters : list
            List of parameters of each observation
        """
        return sum([obs.parameters[parameter] for obs in self.observations], [])

    def make_skymodels(self, skymodel_true_sky, skymodel_apparent_sky=None, regroup=True,
                       find_sources=False, target_flux=None, target_number=None,
                       index=0):
        """
        Groups a sky model into source and calibration patches

        Grouping is done on the apparent-flux sky model if available. Note that the
        source names in the true- and apparent-flux models must be the same (i.e., the
        only differences between the two are the fluxes and spectral indices)

        Parameters
        ----------
        skymodel_true_sky : str or LSMTool skymodel object
            Filename of input makesourcedb true-flux sky model file
        skymodel_apparent_sky : str or LSMTool skymodel object, optional
            Filename of input makesourcedb apparent-flux sky model file
        regroup : bool, optional
            If False, the calibration sky model is not regrouped to the target flux.
            Instead, the existing calibration groups are used
        find_sources : bool, optional
            If True, group the sky model by thresholding to find sources. This is not
            needed if the input sky model was filtered by PyBDSF in the imaging
            pipeline
        target_flux : float, optional
            Target flux in Jy for grouping
        target_number : int, optional
            Target number of patches for grouping
        index : index
            Iteration index
        """
        # Make output directories for sky models and define filenames
        dst_dir = os.path.join(self.working_dir, 'skymodels', 'calibrate_{}'.format(index))
        misc.create_directory(dst_dir)
        self.calibration_skymodel_file = os.path.join(dst_dir, 'calibration_skymodel.txt')
        self.calibrators_only_skymodel_file = os.path.join(dst_dir, 'calibrators_only_skymodel.txt')
        self.source_skymodel_file = os.path.join(dst_dir, 'source_skymodel.txt')
        dst_dir = os.path.join(self.working_dir, 'skymodels', 'image_{}'.format(index))
        misc.create_directory(dst_dir)
        self.bright_source_skymodel_file = os.path.join(dst_dir, 'bright_source_skymodel.txt')

        # First check whether sky models already exist due to a previous run and attempt
        # to load them
        try:
            self.calibration_skymodel = lsmtool.load(str(self.calibration_skymodel_file))
            self.num_patches = len(self.calibration_skymodel.getPatchNames())
            calibrators_skymodel = lsmtool.load(str(self.calibrators_only_skymodel_file))
            self.calibrator_patch_names = calibrators_skymodel.getPatchNames().tolist()
            self.calibrator_fluxes = calibrators_skymodel.getColValues('I', aggregate='sum').tolist()
            self.source_skymodel = lsmtool.load(str(self.source_skymodel_file))

            if self.peel_bright_sources:
                # The bright-source model file may not exist if there are no bright sources,
                # but also if a reset of the imaging pipeline was done. Unfortunately, there
                # is no way to determine which of these two possibilities is the case. So, if
                # it does not exist, we have to regenerate the sky models
                if os.path.exists(self.bright_source_skymodel_file):
                    self.bright_source_skymodel = lsmtool.load(str(self.bright_source_skymodel_file))
                    all_skymodels_loaded = True
                else:
                    all_skymodels_loaded = False
            else:
                self.bright_source_skymodel = calibrators_skymodel
                all_skymodels_loaded = True
        except IOError:
            all_skymodels_loaded = False

        if all_skymodels_loaded:
            # If all the required sky models were loaded from files, return; otherwise,
            # continue on to regenerate the sky models from scratch
            self.log.info('Using {} calibration patches'.format(self.num_patches))
            return

        # If sky models do not already exist, make them
        self.log.info('Analyzing sky model...')
        if type(skymodel_true_sky) is not lsmtool.skymodel.SkyModel:
            skymodel_true_sky = lsmtool.load(str(skymodel_true_sky),
                                             beamMS=self.beam_ms_filename)
        if skymodel_apparent_sky is not None:
            if type(skymodel_apparent_sky) is not lsmtool.skymodel.SkyModel:
                skymodel_apparent_sky = lsmtool.load(str(skymodel_apparent_sky),
                                                     beamMS=self.beam_ms_filename)

        # Check if any sky models included in rapthor are within the region of the
        # input sky model. If so, concatenate them with the input sky model
        if self.parset['calibration_specific']['use_included_skymodels']:
            max_separation_deg = self.fwhm_deg * 2.0
            rapthor_lib_dir = os.path.dirname(os.path.abspath(__file__))
            skymodel_dir = os.path.join(os.path.split(rapthor_lib_dir)[0], 'skymodels')
            skymodels = glob.glob(os.path.join(skymodel_dir, '*.skymodel'))
            concat_skymodels = []
            for skymodel in skymodels:
                try:
                    s = lsmtool.load(str(skymodel))
                    dist_deg = s.getDistance(self.ra, self.dec)
                    if any(dist_deg < max_separation_deg):
                        concat_skymodels.append(skymodel)
                except IOError:
                    pass
            matching_radius_deg = 30.0 / 3600.0  # => 30 arcsec
            for s in concat_skymodels:
                skymodel_true_sky.concatenate(s, matchBy='position', radius=matching_radius_deg,
                                              keep='from2', inheritPatches=True)
                if skymodel_apparent_sky is not None:
                    skymodel_apparent_sky.concatenate(s, matchBy='position', radius=matching_radius_deg,
                                                      keep='from2', inheritPatches=True)

        # If an apparent sky model is given, use it for defining the calibration patches.
        # Otherwise, attenuate the true sky model while grouping into patches
        if skymodel_apparent_sky is not None:
            source_skymodel = skymodel_apparent_sky
            applyBeam_group = False
        else:
            source_skymodel = skymodel_true_sky.copy()
            applyBeam_group = True

        # Make a source sky model, used for source avoidance
        if find_sources:
            self.log.info('Identifying sources...')
            source_skymodel.group('threshold', FWHM='40.0 arcsec', threshold=0.05)
        source_skymodel.write(self.source_skymodel_file, clobber=True)
        self.source_skymodel = source_skymodel.copy()  # save and make copy before grouping

        # Find groups of bright sources to use as basis for calibrator patches
        # and for subtraction if desired. The patch positions are set to the
        # flux-weighted mean position, using the apparent sky model. This position is
        # then propagated and used for the calibrate and predict sky models
        source_skymodel.group('meanshift', byPatch=True, applyBeam=applyBeam_group,
                              lookDistance=0.075, groupingDistance=0.01)
        source_skymodel.setPatchPositions(method='wmean')
        patch_dict = source_skymodel.getPatchPositions()

        # debug
        dst_dir = os.path.join(self.working_dir, 'skymodels', 'calibrate_{}'.format(index))
        misc.create_directory(dst_dir)
        skymodel_true_sky_file = os.path.join(dst_dir, 'skymodel_meanshift.txt')
        source_skymodel.write(skymodel_true_sky_file, clobber=True)
        # debug

        # Determine the flux cut to use to select the bright sources (calibrators)
        if target_flux is None:
            target_flux = self.target_flux
        if target_number is None:
            target_number = self.target_number
        if target_number is not None:
            # Set target_flux so that the target_number-brightest calibrators are
            # kept
            fluxes = source_skymodel.getColValues('I', aggregate='sum',
                                                  applyBeam=applyBeam_group)
            fluxes.sort()
            if target_number > len(fluxes):
                target_number = len(fluxes)
            target_flux = fluxes[-target_number] - 0.001

        # Save the model of the bright sources only, for later subtraction before
        # imaging if needed. Note that the bright-source model (like the other predict
        # models) must be a true-sky one, not an apparent one, so we have to transfer its
        # patches to the true-sky version later
        bright_source_skymodel_apparent_sky = source_skymodel.copy()
        if regroup:
            # Regroup by tessellating with the bright sources as the tessellation
            # centers
            self.log.info('Using a target flux density of {} Jy for grouping'.format(target_flux))
            source_skymodel.group('voronoi', targetFlux=target_flux, applyBeam=applyBeam_group,
                                  weightBySize=True)
            tesselation_patches = source_skymodel.getPatchNames()
            if len(tesselation_patches) == 1 and tesselation_patches[0] == 'Patch':
                # If there was insufficient flux in the sky model to meet the
                # target flux, LSMTool will put all sources into a single patch
                # with the name 'Patch'. So, check for this case and exit if found
                self.log.critical('No groups found that meet the target flux density. Please '
                                  'check the sky model (in working_dir/skymodels/calibrate_{}/) '
                                  'for problems or lower the target flux density. '
                                  'Exiting...'.format(index))
                sys.exit(1)

            # Update the patch positions after the tessellation to ensure they match the
            # ones from the meanshift grouping
            source_skymodel.setPatchPositions(patchDict=patch_dict)

            # Match the bright-source sky model to the tessellated one by removing
            # patches that are not present in the tessellated model
            bright_patch_names = bright_source_skymodel_apparent_sky.getPatchNames()
            for pn in bright_patch_names:
                if pn not in source_skymodel.getPatchNames():
                    bright_source_skymodel_apparent_sky.remove('Patch == {}'.format(pn))

            # debug
            dst_dir = os.path.join(self.working_dir, 'skymodels', 'calibrate_{}'.format(index))
            misc.create_directory(dst_dir)
            skymodel_true_sky_file = os.path.join(dst_dir, 'skymodel_voronoi.txt')
            source_skymodel.write(skymodel_true_sky_file, clobber=True)
            # debug

            # Transfer patches to the true-flux sky model (source names are identical
            # in both, but the order may be different)
            self.transfer_patches(source_skymodel, skymodel_true_sky, patch_dict=patch_dict)

        # For the bright-source true-sky model, duplicate any selections made above to the
        # apparent-sky model
        bright_source_skymodel = skymodel_true_sky.copy()
        source_names = bright_source_skymodel.getColValues('Name').tolist()
        bright_source_names = bright_source_skymodel_apparent_sky.getColValues('Name').tolist()
        matching_ind = []
        for i, sn in enumerate(bright_source_names):
            matching_ind.append(source_names.index(sn))
        bright_source_skymodel.select(np.array(matching_ind))

        # Transfer patches to the bright-source model if needed (i.e., regrouping was not
        # done)
        if not regroup:
            patch_dict = skymodel_true_sky.getPatchPositions()
            self.transfer_patches(skymodel_true_sky, bright_source_skymodel,
                                  patch_dict=patch_dict)

        # Save the bright-source (i.e., calibrator) flux densities (in Jy) for use
        # in the calibration pipeline for weighting of the directions during screen
        # fitting
        bright_source_skymodel.write(self.calibrators_only_skymodel_file, clobber=True)
        self.calibrator_patch_names = bright_source_skymodel.getPatchNames().tolist()
        self.calibrator_fluxes = bright_source_skymodel.getColValues('I', aggregate='sum').tolist()

        # Now remove any bright sources that lie outside the imaged area, as they
        # should not be peeled
        if len(self.imaging_sectors) > 0:
            for i, sector in enumerate(self.imaging_sectors):
                sm = bright_source_skymodel.copy()
                sm = sector.filter_skymodel(sm)
                if i == 0:
                    # For first sector, set filtered sky model
                    filtered_skymodel = sm
                else:
                    # For subsequent sectors, concat
                    if len(sm) > 0:
                        filtered_skymodel.concatenate(sm)
            bright_source_skymodel = filtered_skymodel
            if len(bright_source_skymodel) > 0:
                bright_source_skymodel.setPatchPositions()

        # Write sky models to disk for use in calibration, etc.
        calibration_skymodel = skymodel_true_sky
        self.num_patches = len(calibration_skymodel.getPatchNames())
        if self.num_patches > 1:
            infix = 'es'
        else:
            infix = ''
        self.log.info('Using {0} calibration patch{1}'.format(self.num_patches, infix))
        calibration_skymodel.write(self.calibration_skymodel_file, clobber=True)
        self.calibration_skymodel = calibration_skymodel
        if len(bright_source_skymodel) > 0:
            bright_source_skymodel.write(self.bright_source_skymodel_file, clobber=True)
        self.bright_source_skymodel = bright_source_skymodel

    def update_skymodels(self, index, regroup, target_flux=None,
                         target_number=None):
        """
        Updates the source and calibration sky models from the output sector sky model(s)

        Parameters
        ----------
        index : int
            Iteration index (counts starting from 1)
        regroup : bool
            Regroup sky model
        target_flux : float, optional
            Target flux in Jy for grouping
        target_number : int, optional
            Target number of patches for grouping
        """
        # Except for the first iteration, use the results of the previous iteration to
        # update the sky models, etc.
        if index == 1:
            # Make initial calibration and source sky models
            self.make_skymodels(self.parset['input_skymodel'],
                                skymodel_apparent_sky=self.parset['apparent_skymodel'],
                                regroup=self.parset['regroup_input_skymodel'],
                                find_sources=True, index=index)
        else:
            # Use the sector sky models from the previous iteration to update the master
            # sky model
            self.log.info('Updating sky model...')
            if self.imaged_sources_only:
                # Use new models from the imaged sectors only
                sector_skymodels_apparent_sky = [sector.image_skymodel_file_apparent_sky for
                                                 sector in self.imaging_sectors]
                sector_skymodels_true_sky = [sector.image_skymodel_file_true_sky for
                                             sector in self.imaging_sectors]
                sector_names = [sector.name for sector in self.imaging_sectors]
            else:
                # Use models from all sectors, whether imaged or not
                sector_skymodels_true_sky = []
                sector_skymodels_apparent_sky = None
                for sector in self.imaging_sectors + self.outlier_sectors:
                    if sector.is_outlier:
                        sector_skymodels_true_sky.append(sector.predict_skymodel_file)
                    else:
                        sector_skymodels_true_sky.append(sector.image_skymodel_file_true_sky)
                sector_names = [sector.name for sector in self.sectors]

            # Concatenate the sky models from all sectors, being careful not to duplicate
            # source and patch names
            for i, (sm, sn) in enumerate(zip(sector_skymodels_true_sky, sector_names)):
                if i == 0:
                    skymodel_true_sky = lsmtool.load(str(sm), beamMS=self.beam_ms_filename)
                    patchNames = skymodel_true_sky.getColValues('Patch')
                    new_patchNames = np.array(['{0}_{1}'.format(p, sn) for p in patchNames], dtype='U100')
                    skymodel_true_sky.setColValues('Patch', new_patchNames)
                    sourceNames = skymodel_true_sky.getColValues('Name')
                    new_sourceNames = np.array(['{0}_{1}'.format(s, sn) for s in sourceNames], dtype='U100')
                    skymodel_true_sky.setColValues('Name', new_sourceNames)
                else:
                    skymodel2 = lsmtool.load(str(sm))
                    patchNames = skymodel2.getColValues('Patch')
                    new_patchNames = np.array(['{0}_{1}'.format(p, sn) for p in patchNames], dtype='U100')
                    skymodel2.setColValues('Patch', new_patchNames)
                    sourceNames = skymodel2.getColValues('Name')
                    new_sourceNames = np.array(['{0}_{1}'.format(s, sn) for s in sourceNames], dtype='U100')
                    skymodel2.setColValues('Name', new_sourceNames)
                    table1 = skymodel_true_sky.table.filled()
                    table2 = skymodel2.table.filled()
                    skymodel_true_sky.table = vstack([table1, table2], metadata_conflicts='silent')
            skymodel_true_sky._updateGroups()
            skymodel_true_sky.setPatchPositions(method='wmean')

            if sector_skymodels_apparent_sky is not None:
                for i, (sm, sn) in enumerate(zip(sector_skymodels_apparent_sky, sector_names)):
                    if i == 0:
                        skymodel_apparent_sky = lsmtool.load(str(sm))
                        patchNames = skymodel_apparent_sky.getColValues('Patch')
                        new_patchNames = np.array(['{0}_{1}'.format(p, sn) for p in patchNames], dtype='U100')
                        skymodel_apparent_sky.setColValues('Patch', new_patchNames)
                        sourceNames = skymodel_apparent_sky.getColValues('Name')
                        new_sourceNames = np.array(['{0}_{1}'.format(s, sn) for s in sourceNames], dtype='U100')
                        skymodel_apparent_sky.setColValues('Name', new_sourceNames)
                    else:
                        skymodel2 = lsmtool.load(str(sm))
                        patchNames = skymodel2.getColValues('Patch')
                        new_patchNames = np.array(['{0}_{1}'.format(p, sn) for p in patchNames], dtype='U100')
                        skymodel2.setColValues('Patch', new_patchNames)
                        sourceNames = skymodel2.getColValues('Name')
                        new_sourceNames = np.array(['{0}_{1}'.format(s, sn) for s in sourceNames], dtype='U100')
                        skymodel2.setColValues('Name', new_sourceNames)
                        table1 = skymodel_apparent_sky.table.filled()
                        table2 = skymodel2.table.filled()
                        skymodel_apparent_sky.table = vstack([table1, table2], metadata_conflicts='silent')
                skymodel_apparent_sky._updateGroups()
                skymodel_apparent_sky.setPatchPositions(method='wmean')
            else:
                skymodel_apparent_sky = None

            # Use concatenated sky models to make new calibration model (we set find_sources
            # to False to preserve the source patches defined in the image pipeline by PyBDSF)
            self.make_skymodels(skymodel_true_sky, skymodel_apparent_sky=skymodel_apparent_sky,
                                regroup=regroup, find_sources=False, target_flux=target_flux,
                                target_number=target_number, index=index)

        # Adjust sector boundaries to avoid known sources and update their sky models.
        self.adjust_sector_boundaries()
        self.log.info('Making sector sky models (for predicting)...')
        for sector in self.imaging_sectors:
            sector.calibration_skymodel = self.calibration_skymodel.copy()
            sector.make_skymodel(index)

        # Make bright-source sectors containing only the bright sources that may be
        # subtracted before imaging. These sectors, like the outlier sectors above, are not
        # imaged
        self.define_bright_source_sectors(index)

        # Make outlier sectors containing any remaining calibration sources (not
        # included in any imaging or bright-source sector sky model). These sectors are
        # not imaged; they are only used in prediction and subtraction
        self.define_outlier_sectors(index)

        # Finally, make a list containing all sectors
        self.sectors = self.imaging_sectors + self.outlier_sectors + self.bright_source_sectors
        self.nsectors = len(self.sectors)

        # Clean up to minimize memory usage
        self.calibration_skymodel = None
        self.source_skymodel = None
        for sector in self.sectors:
            sector.calibration_skymodel = None
            sector.predict_skymodel = None
            sector.field.source_skymodel = None

    def transfer_patches(self, from_skymodel, to_skymodel, patch_dict=None):
        """
        Transfers the patches defined in from_skymodel to to_skymodel.

        Parameters
        ----------
        from_skymodel : sky model
            Sky model from which to transfer patches
        to_skymodel : sky model
            Sky model to which to transfer patches
        patch_dict : dict, optional
            Dict of patch positions

        Returns
        -------
        to_skymodel : sky model
            Sky model with patches matching those of from_skymodel
        """
        names_from = from_skymodel.getColValues('Name').tolist()
        names_to = to_skymodel.getColValues('Name').tolist()

        if set(names_from) == set(names_to):
            # Both sky models have the same sources, so use indexing
            ind_ss = np.argsort(names_from)
            ind_ts = np.argsort(names_to)
            to_skymodel.table['Patch'][ind_ts] = from_skymodel.table['Patch'][ind_ss]
            to_skymodel._updateGroups()
        elif set(names_to).issubset(set(names_from)):
            # The to_skymodel is a subset of from_skymodel, so use slower matching algorithm
            for ind_ts, name in enumerate(names_to):
                ind_ss = names_from.index(name)
                to_skymodel.table['Patch'][ind_ts] = from_skymodel.table['Patch'][ind_ss]
        else:
            # Skymodels don't match, raise error
            self.log.critical('Cannot transfer patches since from_skymodel does not contain '
                              'all the sources in to_skymodel! Exiting!')
            sys.exit(1)

        if patch_dict is not None:
            to_skymodel.setPatchPositions(patchDict=patch_dict)
        return to_skymodel

    def define_imaging_sectors(self):
        """
        Defines the imaging sectors
        """
        self.imaging_sectors = []

        # Determine whether we use a user-supplied list of sectors or a grid
        if len(self.parset['imaging_specific']['sector_center_ra_list']) > 0:
            # Use user-supplied list
            sector_center_ra_list = self.parset['imaging_specific']['sector_center_ra_list']
            sector_center_dec_list = self.parset['imaging_specific']['sector_center_dec_list']
            sector_width_ra_deg_list = self.parset['imaging_specific']['sector_width_ra_deg_list']
            sector_width_dec_deg_list = self.parset['imaging_specific']['sector_width_dec_deg_list']
            n = 1
            for ra, dec, width_ra, width_dec in zip(sector_center_ra_list, sector_center_dec_list,
                                                    sector_width_ra_deg_list, sector_width_dec_deg_list):
                name = 'sector_{0}'.format(n)
                self.imaging_sectors.append(Sector(name, ra, dec, width_ra, width_dec, self))
                n += 1
            if len(self.imaging_sectors) > 1:
                infix = 's'
            else:
                infix = ''
            self.log.info('Using {0} user-defined imaging sector{1}'.format(len(self.imaging_sectors), infix))
            # TODO: check whether flux density in each sector meets minimum and warn if not?
        else:
            # Make a regular grid of sectors
            if self.parset['imaging_specific']['grid_center_ra'] is None:
                image_ra = self.ra
            else:
                image_ra = self.parset['imaging_specific']['grid_center_ra']
            if self.parset['imaging_specific']['grid_center_dec'] is None:
                image_dec = self.dec
            else:
                image_dec = self.parset['imaging_specific']['grid_center_dec']
            if self.parset['imaging_specific']['grid_width_ra_deg'] is None:
                image_width_ra = self.fwhm_ra_deg
            else:
                image_width_ra = self.parset['imaging_specific']['grid_width_ra_deg']
            if self.parset['imaging_specific']['grid_width_dec_deg'] is None:
                image_width_dec = self.fwhm_dec_deg
            else:
                image_width_dec = self.parset['imaging_specific']['grid_width_dec_deg']

            nsectors_ra = self.parset['imaging_specific']['grid_nsectors_ra']
            if nsectors_ra == 0:
                # Force a single sector
                nsectors_ra = 1
                nsectors_dec = 1
            else:
                nsectors_dec = int(np.ceil(image_width_dec / (image_width_ra / nsectors_ra)))

            if nsectors_ra == 1 and nsectors_dec == 1:
                # Make a single sector
                nsectors_dec = 1
                width_ra = image_width_ra
                width_dec = image_width_dec
                center_x, center_y = self.radec2xy([image_ra], [image_dec])
                x = np.array([center_x])
                y = np.array([center_y])
            else:
                # Make the grid
                width_ra = image_width_ra / nsectors_ra
                width_dec = image_width_dec / nsectors_dec
                width_x = width_ra / abs(self.wcs.wcs.cdelt[0])
                width_y = width_dec / abs(self.wcs.wcs.cdelt[1])
                center_x, center_y = self.radec2xy([image_ra], [image_dec])
                min_x = center_x - width_x / 2.0 * (nsectors_ra - 1)
                max_x = center_x + width_x / 2.0 * (nsectors_ra - 1)
                min_y = center_y - width_y / 2.0 * (nsectors_dec - 1)
                max_y = center_y + width_y / 2.0 * (nsectors_dec - 1)
                x = np.linspace(min_x, max_x, nsectors_ra)
                y = np.linspace(min_y, max_y, nsectors_dec)
                x, y = np.meshgrid(x, y)

            # Initialize the sectors in the grid
            n = 1
            for i in range(nsectors_ra):
                for j in range(nsectors_dec):
                    if (self.parset['imaging_specific']['skip_corner_sectors'] and
                            i in [0, nsectors_ra-1] and j in [0, nsectors_dec-1] and
                            nsectors_ra > 2 and nsectors_dec > 2):
                        continue
                    name = 'sector_{0}'.format(n)
                    ra, dec = self.xy2radec([x[j, i]], [y[j, i]])
                    self.imaging_sectors.append(Sector(name, ra[0], dec[0], width_ra, width_dec, self))
                    n += 1
            if len(self.imaging_sectors) == 1:
                self.log.info('Using 1 imaging sector')
            else:
                self.log.info('Using {0} imaging sectors ({1} in RA, {2} in Dec)'.format(
                              len(self.imaging_sectors), nsectors_ra, nsectors_dec))

        # Compute bounding box for all imaging sectors and store as a
        # a semi-colon-separated list of [maxRA; minDec; minRA; maxDec] (we use semi-
        # colons as otherwise the pipeline parset parser will split the list). Also
        # store the midpoint as [midRA; midDec]. These values are needed for the aterm
        # image generation, so we use the padded polygons to ensure that the final
        # bounding box encloses all of the images *with* padding included.
        # Note: this is just once, rather than each time the sector borders are
        # adjusted, so that the image sizes do not change with iteration (so
        # mask images from previous iterations may be used)
        all_sectors = MultiPolygon([sector.poly_padded for sector in self.imaging_sectors])
        self.sector_bounds_xy = all_sectors.bounds
        maxRA, minDec = self.xy2radec([self.sector_bounds_xy[0]], [self.sector_bounds_xy[1]])
        minRA, maxDec = self.xy2radec([self.sector_bounds_xy[2]], [self.sector_bounds_xy[3]])
        midRA, midDec = self.xy2radec([(self.sector_bounds_xy[0]+self.sector_bounds_xy[2])/2.0],
                                      [(self.sector_bounds_xy[1]+self.sector_bounds_xy[3])/2.0])
        self.sector_bounds_deg = '[{0:.6f};{1:.6f};{2:.6f};{3:.6f}]'.format(maxRA[0], minDec[0],
                                                                            minRA[0], maxDec[0])
        self.sector_bounds_mid_deg = '[{0:.6f};{1:.6f}]'.format(midRA[0], midDec[0])

    def define_outlier_sectors(self, index):
        """
        Defines the outlier sectors

        Parameters
        ----------
        index : int
            Iteration index
        """
        self.outlier_sectors = []
        if self.peel_outliers:
            outlier_skymodel = self.make_outlier_skymodel()
            nsources = len(outlier_skymodel)
            if nsources > 0:
                # Choose number of sectors to be the no more than ten, but don't allow
                # fewer than 100 sources per sector if possible
                nnodes = max(min(10, round(nsources/100)), 1)  # TODO: tune to number of available nodes and/or memory?
                for i in range(nnodes):
                    outlier_sector = Sector('outlier_{0}'.format(i+1), self.ra, self.dec, 1.0, 1.0, self)
                    outlier_sector.is_outlier = True
                    outlier_sector.predict_skymodel = outlier_skymodel.copy()
                    startind = i * int(nsources/nnodes)
                    if i == nnodes-1:
                        endind = nsources
                    else:
                        endind = startind + int(nsources/nnodes)
                    outlier_sector.predict_skymodel.select(np.array(list(range(startind, endind))))
                    outlier_sector.make_skymodel(index)
                    self.outlier_sectors.append(outlier_sector)

    def define_bright_source_sectors(self, index):
        """
        Defines the bright source sectors

        Parameters
        ----------
        index : int
            Iteration index
        """
        self.bright_source_sectors = []
        if self.peel_bright_sources:
            nsources = len(self.bright_source_skymodel)
            if nsources > 0:
                # Choose number of sectors to be the no more than ten, but don't allow
                # fewer than 100 sources per sector if possible
                nnodes = max(min(10, round(nsources/100)), 1)  # TODO: tune to number of available nodes and/or memory?
                for i in range(nnodes):
                    bright_source_sector = Sector('bright_source_{0}'.format(i+1), self.ra, self.dec, 1.0, 1.0, self)
                    bright_source_sector.is_bright_source = True
                    bright_source_sector.predict_skymodel = self.bright_source_skymodel.copy()
                    startind = i * int(nsources/nnodes)
                    if i == nnodes-1:
                        endind = nsources
                    else:
                        endind = startind + int(nsources/nnodes)
                    bright_source_sector.predict_skymodel.select(np.array(list(range(startind, endind))))
                    bright_source_sector.make_skymodel(index)
                    self.bright_source_sectors.append(bright_source_sector)

    def find_intersecting_sources(self):
        """
        Finds sources that intersect with the intial sector boundaries

        Returns
        -------
        intersecting_source_polys: list of Polygons
            List of source polygons that intersect one or more sector boundaries
        """
        idx = rtree.index.Index()
        skymodel = self.source_skymodel
        RA, Dec = skymodel.getPatchPositions(asArray=True)
        x, y = self.radec2xy(RA, Dec)
        sizes = skymodel.getPatchSizes(units='degree')
        minsize = 1  # minimum allowed source size in pixels
        sizes = [max(minsize, s/2.0/self.wcs_pixel_scale) for s in sizes]  # radii in pixels

        for i, (xs, ys, ss) in enumerate(zip(x, y, sizes)):
            xmin = xs - ss
            xmax = xs + ss
            ymin = ys - ss
            ymax = ys + ss
            idx.insert(i, (xmin, ymin, xmax, ymax))

        # For each sector side, query the index to find any intersections
        intersecting_ind = []
        buffer = 2  # how many pixels away from each side to check
        for sector in self.imaging_sectors:
            xmin, ymin, xmax, ymax = sector.initial_poly.bounds
            side1 = (xmin-buffer, ymin, xmin+buffer, ymax)
            intersecting_ind.extend(list(idx.intersection(side1)))
            side2 = (xmax-buffer, ymin, xmax+buffer, ymax)
            intersecting_ind.extend(list(idx.intersection(side2)))
            side3 = (xmin, ymin-buffer, xmax, ymin+buffer)
            intersecting_ind.extend(list(idx.intersection(side3)))
            side4 = (xmin, ymax-buffer, xmax, ymax+buffer)
            intersecting_ind.extend(list(idx.intersection(side4)))

        # Make polygons for intersecting sources, with a size = 1.5 * radius of source
        if len(intersecting_ind) > 0:
            xfilt = np.array(x)[(np.array(intersecting_ind),)]
            yfilt = np.array(y)[(np.array(intersecting_ind),)]
            sfilt = np.array(sizes)[(np.array(intersecting_ind),)]
            intersecting_source_polys = [Point(xp, yp).buffer(sp*1.5) for
                                         xp, yp, sp in zip(xfilt, yfilt, sfilt)]
        else:
            intersecting_source_polys = []
        return intersecting_source_polys

    def adjust_sector_boundaries(self):
        """
        Adjusts the imaging sector boundaries for overlaping sources
        """
        self.log.info('Adusting sector boundaries to avoid sources...')

        # Note: this adjustment only needs to be done when there are multiple sectors,
        # since its purpose is to ensure that sources don't fall in between sectors
        if len(self.imaging_sectors) > 1:
            intersecting_source_polys = self.find_intersecting_sources()
            for sector in self.imaging_sectors:
                # Make sure all sectors start from their initial polygons
                sector.poly = sector.initial_poly
            for sector in self.imaging_sectors:
                # Adjust boundaries for intersection with sources
                for p2 in intersecting_source_polys:
                    poly_bkup = sector.poly
                    if sector.poly.contains(p2.centroid):
                        # If point is inside, union the sector poly with the source one
                        sector.poly = sector.poly.union(p2)
                    else:
                        # If centroid of point is outside, difference the sector poly with
                        # the source one
                        sector.poly = sector.poly.difference(p2)
                    if type(sector.poly) is not Polygon:
                        # use backup
                        sector.poly = poly_bkup

        # Make sector region and vertices files
        for sector in self.imaging_sectors:
            sector.make_vertices_file()
            sector.make_region_file(os.path.join(self.working_dir, 'regions',
                                                 '{}_region_ds9.reg'.format(sector.name)))

    def make_outlier_skymodel(self):
        """
        Make a sky model of any outlier calibration sources, not included in any
        imaging sector
        """
        all_source_names = self.calibration_skymodel.getColValues('Name').tolist()
        sector_source_names = []
        for sector in self.imaging_sectors:
            skymodel = lsmtool.load(str(sector.predict_skymodel_file))
            sector_source_names.extend(skymodel.getColValues('Name').tolist())
        if self.peel_bright_sources:
            # The bright sources were removed from the sector predict sky models, so
            # add them to the list
            sector_source_names.extend(self.bright_source_skymodel.getColValues('Name').tolist())

        outlier_ind = np.array([all_source_names.index(sn) for sn in all_source_names
                                if sn not in sector_source_names])
        outlier_skymodel = self.calibration_skymodel.copy()
        outlier_skymodel.select(outlier_ind, force=True)
        return outlier_skymodel

    def radec2xy(self, RA, Dec):
        """
        Returns x, y for input RA, Dec

        Parameters
        ----------
        RA : list
            List of RA values in degrees
        Dec : list
            List of Dec values in degrees

        Returns
        -------
        x, y : list, list
            Lists of x and y pixel values corresponding to the input RA and Dec
            values
        """
        x = []
        y = []

        for ra_deg, dec_deg in zip(RA, Dec):
            ra_dec = np.array([[ra_deg, dec_deg]])
            x.append(self.wcs.wcs_world2pix(ra_dec, 0)[0][0])
            y.append(self.wcs.wcs_world2pix(ra_dec, 0)[0][1])
        return x, y

    def xy2radec(self, x, y):
        """
        Returns input RA, Dec for input x, y

        Parameters
        ----------
        x : list
            List of x values in pixels
        y : list
            List of y values in pixels

        Returns
        -------
        RA, Dec : list, list
            Lists of RA and Dec values corresponding to the input x and y pixel
            values
        """
        RA = []
        Dec = []

        for xp, yp in zip(x, y):
            x_y = np.array([[xp, yp]])
            RA.append(self.wcs.wcs_pix2world(x_y, 0)[0][0])
            Dec.append(self.wcs.wcs_pix2world(x_y, 0)[0][1])
        return RA, Dec

    def makeWCS(self):
        """
        Makes simple WCS object

        Returns
        -------
        w : astropy.wcs.WCS object
            A simple TAN-projection WCS object for specified reference position
        """
        from astropy.wcs import WCS

        self.wcs_pixel_scale = 10.0 / 3600.0  # degrees/pixel (= 10"/pixel)
        w = WCS(naxis=2)
        w.wcs.crpix = [1000, 1000]
        w.wcs.cdelt = np.array([-self.wcs_pixel_scale, self.wcs_pixel_scale])
        w.wcs.crval = [self.ra, self.dec]
        w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        w.wcs.set_pv([(2, 1, 45.0)])
        self.wcs = w

    def check_selfcal_progress(self):
        """
        Checks whether selfcal has converged or diverged by comparing the current
        image noise to that of the previous cycle

        Convergence is determined by comparing the noise ratio to
        self.convergence_ratio, which is the minimum ratio of the current noise
        to the previous noise above which selfcal is considered to have
        converged (must be in the range 0.5 -- 2). E.g., self.convergence_ratio
        = 0.95 means that the image noise must improve by ~ 5% or more from the
        previous cycle for selfcal to be considered as nonconverged

        Divergence is determined by comparing the noise ratio to
        self.divergence_ratio, which is the minimum ratio of the current noise
        to the previous noise above which selfcal is considered to have diverged
        (must be >= 1). E.g., divergence_ratio = 1.1 means that, if image noise
        worsens by ~ 10% or more from the previous cycle, selfcal is considered
        to have diverged

        Returns
        -------
        converged : bool
            True if selfcal has converged, False if not
        diverged : bool
            True if selfcal has diverged, False if not
        """
        convergence_ratio = self.convergence_ratio
        divergence_ratio = self.divergence_ratio

        # Check that convergence and divergence limits are sensible
        if convergence_ratio > 2.0:
            self.log.info('The convergence ratio is set to {} but must be <= 2. '
                          'Using 2.0 instead'.format(convergence_ratio))
            convergence_ratio = 2.0
        if convergence_ratio < 0.5:
            self.log.info('The convergence ratio is set to {} but must be >= 0.5. '
                          'Using 0.5 instead'.format(convergence_ratio))
            convergence_ratio = 0.5
        if divergence_ratio < 1.0:
            self.log.info('The divergence ratio is set to {} but must be >= 1. '
                          'Using 1.0 instead'.format(divergence_ratio))
            divergence_ratio = 1.0

        if self.field_image_filename_prev is None:
            # No previous iteration, so report nonconvergence
            return False

        # Get noise from previous and current images
        image = FITSImage(self.field_image_filename_prev)
        image.calc_noise()
        rmspre = image.noise
        image = FITSImage(self.field_image_filename)
        image.calc_noise()
        rmspost = image.noise
        self.log.info('Ratio of current image noise to previous image '
                      'noise = {0:.2f}'.format(rmspost/rmspre))

        if rmspost / rmspre < convergence_ratio:
            # Report not converged (and not diverged)
            return False, False
        elif rmspost / rmspre > divergence_ratio:
            # Report diverged (and not converged)
            return False, True
        else:
            # Report converged (and not diverged)
            return True, False

    def update(self, step_dict, index):
        """
        Updates parameters, sky models, etc. for current step
        """
        # If the data fraction has changed, create new observations
        if step_dict['data_fraction'] != self.__dict__['data_fraction']:
            self.chunk_observations(step_dict['data_fraction'])

        # Update field and sector dicts with the parameters for this iteration
        self.__dict__.update(step_dict)
        for sector in self.imaging_sectors:
            sector.__dict__.update(step_dict)
        self.update_skymodels(index, step_dict['regroup_model'],
                              target_flux=step_dict['target_flux'])

        # Check whether outliers and bright sources need to be peeled
        nr_outlier_sectors = len(self.outlier_sectors)
        nr_imaging_sectors = len(self.imaging_sectors)
        nr_bright_source_sectors = len(self.bright_source_sectors)
        if nr_bright_source_sectors == 0:
            self.peel_bright_sources = False
        if nr_outlier_sectors == 0:
            self.peel_outliers = False
            self.imaged_sources_only = True  # No outliers means all sources are imaged

        # Determine whether a predict step is needed or not. It's needed when:
        # - there are two or more imaging sectors
        # - there are one or more outlier sectors (whether or not the outliers will
        #   be peeled for calibration, which is set by self.peel_outliers, they must
        #   still be predicted since they lie outside of the imaged areas)
        # - bright sources will be peeled, set by self.peel_bright_sources
        # - reweighting is to be done (since datasets with all sources subtracted are
        #   needed for this)
        if (nr_imaging_sectors > 1 or
                nr_outlier_sectors > 0 or
                self.peel_bright_sources or
                self.reweight):
            self.do_predict = True
        else:
            self.do_predict = False
