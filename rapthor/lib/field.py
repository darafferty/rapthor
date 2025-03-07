"""
Definition of the Field class
"""
import os
import logging
import numpy as np
import lsmtool
import lsmtool.skymodel
from rapthor.lib import miscellaneous as misc
from rapthor.lib.observation import Observation
from rapthor.lib.sector import Sector
from rapthor.lib.facet import read_ds9_region_file, read_skymodel
from shapely.geometry import Point, Polygon, MultiPolygon
from astropy.table import vstack
import rtree.index
import glob
from astropy.coordinates import SkyCoord, Angle
import astropy.units as u
from typing import List, Dict
from collections import namedtuple
import matplotlib
matplotlib.use('Agg')
from matplotlib.patches import Ellipse
from matplotlib.pyplot import figure
from astropy.visualization.wcsaxes import SphericalCircle
import mocpy
from losoto.h5parm import h5parm


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
    def __init__(self, parset, minimal=False):
        # Initialize basic attributes. These can be overridden later by the strategy
        # values and/or the operations
        self.name = 'field'
        self.log = logging.getLogger('rapthor:{}'.format(self.name))
        self.parset = parset.copy()
        self.working_dir = self.parset['dir_working']
        self.ms_filenames = self.parset['mss']
        self.numMS = len(self.ms_filenames)
        self.data_colname = 'DATA'
        self.bda_timebase_fast = self.parset['calibration_specific']['fast_bda_timebase']
        self.bda_timebase_slow_joint = self.parset['calibration_specific']['slow_bda_timebase_joint']
        self.bda_timebase_slow_separate = self.parset['calibration_specific']['slow_bda_timebase_separate']
        self.dd_interval_factor = self.parset['calibration_specific']['dd_interval_factor']
        self.h5parm_filename = self.parset['input_h5parm']
        self.fulljones_h5parm_filename = self.parset['input_fulljones_h5parm']
        self.dde_mode = self.parset['dde_mode']
        self.fast_smoothnessconstraint = self.parset['calibration_specific']['fast_smoothnessconstraint']
        self.fast_smoothnessreffrequency = self.parset['calibration_specific']['fast_smoothnessreffrequency']
        self.fast_smoothnessrefdistance = self.parset['calibration_specific']['fast_smoothnessrefdistance']
        self.slow_smoothnessconstraint_joint = self.parset['calibration_specific']['slow_smoothnessconstraint_joint']
        self.slow_smoothnessconstraint_separate = self.parset['calibration_specific']['slow_smoothnessconstraint_separate']
        self.fulljones_timestep_sec = self.parset['calibration_specific']['fulljones_timestep_sec']
        self.smoothnessconstraint_fulljones = self.parset['calibration_specific']['fulljones_smoothnessconstraint']
        self.propagatesolutions = self.parset['calibration_specific']['propagatesolutions']
        self.solveralgorithm = self.parset['calibration_specific']['solveralgorithm']
        self.onebeamperpatch = self.parset['calibration_specific']['onebeamperpatch']
        self.llssolver = self.parset['calibration_specific']['llssolver']
        self.maxiter = self.parset['calibration_specific']['maxiter']
        self.stepsize = self.parset['calibration_specific']['stepsize']
        self.stepsigma = self.parset['calibration_specific']['stepsigma']
        self.tolerance = self.parset['calibration_specific']['tolerance']
        self.dde_method = self.parset['imaging_specific']['dde_method']
        self.save_visibilities = self.parset['imaging_specific']['save_visibilities']
        self.use_mpi = self.parset['imaging_specific']['use_mpi']
        self.parallelbaselines = self.parset['calibration_specific']['parallelbaselines']
        self.sagecalpredict = self.parset['calibration_specific']['sagecalpredict']
        self.fast_datause = self.parset['calibration_specific']['fast_datause']
        self.slow_datause = self.parset['calibration_specific']['slow_datause']
        self.reweight = self.parset['imaging_specific']['reweight']
        self.do_multiscale_clean = self.parset['imaging_specific']['do_multiscale_clean']
        self.apply_diagonal_solutions = self.parset['imaging_specific']['apply_diagonal_solutions']
        self.make_quv_images = self.parset['imaging_specific']['make_quv_images']
        self.pol_combine_method = self.parset['imaging_specific']['pol_combine_method']
        self.solverlbfgs_dof = self.parset['calibration_specific']['solverlbfgs_dof']
        self.solverlbfgs_iter = self.parset['calibration_specific']['solverlbfgs_iter']
        self.solverlbfgs_minibatches = self.parset['calibration_specific']['solverlbfgs_minibatches']
        self.cycle_number = 1
        self.apply_amplitudes = False
        self.apply_screens = False

        # Set strategy parameter defaults
        self.fast_timestep_sec = 8.0
        self.slow_timestep_joint_sec = 0
        self.slow_timestep_separate_sec = 600.0
        self.convergence_ratio = 0.95
        self.divergence_ratio = 1.1
        self.failure_ratio = 10.0
        self.max_distance = 3.0
        self.max_normalization_delta = 0.3
        self.solve_min_uv_lambda = 150
        self.scale_normalization_delta = True
        self.lofar_to_true_flux_ratio = 1.0
        self.lofar_to_true_flux_std = 0.0
        self.peel_outliers = False
        self.imaged_sources_only = False
        self.peel_bright_sources = False
        self.peel_non_calibrator_sources = False
        self.do_slowgain_solve = False
        self.do_normalize = False
        self.make_image_cube = False
        self.use_scalarphase = True
        self.field_image_filename_prev = None
        self.field_image_filename = None

        # Scan MS files to get observation info
        self.scan_observations()

        if not minimal:
            # Scan calibration h5parm files (if any) to get solution info
            self.scan_h5parms()

            # Set up imaging sectors
            self.makeWCS()
            self.define_imaging_sectors()

    def scan_observations(self):
        """
        Checks input MS files and initializes the associated Observation objects
        """
        suffix = 's' if len(self.ms_filenames) > 1 else ''
        self.log.debug('Scanning input MS file{}...'.format(suffix))
        self.full_observations = []
        for ms_filename in self.ms_filenames:
            self.full_observations.append(Observation(ms_filename))
        self.observations = self.full_observations[:]  # make copy so we don't alter originals

        # Define a reference observation for the comparisons below
        obs0 = self.full_observations[0]

        # Check that all observations have the same antenna type
        self.antenna = obs0.antenna
        for obs in self.full_observations:
            if self.antenna != obs.antenna:
                raise ValueError('Antenna type for MS {0} differs from the one for MS '
                                 '{1}'.format(obs.ms_filename, obs0.ms_filename))

        # Check for multiple epochs
        self.epoch_starttimes = set([obs.starttime for obs in self.full_observations])
        suffix = 's' if len(self.epoch_starttimes) > 1 else ''
        self.log.debug('Input data comprise {0} epoch{1}'.format(len(self.epoch_starttimes), suffix))
        self.epoch_observations = []
        for i, epoch_starttime in enumerate(self.epoch_starttimes):
            epoch_observations = [obs for obs in self.full_observations if
                                  obs.starttime == epoch_starttime]
            self.epoch_observations.append(epoch_observations)
            if len(epoch_observations) > 1:
                # Multiple MS files per epoch implies differing frequencies. Check for
                # overlapping frequency coverage and raise error if found
                startfreqs = np.array([obs.startfreq-obs.channelwidth/2 for obs in epoch_observations])
                endfreqs = np.array([obs.endfreq+obs.channelwidth/2 for obs in epoch_observations])
                msfiles = np.array([obs.ms_filename for obs in epoch_observations])
                sort_ind = np.argsort(startfreqs)
                for j, (startfreq, endfreq) in enumerate(zip(startfreqs[sort_ind][1:],
                                                             endfreqs[sort_ind][:-1])):
                    if startfreq < endfreq:
                        ms1 = msfiles[sort_ind][j]  # MS file from which endfreq is taken
                        ms2 = msfiles[sort_ind][j+1]  # MS file from which startfreq is taken
                        raise ValueError('Overlapping frequency coverage found for the '
                                         f'following input MS files: {ms1} and {ms2}')

        # Check that all observations have the same pointing
        self.ra = obs0.ra
        self.dec = obs0.dec
        for obs in self.full_observations:
            if self.ra != obs.ra or self.dec != obs.dec:
                raise ValueError('Pointing for MS {0} differs from the one for MS '
                                 '{1}'.format(obs.ms_filename, obs0.ms_filename))

        # Check that all observations have the same station diameter
        self.diam = obs0.diam
        for obs in self.full_observations:
            if self.diam != obs.diam:
                raise ValueError('Station diameter for MS {0} differs from the one for MS '
                                 '{1}'.format(obs.ms_filename, obs0.ms_filename))

        # Check that all observations have the same stations
        self.stations = obs0.stations
        for obs in self.full_observations:
            if self.stations != obs.stations:
                raise ValueError('Stations in MS {0} differ from those in MS '
                                 '{1}'.format(obs.ms_filename, obs0.ms_filename))

        # Find mean elevation and FOV over all observations
        el_rad_list = []
        ref_freq_list = []
        for obs in self.full_observations:
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
        times = [(obs.endtime+obs.starttime)/2.0 for obs in self.full_observations]
        weights = [(obs.endtime-obs.starttime) for obs in self.full_observations]
        mid_time = np.average(times, weights=weights)
        mid_index = np.argmin(np.abs(np.array(times)-mid_time))
        self.beam_ms_filename = self.full_observations[mid_index].ms_filename

    def chunk_observations(self, mintime, prefer_high_el_periods=True):
        """
        Break observations into smaller time chunks if desired

        Chunking is done if:

        - obs.data_fraction < 1 (so part of an observation is to be processed)
        - nobs * nsectors < nnodes (so all nodes can be used efficiently. In particular,
          the predict operation parallelizes over sectors and observations, so we need
          enough observations to allow all nodes to be occupied.)

        Parameters
        ----------
        mintime : float
            Minimum time in sec for a chunk
        prefer_high_el_periods : bool, optional
            Prefer periods for which the elevation is in the highest 80% of values for a
            given observation. This option is useful for removing periods of lower
            signal-to-noise (e.g., due to being at lower elevations where ionospheric
            activity can increase and sensitivity decrease). If the requested mintime is
            larger than the total time of the high-elevation period for a given
            observation, then the full observation is used instead
        """
        # Set the chunk size so that it is at least mintime
        self.observations = []
        for obs in self.full_observations:
            target_starttime = obs.starttime
            target_endtime = obs.endtime
            data_fraction = obs.data_fraction
            if prefer_high_el_periods and (mintime < obs.high_el_endtime - obs.high_el_starttime):
                # Use high-elevation period for chunking. We increase the data fraction
                # to account for the decreased total observation time so that the
                # amount of data used is kept the same
                target_starttime = obs.high_el_starttime
                target_endtime = obs.high_el_endtime
                data_fraction = min(1, data_fraction * (obs.endtime - obs.starttime) /
                                    (target_endtime - target_starttime))
            tottime = target_endtime - target_starttime

            nchunks = max(1, int(np.floor(data_fraction / (mintime / tottime))))
            if nchunks == 1:
                # Center the chunk around the midpoint (which is generally the most
                # sensitive, near transit)
                midpoint = target_starttime + tottime / 2
                chunktime = min(tottime, max(mintime, data_fraction*tottime))
                if chunktime < tottime:
                    self.observations.append(Observation(obs.ms_filename,
                                                         starttime=midpoint-chunktime/2,
                                                         endtime=midpoint+chunktime/2))
                else:
                    self.observations.append(obs)
            else:
                steptime = mintime * (tottime / mintime - nchunks) / nchunks + mintime
                starttimes = np.arange(target_starttime, target_endtime, steptime)
                endtimes = np.arange(target_starttime+mintime, target_endtime+mintime, steptime)
                for starttime, endtime in zip(starttimes, endtimes):
                    if endtime > target_endtime:
                        endtime = target_endtime
                        if endtime - starttime < mintime / 2:
                            # Skip this chunk if too short
                            self.log.warning('Skipping final chunk as being too short. Fraction '
                                             'of data used will be lower than requested.')
                            continue
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

        # Update the copies stored in the imaging sectors (including the full-field
        # sector, used to make the initial sky model). Other (non-imaging) sectors do not
        # need to be updated
        sectors_to_update = self.imaging_sectors[:]  # don't alter original with append below
        if hasattr(self, 'full_field_sector'):
            sectors_to_update.append(self.full_field_sector)
        for sector in sectors_to_update:
            sector.observations = []
            for obs in self.observations:
                cobs = obs.copy()
                sector.observations.append(cobs)

    def set_obs_parameters(self):
        """
        Sets parameters for all observations from current parset and sky model
        """
        ntimechunks = 0
        nfreqchunks_joint = 0
        nfreqchunks_separate = 0
        nfreqchunks_fulljones = 0
        for obs in self.observations:
            obs.set_calibration_parameters(self.parset, self.num_patches, len(self.observations),
                                           self.calibrator_fluxes, self.fast_timestep_sec,
                                           self.slow_timestep_joint_sec, self.slow_timestep_separate_sec,
                                           self.fulljones_timestep_sec, self.target_flux)
            ntimechunks += obs.ntimechunks
            nfreqchunks_joint += obs.nfreqchunks_joint
            nfreqchunks_separate += obs.nfreqchunks_separate
            nfreqchunks_fulljones += obs.nfreqchunks_fulljones
        self.ntimechunks = ntimechunks
        self.nfreqchunks_joint = nfreqchunks_joint
        self.nfreqchunks_separate = nfreqchunks_separate
        self.nfreqchunks_fulljones = nfreqchunks_fulljones

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
                       calibrator_max_dist_deg=None, index=0):
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
            operation
        target_flux : float, optional
            Target flux in Jy for grouping
        target_number : int, optional
            Target number of patches for grouping
        calibrator_max_dist_deg : float, optional
            Maximum distance in degrees from phase center for grouping
        index : index
            Iteration index
        """
        # Save the filename of the calibrator-only sky model from the previous cycle (needed
        # for some operations), if available
        if index > 1:
            dst_dir_prev_cycle = os.path.join(self.working_dir, 'skymodels', 'calibrate_{}'.format(index-1))
            self.calibrators_only_skymodel_file_prev_cycle = os.path.join(dst_dir_prev_cycle,
                                                                          'calibrators_only_skymodel.txt')
        else:
            self.calibrators_only_skymodel_file_prev_cycle = None

        # Make output directories for new sky models and define filenames
        dst_dir = os.path.join(self.working_dir, 'skymodels', 'calibrate_{}'.format(index))
        os.makedirs(dst_dir, exist_ok=True)
        self.calibration_skymodel_file = os.path.join(dst_dir, 'calibration_skymodel.txt')
        self.calibrators_only_skymodel_file = os.path.join(dst_dir, 'calibrators_only_skymodel.txt')
        self.source_skymodel_file = os.path.join(dst_dir, 'source_skymodel.txt')
        dst_dir = os.path.join(self.working_dir, 'skymodels', 'image_{}'.format(index))
        os.makedirs(dst_dir, exist_ok=True)
        self.bright_source_skymodel_file = os.path.join(dst_dir, 'bright_source_skymodel.txt')

        # First check whether sky models already exist due to a previous run and attempt
        # to load them
        try:
            self.calibration_skymodel = lsmtool.load(str(self.calibration_skymodel_file))
            self.calibrators_only_skymodel = lsmtool.load(str(self.calibrators_only_skymodel_file))
            self.source_skymodel = lsmtool.load(str(self.source_skymodel_file))

            if self.peel_bright_sources:
                # The bright-source model file may not exist if there are no bright sources,
                # but also if a reset of the imaging operation was done. Unfortunately, there
                # is no way to determine which of these two possibilities is the case. So, if
                # it does not exist, we have to regenerate the sky models
                if os.path.exists(self.bright_source_skymodel_file):
                    self.bright_source_skymodel = lsmtool.load(str(self.bright_source_skymodel_file))
                    all_skymodels_loaded = True
                else:
                    all_skymodels_loaded = False
            else:
                self.bright_source_skymodel = self.calibrators_only_skymodel
                all_skymodels_loaded = True
        except IOError:
            all_skymodels_loaded = False

        if all_skymodels_loaded:
            # If all the required sky models were loaded from files, return; otherwise,
            # continue on to regenerate the sky models from scratch
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

        # Save the model of the bright sources only, for later subtraction before
        # imaging if needed. Note that the bright-source model (like the other predict
        # models) must be a true-sky one, not an apparent one, so we have to transfer its
        # patches to the true-sky version later
        bright_source_skymodel_apparent_sky = source_skymodel.copy()

        # Regroup the true-sky model into calibration patches
        if regroup:
            if self.parset['facet_layout'] is not None:
                # Regroup using the supplied ds9 region file of the facets
                facets = read_ds9_region_file(self.parset['facet_layout'])
                suffix = 'es' if len(facets) > 1 else ''
                self.log.info(f'Read {len(facets)} patch{suffix} from supplied facet '
                              'layout file')
                facet_names = []
                facet_patches_dict = {}
                for facet in facets:
                    facet_names.append(facet.name)
                    facet_patches_dict.update({facet.name: [facet.ra, facet.dec]})

                if len(facet_names) > len(skymodel_true_sky):
                    raise ValueError('The sky model has {0} sources but the input facet '
                                     'layout file has {1} facets. There must be at least '
                                     'as many sources in the sky model as facets in the '
                                     'facet layout file.'.format(len(skymodel_true_sky),
                                                                 len(facet_names)))

                # Update the sky models with the new patches and group using the
                # Voronoi algorithm. We do this by setting the "Patch" column
                # values to a list of the patch names we want (here a simple
                # repeating list is sufficient; it's only necessary that each
                # patch name appear at least once) and the patch positions to
                # those from the facet file. Grouping will then use the new
                # patches
                for skymodel in [skymodel_true_sky, bright_source_skymodel_apparent_sky]:
                    patch_names = facet_names * (len(skymodel) // len(facet_names) + 1)
                    skymodel.setColValues('Patch', patch_names[:len(skymodel)])
                    for i in range(2):
                        # Update the patch positions twice (before and after grouping). We
                        # need to do it after as well since group() changes the positions
                        for patch, pos in facet_patches_dict.items():
                            pos[0] = Angle(pos[0], unit=u.deg)
                            pos[1] = Angle(pos[1], unit=u.deg)
                            skymodel.table.meta[patch] = pos
                        if i == 0:
                            skymodel.group('voronoi', patchNames=facet_names)

                n_removed = len(facet_names) - len(skymodel_true_sky.getPatchNames().tolist())
                if n_removed > 0:
                    # One or more empty facets removed during grouping, so
                    # report this to user
                    suffix = 'es' if n_removed > 1 else ''
                    self.log.warning(f'Removed {n_removed} empty patch{suffix}. The facet '
                                     'layout used in this cycle will therefore differ '
                                     'from that given in the input facet layout file.')
            else:
                # Regroup by tessellating with the bright sources as the tessellation
                # centers.
                # First do some checks:
                if target_flux is None and target_number is None:
                    raise ValueError('Either the target flux density or the target number '
                                     'of directions must be specified when regrouping the '
                                     'sky model.')
                if target_flux is not None and target_flux <= 0.0:
                    raise ValueError('The target flux density cannot be less than or equal '
                                     'to 0.')
                if target_number is not None and target_number < 1:
                    raise ValueError('The target number of directions cannot be less than 1.')

                # Apply the distance filter (if any), keeping only patches inside the
                # maximum distance
                if calibrator_max_dist_deg is not None:
                    names, distances = self.get_source_distances(source_skymodel.getPatchPositions())
                    inside_ind = np.where(distances < calibrator_max_dist_deg)
                    calibrator_names = names[inside_ind]
                else:
                    calibrator_names = source_skymodel.getPatchNames()
                all_names = source_skymodel.getPatchNames()
                keep_ind = np.array([i for i, name in enumerate(all_names) if name in calibrator_names])
                calibrator_names = all_names[keep_ind]  # to ensure order matches that of fluxes
                all_fluxes = source_skymodel.getColValues('I', aggregate='sum', applyBeam=applyBeam_group)
                fluxes = all_fluxes[keep_ind]

                # Check if target flux can be met in at least one direction
                total_flux = np.sum(fluxes)
                if total_flux < target_flux:
                    raise RuntimeError('There is insufficient flux density in the model to meet '
                                       'the target flux density. Please check the sky model '
                                       '(in dir_working/skymodels/calibrate_{}/) for problems, '
                                       'or lower the target flux density and/or increase the '
                                       'maximum calibrator distance.'.format(index))

                # Weight the fluxes by source size (larger sources are down weighted)
                sizes = source_skymodel.getPatchSizes(units='arcsec', weight=True,
                                                      applyBeam=applyBeam_group)
                sizes = sizes[keep_ind]
                sizes[sizes < 1.0] = 1.0
                medianSize = np.median(sizes)
                weights = medianSize / sizes
                weights[weights > 1.0] = 1.0
                weights[weights < 0.5] = 0.5
                fluxes *= weights

                # Determine the flux cut to use to select the bright sources (calibrators)
                if target_number is not None:
                    # Set target_flux so that the target_number-brightest calibrators are
                    # kept
                    if target_number >= len(fluxes):
                        target_number = len(fluxes)
                        target_flux_for_number = np.min(fluxes)
                    else:
                        target_flux_for_number = np.sort(fluxes)[-target_number]

                    if target_flux is None:
                        target_flux = target_flux_for_number
                        self.log.info('Using a target flux density of {0:.2f} Jy for grouping '
                                      'to meet the specified target number of '
                                      'directions ({1:.2f})'.format(target_flux, target_number))
                    else:
                        if target_flux_for_number > target_flux and target_number < len(fluxes):
                            # Only use the new target flux if the old value might result
                            # in more than target_number of calibrators
                            self.log.info('Using a target flux density of {0:.2f} Jy for '
                                          'grouping (raised from {1:.2f} Jy to ensure that '
                                          'the target number of {2} directions is not '
                                          'exceeded)'.format(target_flux_for_number, target_flux, target_number))
                            target_flux = target_flux_for_number
                        else:
                            self.log.info('Using a target flux density of {0:.2f} Jy for grouping'.format(target_flux))
                else:
                    self.log.info('Using a target flux density of {0:.2f} Jy for grouping'.format(target_flux))

                # Check if target flux can be met for at least one source
                #
                # Note: the weighted fluxes are used here (with larger sources down-weighted)
                if np.max(fluxes) < target_flux:
                    raise RuntimeError('No sources found that meet the target flux density (after '
                                       'down-weighting larger sources by up to a factor of two). Please '
                                       'check the sky model (in dir_working/skymodels/calibrate_{}/) '
                                       'for problems, or lower the target flux density and/or increase '
                                       'the maximum calibrator distance.'.format(index))

                # Tesselate the model
                calibrator_names = calibrator_names[np.where(fluxes >= target_flux)]
                source_skymodel.group('voronoi', patchNames=calibrator_names)

                # Update the patch positions after the tessellation to ensure they match the
                # ones from the meanshift grouping
                source_skymodel.setPatchPositions(patchDict=patch_dict)

                # Match the bright-source sky model to the tessellated one by removing
                # patches that are not present in the tessellated model
                bright_patch_names = bright_source_skymodel_apparent_sky.getPatchNames()
                for pn in bright_patch_names:
                    if pn not in source_skymodel.getPatchNames():
                        bright_source_skymodel_apparent_sky.remove('Patch == {}'.format(pn))

                # Transfer patches to the true-flux sky model (source names are identical
                # in both, but the order may be different)
                misc.transfer_patches(source_skymodel, skymodel_true_sky, patch_dict=patch_dict)

                # Rename the patches so that the numbering starts at high Dec, high RA
                # and increases as RA and Dec decrease (i.e., from top-left to bottom-right)
                for model in [source_skymodel, skymodel_true_sky, bright_source_skymodel_apparent_sky]:
                    misc.rename_skymodel_patches(model)

        # For the bright-source true-sky model, duplicate any selections made above to the
        # apparent-sky model
        bright_source_skymodel = skymodel_true_sky.copy()
        source_names = bright_source_skymodel.getColValues('Name').tolist()
        bright_source_names = bright_source_skymodel_apparent_sky.getColValues('Name').tolist()
        matching_ind = []
        for i, sn in enumerate(bright_source_names):
            matching_ind.append(source_names.index(sn))
        bright_source_skymodel.select(np.array(matching_ind))

        # Transfer patches to the bright-source model. At this point, it is now the
        # model of calibrator sources only, so copy and save it and write it out for
        # later use
        if regroup:
            # Transfer from the apparent-flux sky model (regrouped above)
            misc.transfer_patches(bright_source_skymodel_apparent_sky, bright_source_skymodel)
        else:
            # Transfer from the true-flux sky model
            patch_dict = skymodel_true_sky.getPatchPositions()
            misc.transfer_patches(skymodel_true_sky, bright_source_skymodel,
                                  patch_dict=patch_dict)
        bright_source_skymodel.write(self.calibrators_only_skymodel_file, clobber=True)
        self.calibrators_only_skymodel = bright_source_skymodel.copy()

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
        calibration_skymodel.write(self.calibration_skymodel_file, clobber=True)
        self.calibration_skymodel = calibration_skymodel
        if len(bright_source_skymodel) > 0:
            bright_source_skymodel.write(self.bright_source_skymodel_file, clobber=True)
        self.bright_source_skymodel = bright_source_skymodel

        # Save the final target flux
        self.target_flux = target_flux

    def update_skymodels(self, index, regroup, target_flux=None, target_number=None,
                         calibrator_max_dist_deg=None, combine_current_and_intial=False):
        """
        Updates the source and calibration sky models from the output sector sky model(s)

        Parameters
        ----------
        index : int
            Iteration index (counts starting from 1)
        regroup : bool
            Regroup sky model. This parameter is not used for the first cycle, as its
            value is taken from the parset. For later cycles, it controls whether the
            sky models that come from imaging are to be regrouped into calibration
            patches. In almost all cases, regrouping should be done. The exception is
            when using small imaging sectors when the sources in each sector should be
            grouped into a single patch together.
        target_flux : float, optional
            Target flux in Jy for grouping
        target_number : int, optional
            Target number of patches for grouping
        calibrator_max_dist_deg : float, optional
            Maximum distance in degrees from phase center for grouping
        combine_current_and_intial : bool, optional
            If True, combine the initial and current sky models (needed for the final
            calibration in order to include potential outlier sources)
        """
        # Except for the first iteration, use the results of the previous iteration to
        # update the sky models, etc.
        if index == 1:
            # For the first iteration, use either the input sky model, the model
            # generated by the intial imaging operation, or the one requested for download
            #
            # Note: the flags for generating or downloading the sky model are checked
            # and adjusted, if needed, during the reading of the parset
            moc = None  # default to no multi-order coverage map (which is, for now, LoTSS only)
            if self.parset['generate_initial_skymodel']:
                self.parset['input_skymodel'] = self.full_field_sector.image_skymodel_file_true_sky
                self.parset['apparent_skymodel'] = self.full_field_sector.image_skymodel_file_apparent_sky
            elif self.parset['download_initial_skymodel']:
                catalog = self.parset['download_initial_skymodel_server'].lower()
                self.parset['input_skymodel'] = os.path.join(self.working_dir, 'skymodels',
                                                             f'initial_skymodel_{catalog}.txt')
                self.parset['apparent_skymodel'] = None
                misc.download_skymodel(self.ra, self.dec,
                                       skymodel_path=self.parset['input_skymodel'],
                                       radius=self.parset['download_initial_skymodel_radius'],
                                       source=self.parset['download_initial_skymodel_server'],
                                       overwrite=self.parset['download_overwrite_skymodel'])
                if catalog == 'lotss':
                    moc = os.path.join(self.working_dir, 'skymodels', 'dr2-moc.moc')

            # Plot the field overview showing the initial sky-model coverage
            self.log.info('Plotting field overview with initial sky-model coverage...')
            self.plot_overview('initial_field_overview.png', show_initial_coverage=True,
                               moc=moc)

            self.make_skymodels(self.parset['input_skymodel'],
                                skymodel_apparent_sky=self.parset['apparent_skymodel'],
                                regroup=self.parset['regroup_input_skymodel'],
                                target_flux=target_flux, target_number=target_number,
                                find_sources=True, calibrator_max_dist_deg=calibrator_max_dist_deg,
                                index=index)
        else:
            # Use the imaging sector sky models from the previous iteration to update
            # the master sky model
            self.log.info('Updating sky model...')
            sector_skymodels_apparent_sky = [sector.image_skymodel_file_apparent_sky for
                                             sector in self.imaging_sectors]
            sector_skymodels_true_sky = [sector.image_skymodel_file_true_sky for
                                         sector in self.imaging_sectors]
            sector_names = [sector.name for sector in self.imaging_sectors]

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

            # Concatenate the starting sky model with the new one. This step needs to be
            # done if, e.g., this is the final cycle and sources outside of imaged areas
            # must be subtracted, since we have to go back to the original input MS files
            # for which no subtraction has been done) or if all sources (and not only the
            # imaged sources) are to be used in calibration
            if combine_current_and_intial or not self.imaged_sources_only:
                # Load starting sky model and regroup to one patch per entry to ensure
                # any existing patches are removed (otherwise they may propagate to
                # the DDE direction determination, leading to unexpected results)
                skymodel_true_sky_start = lsmtool.load(self.parset['input_skymodel'])
                skymodel_true_sky_start.group('every')

                # Concatenate by position. Any entries in the initial sky model that match
                # to one or more entires in the new one will be removed. A fairly large
                # matching radius is used to favor entries in the new model over those in
                # the initial one (i.e., ones from the initial model are only included if
                # they are far from any in the new model and thus not likely to be
                # duplicates)
                matching_radius_deg = 30.0 / 3600.0  # => 30 arcsec
                skymodel_true_sky.concatenate(skymodel_true_sky_start, matchBy='position',
                                              radius=matching_radius_deg, keep='from1')
                skymodel_true_sky.setPatchPositions()
                skymodel_apparent_sky = None

            # Use concatenated sky models to make new calibration model (we set find_sources
            # to False to preserve the source patches defined in the image operation by PyBDSF)
            self.make_skymodels(skymodel_true_sky, skymodel_apparent_sky=skymodel_apparent_sky,
                                regroup=regroup, find_sources=False, target_flux=target_flux,
                                target_number=target_number, calibrator_max_dist_deg=calibrator_max_dist_deg,
                                index=index)

        # Save the number of calibrators and their names, positions, and flux
        # densities (in Jy) for use in the calibration and imaging operations
        self.calibrator_patch_names = self.calibrators_only_skymodel.getPatchNames().tolist()
        self.calibrator_fluxes = self.calibrators_only_skymodel.getColValues('I', aggregate='sum').tolist()
        self.calibrator_positions = self.calibrators_only_skymodel.getPatchPositions()
        self.num_patches = len(self.calibrator_patch_names)
        suffix = 'es' if self.num_patches > 1 else ''
        self.log.info('Using {0} calibration patch{1}'.format(self.num_patches, suffix))

        # Plot an overview of the field for this cycle, showing the calibration facets
        # (patches)
        self.log.info('Plotting field overview with calibration patches...')
        if index == 1 or combine_current_and_intial:
            # Check the sky model bounds, as they may differ from the sector ones
            check_skymodel_bounds = True
        else:
            # Sky model bounds will always match the sector ones
            check_skymodel_bounds = False
        self.plot_overview(f'field_overview_{index}.png', show_calibration_patches=True,
                           check_skymodel_bounds=check_skymodel_bounds)

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

        # Make predict sectors containing all calibration sources. These sectors are
        # not imaged; they are only used in prediction for direction-independent solves
        self.define_predict_sectors(index)

        # Make non-calibrator-source sectors containing non-calibrator sources
        # that may be subtracted before calibration. These sectors are not
        # imaged
        self.define_non_calibrator_source_sectors(index)

        # Finally, make a list containing all sectors
        self.sectors = (self.imaging_sectors + self.outlier_sectors +
                        self.bright_source_sectors + self.predict_sectors +
                        self.non_calibrator_source_sectors)
        self.nsectors = len(self.sectors)

    def remove_skymodels(self):
        """
        Remove sky models to minimize memory usage
        """
        self.calibration_skymodel = None
        self.source_skymodel = None
        self.calibrators_only_skymodel = None
        for sector in self.sectors:
            sector.calibration_skymodel = None
            sector.predict_skymodel = None
            sector.field.source_skymodel = None

    def get_source_distances(self, source_dict: Dict[str, List[float]]):
        """
        Returns source distances in degrees from the phase center

        Parameters
        ----------
        source_dict : dict
            Dict of source patch names and coordinates in degrees
            (e.g., {'name': [RA, Dec]})

        Returns
        -------
        names : numpy array
            Array of source names
        distances : numpy array
            Array of distances from the phase center in degrees
        """
        phase_center_coord = SkyCoord(ra=self.ra*u.degree, dec=self.dec*u.degree)
        names = []
        source_ra = []
        source_dec = []
        for name, coord in source_dict.items():
            names.append(name)
            source_ra.append(coord[0])
            source_dec.append(coord[1])
        source_coord = SkyCoord(ra=source_ra, dec=source_dec)
        separation = phase_center_coord.separation(source_coord)
        distances = [sep.value for sep in separation]
        return np.array(names), np.array(distances)

    def get_calibration_radius(self):
        """
        Returns the radius in degrees that encloses all calibrators
        """
        _, separation = self.get_source_distances(self.calibrator_positions)
        return np.max(separation)

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
            suffix = 's' if len(self.imaging_sectors) > 1 else ''
            self.log.info('Using {0} user-defined imaging sector{1}'.format(len(self.imaging_sectors), suffix))
            self.uses_sector_grid = False
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
                image_width_ra = self.fwhm_ra_deg * 1.7
            else:
                image_width_ra = self.parset['imaging_specific']['grid_width_ra_deg']
            if self.parset['imaging_specific']['grid_width_dec_deg'] is None:
                image_width_dec = self.fwhm_dec_deg * 1.7
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
                center_x, center_y = misc.radec2xy(self.wcs, [image_ra], [image_dec])
                x = np.array([center_x])
                y = np.array([center_y])
            else:
                # Make the grid
                width_ra = image_width_ra / nsectors_ra
                width_dec = image_width_dec / nsectors_dec
                width_x = width_ra / abs(self.wcs.wcs.cdelt[0])
                width_y = width_dec / abs(self.wcs.wcs.cdelt[1])
                center_x, center_y = misc.radec2xy(self.wcs, [image_ra], [image_dec])
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
                    ra, dec = misc.xy2radec(self.wcs, x[j, i], y[j, i])
                    self.imaging_sectors.append(Sector(name, ra, dec, width_ra, width_dec, self))
                    n += 1
            if len(self.imaging_sectors) == 1:
                self.log.info('Using 1 imaging sector')
            else:
                self.log.info('Using {0} imaging sectors ({1} in RA, {2} in Dec)'.format(
                              len(self.imaging_sectors), nsectors_ra, nsectors_dec))
            self.uses_sector_grid = True

        # Compute bounding box for all imaging sectors and store as a
        # a semi-colon-separated list of [maxRA; minDec; minRA; maxDec] (we use semi-
        # colons as otherwise the workflow parset parser will split the list). Also
        # store the midpoint as [midRA; midDec].
        # Note: this is just once, rather than each time the sector borders are
        # adjusted, so that the image sizes do not change with iteration (so
        # mask images from previous iterations may be used)
        all_sectors = MultiPolygon([sector.poly_padded for sector in self.imaging_sectors])
        self.sector_bounds_xy = all_sectors.bounds
        maxRA, minDec = misc.xy2radec(self.wcs, self.sector_bounds_xy[0], self.sector_bounds_xy[1])
        minRA, maxDec = misc.xy2radec(self.wcs, self.sector_bounds_xy[2], self.sector_bounds_xy[3])
        midRA, midDec = misc.xy2radec(self.wcs, (self.sector_bounds_xy[0]+self.sector_bounds_xy[2])/2.0,
                                      (self.sector_bounds_xy[1]+self.sector_bounds_xy[3])/2.0)
        self.sector_bounds_width_ra = abs((self.sector_bounds_xy[0] - self.sector_bounds_xy[2]) *
                                          self.wcs.wcs.cdelt[0])
        self.sector_bounds_width_dec = abs((self.sector_bounds_xy[3] - self.sector_bounds_xy[1]) *
                                           self.wcs.wcs.cdelt[1])
        self.sector_bounds_mid_ra = midRA
        self.sector_bounds_mid_dec = midDec
        self.sector_bounds_deg = '[{0:.6f};{1:.6f};{2:.6f};{3:.6f}]'.format(maxRA, minDec,
                                                                            minRA, maxDec)
        self.sector_bounds_mid_deg = '[{0:.6f};{1:.6f}]'.format(midRA, midDec)

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

    def define_non_calibrator_source_sectors(self, index):
        """
        Defines the non-calibrator source sectors

        These sectors are defined if peeling of non-calibrator sources is
        explicitly enabled or if the antenna is LBA (where it's always
        needed)

        Parameters
        ----------
        index : int
            Iteration index
        """
        self.non_calibrator_source_sectors = []
        if self.peel_non_calibrator_sources or self.antenna == 'LBA':
            non_calibrator_skymodel = self.make_non_calibrator_skymodel()
            nsources = len(non_calibrator_skymodel)
            if nsources > 0:
                # Choose number of sectors to be the no more than ten, but don't allow
                # fewer than 100 sources per sector if possible
                nnodes = max(min(10, round(nsources/100)), 1)  # TODO: tune to number of available nodes and/or memory?
                for i in range(nnodes):
                    non_calibrator_source_sector = Sector('non_calibrator_source_{0}'.format(i+1), self.ra, self.dec, 1.0, 1.0, self)
                    non_calibrator_source_sector.is_predict = True
                    non_calibrator_source_sector.predict_skymodel = non_calibrator_skymodel.copy()
                    startind = i * int(nsources/nnodes)
                    if i == nnodes-1:
                        endind = nsources
                    else:
                        endind = startind + int(nsources/nnodes)
                    non_calibrator_source_sector.predict_skymodel.select(np.array(list(range(startind, endind))))
                    non_calibrator_source_sector.make_skymodel(index)
                    self.non_calibrator_source_sectors.append(non_calibrator_source_sector)

    def define_predict_sectors(self, index):
        """
        Defines the predict sectors

        Parameters
        ----------
        index : int
            Iteration index
        """
        self.predict_sectors = []
        predict_skymodel = self.calibration_skymodel
        nsources = len(predict_skymodel)
        if nsources > 0:
            # Choose number of sectors to be the no more than ten, but don't allow
            # fewer than 100 sources per sector if possible
            nnodes = max(min(10, round(nsources/100)), 1)  # TODO: tune to number of available nodes and/or memory?
            for i in range(nnodes):
                predict_sector = Sector('predict_{0}'.format(i+1), self.ra, self.dec, 1.0, 1.0, self)
                predict_sector.is_predict = True
                predict_sector.predict_skymodel = predict_skymodel.copy()
                startind = i * int(nsources/nnodes)
                if i == nnodes-1:
                    endind = nsources
                else:
                    endind = startind + int(nsources/nnodes)
                predict_sector.predict_skymodel.select(np.array(list(range(startind, endind))))
                predict_sector.make_skymodel(index)
                self.predict_sectors.append(predict_sector)

    def define_full_field_sector(self, radius=None):
        """
        Defines the full-field imaging sector, used for generation of the initial
        sky model

        Parameters
        ----------
        radius : float, optional
            Radius in degrees of region to image. If None, an area corresponding
            to 2*FWHM is used
        """
        if radius is None:
            width_ra = self.fwhm_ra_deg * 2
            width_dec = self.fwhm_dec_deg * 2
        else:
            width_ra = radius * 2
            width_dec = radius * 2
        self.full_field_sector = Sector('full_field', self.ra, self.dec,
                                        width_ra, width_dec, self)

        # Make sector region and vertices files
        self.full_field_sector.make_vertices_file()
        self.full_field_sector.make_region_file(os.path.join(self.working_dir, 'regions',
                                                f'{self.full_field_sector.name}_region_ds9.reg'))

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
        x, y = misc.radec2xy(self.wcs, RA, Dec)
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

        Note: this adjustment is only done when there are multiple sectors in a
        grid, since its purpose is to ensure that sources don't get split
        between two neighboring sectors
        """
        if len(self.imaging_sectors) > 1 and self.uses_sector_grid:
            self.log.info('Adusting sector boundaries to avoid sources...')
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

    def make_non_calibrator_skymodel(self):
        """
        Make a sky model of any non-calibrator sources

        Since the peeling of non-calibrator sources uses the calibration
        solutions from the previous cycle (if any), the calibration patches from
        that cycle are applied to the current sky model to ensure agreement
        between the sky model patches and the calibration patches.

        Note: if a previous cycle was not done (and therefore a model from it
        does not exist), then either peeling will be done without using
        calibration solutions (and therefore the patches are ignored) or a
        solutions file and sky model have been provided by the user, in which
        case the patches in the model and solutions must already agree with each
        other.
        """
        non_calibrator_skymodel = self.calibration_skymodel.copy()

        # Transfer the patches from the previous sky model (if any) to the
        # current one
        if self.calibrators_only_skymodel_file_prev_cycle is not None:
            calibrators_only_skymodel = lsmtool.load(self.calibrators_only_skymodel_file_prev_cycle)
            calibrator_names = calibrators_only_skymodel.getPatchNames()
            patch_dict = calibrators_only_skymodel.getPatchPositions()
            non_calibrator_skymodel.transfer(calibrators_only_skymodel,
                                             matchBy='position', radius='30 arcsec')
            non_calibrator_skymodel.setPatchPositions(patchDict=patch_dict)
            non_calibrator_skymodel.group('voronoi', patchNames=calibrator_names)

        # Remove the calibrator (bright) sources
        remove_source_names = self.bright_source_skymodel.getColValues('Name').tolist()
        all_source_names = non_calibrator_skymodel.getColValues('Name').tolist()
        keep_ind = np.array([all_source_names.index(sn) for sn in all_source_names
                             if sn not in remove_source_names])
        non_calibrator_skymodel.select(keep_ind, force=True)
        return non_calibrator_skymodel

    def makeWCS(self):
        """
        Makes simple WCS object
        """
        self.wcs_pixel_scale = 10.0 / 3600.0  # degrees/pixel (= 10"/pixel)
        self.wcs = misc.make_wcs(self.ra, self.dec, self.wcs_pixel_scale)

    def scan_h5parms(self):
        """
        Scans the calibration h5parms

        The basic structure is checked for correctness and for the presence of
        amplitude solutions (which may require different processing steps).
        """
        if self.h5parm_filename is not None:
            solutions = h5parm(self.h5parm_filename)
            if 'sol000' not in solutions.getSolsetNames():
                raise ValueError('The direction-dependent solutions file "{0}" must '
                                 'have the solutions stored in the sol000 '
                                 'solset.'.format(self.h5parm_filename))
            solset = solutions.getSolset('sol000')
            if 'phase000' not in solset.getSoltabNames():
                raise ValueError('The direction-dependent solutions file "{0}" must '
                                 'have a phase000 soltab.'.format(self.h5parm_filename))
            if 'amplitude000' in solset.getSoltabNames():
                self.apply_amplitudes = True
            else:
                self.apply_amplitudes = False
        else:
            self.apply_amplitudes = False

        if self.fulljones_h5parm_filename is not None:
            self.apply_fulljones = True
            solutions = h5parm(self.fulljones_h5parm_filename)
            if 'sol000' not in solutions.getSolsetNames():
                raise ValueError('The full-Jones solution file "{0}" must have '
                                 'the solutions stored in the sol000 '
                                 'solset.'.format(self.fulljones_h5parm_filename))
            solset = solutions.getSolset('sol000')
            if ('phase000' not in solset.getSoltabNames() or
                    'amplitude000' not in solset.getSoltabNames()):
                raise ValueError('The full-Jones solution file "{0}" must have both '
                                 'a phase000 soltab and a amplitude000 '
                                 'soltab.'.format(self.fulljones_h5parm_filename))
        else:
            self.apply_fulljones = False

    def check_selfcal_progress(self):
        """
        Checks whether selfcal has converged or diverged by comparing the current
        image noise to that of the previous cycle. A check is also done on the
        absolute value of the noise.

        Convergence is determined by comparing the noise and dynamic range ratios
        to self.convergence_ratio, which is the minimum ratio of the current noise
        to the previous noise above which selfcal is considered to have
        converged (must be in the range 0.5 -- 2). E.g., self.convergence_ratio
        = 0.95 means that the image noise must decrease by ~ 5% or more from the
        previous cycle for selfcal to be considered as not yet converged. The same
        is true for the dynamic range but reversed (the dynamic range must increase
        by ~ 5% or more from the previous cycle for selfcal to be considered as
        not yet converged).

        Divergence is determined by comparing the noise ratio to
        self.divergence_ratio, which is the minimum ratio of the current noise
        to the previous noise above which selfcal is considered to have diverged
        (must be >= 1). E.g., divergence_ratio = 1.1 means that, if image noise
        worsens by ~ 10% or more from the previous cycle, selfcal is considered
        to have diverged.

        Failure is determined by comparing the absolute value of the noise in
        the current cycle with the theoretical noise. If the ratio of the current
        median noise to the theoretical one is greater than failure_ratio, selfcal
        is considered to have failed.

        Returns
        -------
        selfcal_state : namedtuple
            The selfcal state, with the following elements:
                selfcal_state.converged - True if selfcal has converged in all
                                          sectors
                selfcal_state.diverged - True if selfcal has diverged in one or
                                         more sectors
                selfcal_state.failed - True if selfcal has failed in one or
                                       more sectors
        """
        convergence_ratio = self.convergence_ratio
        divergence_ratio = self.divergence_ratio
        failure_ratio = self.failure_ratio
        SelfcalState = namedtuple('SelfcalState', ['converged', 'diverged', 'failed'])

        # Check that convergence and divergence limits are sensible
        if convergence_ratio > 2.0:
            self.log.warning('The convergence ratio is set to {} but must be <= 2. '
                             'Using 2.0 instead'.format(convergence_ratio))
            convergence_ratio = 2.0
        if convergence_ratio < 0.5:
            self.log.warning('The convergence ratio is set to {} but must be >= 0.5. '
                             'Using 0.5 instead'.format(convergence_ratio))
            convergence_ratio = 0.5
        if divergence_ratio < 1.0:
            self.log.warning('The divergence ratio is set to {} but must be >= 1. '
                             'Using 1.0 instead'.format(divergence_ratio))
            divergence_ratio = 1.0
        if failure_ratio < 1.0:
            self.log.warning('The failure ratio is set to {} but must be >= 1. '
                             'Using 1.0 instead'.format(failure_ratio))
            failure_ratio = 1.0

        if (not hasattr(self, 'imaging_sectors') or
                not self.imaging_sectors or
                len(self.imaging_sectors[0].diagnostics) <= 1):
            # Either no imaging sectors or no previous iteration, so report not yet
            # converged, diverged, or failed
            return SelfcalState(False, False, False)

        # Get noise, dynamic range, and number of sources from previous and current
        # images of each sector
        converged = []
        diverged = []
        failed = []
        for sector in self.imaging_sectors:
            rmspre = sector.diagnostics[-2]['median_rms_flat_noise']
            rmspost = sector.diagnostics[-1]['median_rms_flat_noise']
            rmsideal = sector.diagnostics[-1]['theoretical_rms']
            self.log.info('Ratio of current median image noise (non-PB-corrected) to previous image '
                          'noise for {0} = {1:.2f}'.format(sector.name, rmspost/rmspre))
            self.log.info('Ratio of current median image noise (non-PB-corrected) to theorectical '
                          'minimum image noise for {0} = {1:.2f}'.format(sector.name, rmspost/rmsideal))
            dynrpre = sector.diagnostics[-2]['dynamic_range_global_flat_noise']
            dynrpost = sector.diagnostics[-1]['dynamic_range_global_flat_noise']
            self.log.info('Ratio of current image dynamic range (non-PB-corrected) to previous image '
                          'dynamic range for {0} = {1:.2f}'.format(sector.name, dynrpost/dynrpre))
            nsrcpre = sector.diagnostics[-2]['nsources']
            nsrcpost = sector.diagnostics[-1]['nsources']
            self.log.info('Ratio of current number of sources to previous number '
                          'of sources for {0} = {1:.2f}'.format(sector.name, nsrcpost/nsrcpre))
            if (rmspost / rmspre < convergence_ratio or
                    dynrpost / dynrpre > 1 / convergence_ratio or
                    nsrcpost / nsrcpre > 1 / convergence_ratio):
                # Report not converged (and not diverged)
                converged.append(False)
                diverged.append(False)
            elif rmspost / rmspre > divergence_ratio:
                # Report diverged (and not converged)
                converged.append(False)
                diverged.append(True)
            else:
                # Report converged (and not diverged)
                converged.append(True)
                diverged.append(False)
            if rmspost > self.failure_ratio * rmsideal:
                failed.append(True)
            else:
                failed.append(False)

        if any(failed):
            # Report failed
            return SelfcalState(False, False, True)
        elif any(diverged):
            # Report diverged
            return SelfcalState(False, True, False)
        elif all(converged):
            # Report converged
            return SelfcalState(True, False, False)
        else:
            # Report not converged, not diverged, and not failed
            return SelfcalState(False, False, False)

    def update(self, step_dict, index, final=False):
        """
        Updates parameters, sky models, etc. for current step

        Parameters
        ----------
        step_dict : dict
            Dict of parameter values for given iteration
        index : int
            Index of iteration
        final : bool, optional
            If True, process as the final pass (needed for correct processing of
            the sky models)
        """
        # If this is set as a final pass, set some peeling/calibration options to
        # the defaults
        if final:
            self.imaged_sources_only = False

        # Update field and sector dicts with the parameters for this iteration
        self.__dict__.update(step_dict)
        for sector in self.imaging_sectors:
            sector.__dict__.update(step_dict)

        # Update the sky models
        if step_dict['regroup_model']:
            # If regrouping is to be done, we adjust the target flux used for calibrator
            # selection by the ratio of (LOFAR / true) fluxes determined in the image
            # operation of the previous selfcal cycle. This adjustment is only done if the
            # fractional change is significant (as measured by the standard deviation in
            # the ratio)
            target_flux = step_dict['target_flux']
            target_number = step_dict['max_directions']
            calibrator_max_dist_deg = step_dict['max_distance']
            if self.lofar_to_true_flux_ratio <= 0:
                self.lofar_to_true_flux_ratio = 1.0  # disable adjustment
            if self.lofar_to_true_flux_ratio <= 1:
                fractional_change = 1 / self.lofar_to_true_flux_ratio - 1
            else:
                fractional_change = self.lofar_to_true_flux_ratio - 1
            if fractional_change > self.lofar_to_true_flux_std:
                target_flux *= self.lofar_to_true_flux_ratio
                self.log.info('Adjusting the target flux for calibrator selection '
                              'from {0:.2f} Jy to {1:.2f} Jy to account for the offset found '
                              'in the global flux scale'.format(step_dict['target_flux'],
                                                                target_flux))
        else:
            target_flux = None
            target_number = None
            calibrator_max_dist_deg = None

        self.update_skymodels(index, step_dict['regroup_model'],
                              target_flux=target_flux, target_number=target_number,
                              calibrator_max_dist_deg=calibrator_max_dist_deg,
                              combine_current_and_intial=final)
        self.remove_skymodels()  # clean up sky models to reduce memory usage

        # Always try to peel non-calibrator sources if LBA (we check below whether
        # or not there are sources that need to be peeled)
        if self.antenna == 'LBA':
            self.peel_non_calibrator_sources = True

        # Check whether any sources need to be peeled
        nr_outlier_sectors = len(self.outlier_sectors)
        nr_imaging_sectors = len(self.imaging_sectors)
        nr_bright_source_sectors = len(self.bright_source_sectors)
        nr_non_calibrator_source_sectors = len(self.non_calibrator_source_sectors)
        if nr_bright_source_sectors == 0:
            self.peel_bright_sources = False
        if nr_outlier_sectors == 0:
            self.peel_outliers = False
            self.imaged_sources_only = True  # No outliers means all sources are imaged
        if nr_non_calibrator_source_sectors == 0:
            self.peel_non_calibrator_sources = False

        # Determine whether the main predict operation is needed or not (note:
        # this does not apply to direction-independent or non-calibrator predict
        # operations). It's needed when:
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

    def get_matplotlib_patch(self, wcs=None):
        """
        Returns a matplotlib patch for the field primary-beam FOV polygon

        Parameters
        ----------
        wcs : WCS object, optional
            WCS object defining (RA, Dec) <-> (x, y) transformation. If not given,
            the field's transformation is used

        Returns
        -------
        patch : matplotlib patch object
            The patch for the field polygon
        """
        if wcs is None:
            wcs = self.wcs
            wcs_pixel_scale = self.wcs_pixel_scale  # degrees/pixel
        else:
            # Take the scale (in degrees/pixel) as the average of those of the
            # two axes
            wcs_pixel_scale = (wcs.proj_plane_pixel_scales()[0].value +
                               wcs.proj_plane_pixel_scales()[1].value) / 2
        x, y = misc.radec2xy(wcs, self.ra, self.dec)
        patch = Ellipse((x, y), width=self.fwhm_ra_deg/wcs_pixel_scale,
                        height=self.fwhm_dec_deg/wcs_pixel_scale,
                        edgecolor='k', facecolor='lightgray', linestyle=':',
                        label='Pointing FWHM', linewidth=2, alpha=0.5)

        return patch

    def plot_overview(self, output_filename, show_initial_coverage=False,
                      show_calibration_patches=False, moc=None,
                      check_skymodel_bounds=False):
        """
        Plots an overview of the field, with optional intial sky-model coverage
        and calibration facets shown

        Parameters
        ----------
        output_filename : str
            Base filename of ouput file, to be output to 'dir_working/plots/'
        show_initial_coverage : bool, optional
            If True, plot the intial sky-model coverage. The plot will be centered
            on the center of the field. If False, the plot will be centered on the
            center of the imaging region(s)
        show_calibration_patches : bool, optional
            If True, plot the calibration patches
        moc : str or None, optional
            If not None, the multi-order coverage map to plot alongside the usual
            quantiies. Only shown if show_initial_coverage = True
        check_skymodel_bounds : bool, optional
            If True (and show_calibration_patches is True), the bounds from the
            calibration sky model are checked when calculating the faceting bounds
        """
        size_ra = self.sector_bounds_width_ra * u.deg
        size_dec = self.sector_bounds_width_dec * u.deg
        if self.parset['generate_initial_skymodel']:
            initial_skymodel_radius = max(self.full_field_sector.width_ra,
                                          self.full_field_sector.width_dec) / 2
        elif self.parset['download_initial_skymodel']:
            initial_skymodel_radius = self.parset['download_initial_skymodel_radius']
        else:
            # User-supplied sky model (unknown coverage)
            initial_skymodel_radius = 0
        if show_initial_coverage:
            size_skymodel = initial_skymodel_radius * u.deg
        else:
            size_skymodel = 0 * u.deg

        # Find the minimum size in degrees for the plot (can be overridden by
        # a MOC if given)
        fake_size = size_ra if size_ra > size_dec else size_dec
        fake_size = size_skymodel if size_skymodel > fake_size else fake_size

        # Make the figure and subplot with the appropriate WCS projection
        fig = figure(figsize=(8, 8), dpi=300)
        if moc is not None:
            pmoc = mocpy.MOC.from_fits(moc)
            wcs = mocpy.WCS(fig, fov=fake_size*2,
                            center=SkyCoord(self.ra*u.deg, self.dec*u.deg, frame='fk5')).w
        else:
            wcs = self.wcs
        ax = fig.add_subplot(111, projection=wcs)

        # Plot the MOC and initial sky model area
        if show_initial_coverage:
            if moc is not None:
                pmoc.fill(ax=ax, wcs=wcs, linewidth=2, edgecolor='b', facecolor='lightblue',
                          label='Skymodel MOC', alpha=0.5)

            # If sky model was generated or downloaded, indicate the region out
            # to which the initial sky model extends, centered on the field
            if initial_skymodel_radius > 0:
                # Nonzero radius implies model was either generated or downloaded (see
                # above)
                if self.parset['generate_initial_skymodel']:
                    skymodel_region = self.full_field_sector.get_matplotlib_patch(wcs=wcs)
                    skymodel_region.set(edgecolor='b', facecolor='lightblue', alpha=0.5,
                                        label='Initial sky model coverage')
                elif self.parset['download_initial_skymodel']:
                    skymodel_region = SphericalCircle((self.ra*u.deg, self.dec*u.deg), size_skymodel,
                                                      transform=ax.get_transform('fk5'),
                                                      label='Initial sky model query cone', edgecolor='b',
                                                      facecolor='lightblue', linewidth=2, alpha=0.5)
                ax.add_patch(skymodel_region)

        # Plot the calibration patches (facets)
        if show_calibration_patches:
            # Find the faceting limits defined from the sky model
            if check_skymodel_bounds:
                if initial_skymodel_radius > 0:
                    skymodel_bounds_width_ra = skymodel_bounds_width_dec = initial_skymodel_radius * 2  # deg
                else:
                    # User-supplied sky model: estimate the size from the maximum distance
                    # of any patch from the phase center (plus 20% padding)
                    _, distances = self.get_source_distances(self.calibrator_positions)
                    skymodel_bounds_width_ra = skymodel_bounds_width_dec = 2 * 1.2 * np.max(distances)  # deg
            else:
                skymodel_bounds_width_ra = skymodel_bounds_width_dec = 0

            # Find the faceting limits defined from the sector bounds
            #
            # Note: we need the bounds for the unpadded sector polygons, so we do not
            # use self.sector_bounds_width_ra and self.sector_bounds_width_dec as they
            # were calculated for the padded polygons
            all_sectors = MultiPolygon([sector.poly for sector in self.imaging_sectors])
            bounds_xy = all_sectors.bounds  # pix
            sector_bounds_width_ra = abs((bounds_xy[0] - bounds_xy[2]) * wcs.wcs.cdelt[0])  # deg
            sector_bounds_width_dec = abs((bounds_xy[3] - bounds_xy[1]) * wcs.wcs.cdelt[1])  # deg

            facets = read_skymodel(self.calibration_skymodel_file,
                                   self.sector_bounds_mid_ra,
                                   self.sector_bounds_mid_dec,
                                   max(skymodel_bounds_width_ra, sector_bounds_width_ra),
                                   max(skymodel_bounds_width_dec, sector_bounds_width_dec))
            for i, facet in enumerate(facets):
                facet_patch = facet.get_matplotlib_patch(wcs=wcs)
                label = 'Calibration facets' if i == 0 else None  # first only to avoid multiple lines in legend
                facet_patch.set(edgecolor='b', facecolor='lightblue', alpha=0.5, label=label)
                ax.add_patch(facet_patch)
                x, y = misc.radec2xy(wcs, facet.ra, facet.dec)
                ax.annotate(facet.name, (x, y), va='center', ha='center', fontsize='small',
                            color='b')

        # Plot the imaging sectors
        for i, sector in enumerate(self.imaging_sectors):
            sector_patch = sector.get_matplotlib_patch(wcs=wcs)
            label = 'Imaging sectors' if i == 0 else None  # first only to avoid multiple lines in legend
            sector_patch.set(label=label)
            ax.add_patch(sector_patch)
            x, y = misc.radec2xy(wcs, sector.ra, sector.dec+sector.width_dec/2)  # center-top
            ax.annotate(sector.name, (x, y), va='bottom', ha='center', fontsize='large')

        # Plot the observation's FWHM and phase center
        if show_initial_coverage:
            # Plot the primary beam FWHM
            ax.add_patch(self.get_matplotlib_patch(wcs=wcs))

            # Plot the phase center
            ax.scatter(self.ra*u.deg, self.dec*u.deg, marker='s', color='k',
                       transform=ax.get_transform('fk5'), label='Phase center')

        # Set the minimum plot FoV by adding an invisible point and circle. The
        # final FoV will be set either by this circle or the MOC (if given)
        if show_initial_coverage:
            ra = self.ra*u.deg
            dec = self.dec*u.deg
        else:
            ra = self.sector_bounds_mid_ra*u.deg
            dec = self.sector_bounds_mid_dec*u.deg
        ax.scatter(ra, dec, color='none', transform=ax.get_transform('fk5'))
        fake_FoV_circle = SphericalCircle((ra, dec), fake_size/2,
                                          transform=ax.get_transform('fk5'),
                                          edgecolor='none', facecolor='none',
                                          linewidth=0)
        ax.add_patch(fake_FoV_circle)

        ax.set(xlabel='Right Ascension [J2000]', ylabel='Declination [J2000]')
        ax.legend(loc='upper left')
        ax.grid()
        fig.savefig(os.path.join(self.working_dir, 'plots', output_filename))
