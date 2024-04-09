"""
Definition of the Observation class that holds parameters for each measurement set
"""
import os
import logging
import casacore.tables as pt
import numpy as np
from rapthor.lib.cluster import get_fast_solve_intervals, get_slow_solve_intervals
from scipy.special import erf
from rapthor.lib import miscellaneous as misc
import copy


class Observation(object):
    """
    The Observation object contains various MS-related parameters

    Parameters
    ----------
    ms_filename : str
        Filename of the MS file
    starttime : float, optional
        The start time of the observation (in MJD seconds). If None, the start time
        is the start of the MS file
    endtime : float, optional
        The end time of the observation (in MJD seconds). If None, the end time
        is the end of the MS file
    """
    def __init__(self, ms_filename, starttime=None, endtime=None):
        self.ms_filename = str(ms_filename)
        self.ms_predict_di_filename = None
        self.name = os.path.basename(self.ms_filename)
        self.log = logging.getLogger('rapthor:{}'.format(self.name))
        self.starttime = starttime
        self.endtime = endtime
        self.parameters = {}
        self.scan_ms()

        # Define the infix for filenames
        if self.startsat_startofms and self.goesto_endofms:
            # Don't include starttime if observation covers full MS
            self.infix = ''
        else:
            # Include starttime to avoid naming conflicts
            self.infix = '.mjd{}'.format(int(self.starttime))

    def copy(self):
        """
        Returns a copy of the observation
        """
        # The logger's stream handlers are not copyable with deepcopy, so copy
        # them by hand:
        self.log = None
        obs_copy = copy.deepcopy(self)
        obs_copy.log = logging.getLogger('rapthor:{}'.format(self.name))
        self.log = logging.getLogger('rapthor:{}'.format(self.name))

        return obs_copy

    def scan_ms(self):
        """
        Scans input MS and stores info
        """
        # Get time info
        tab = pt.table(self.ms_filename, ack=False)
        if self.starttime is None:
            self.starttime = np.min(tab.getcol('TIME'))
        else:
            valid_times = np.where(tab.getcol('TIME') >= self.starttime)[0]
            if len(valid_times) == 0:
                raise ValueError('Start time of {0} is greater than the last time in the '
                                 'MS'.format(self.starttime))
            self.starttime = tab.getcol('TIME')[valid_times[0]]

        # DPPP takes ceil(startTimeParset - startTimeMS), so ensure that our start time is
        # slightly less than the true one (if it's slightly larger, DPPP sets the first
        # time to the next time, skipping the first time slot)
        self.starttime -= 0.1
        if self.starttime > np.min(tab.getcol('TIME')):
            self.startsat_startofms = False
        else:
            self.startsat_startofms = True
        if self.endtime is None:
            self.endtime = np.max(tab.getcol('TIME'))
        else:
            valid_times = np.where(tab.getcol('TIME') <= self.endtime)[0]
            if len(valid_times) == 0:
                raise ValueError('End time of {0} is less than the first time in the '
                                 'MS'.format(self.endtime))
            self.endtime = tab.getcol('TIME')[valid_times[-1]]
        if self.endtime < np.max(tab.getcol('TIME')):
            self.goesto_endofms = False
        else:
            self.goesto_endofms = True
        self.timepersample = tab.getcell('EXPOSURE', 0)
        self.numsamples = int(np.ceil((self.endtime - self.starttime) / self.timepersample))
        tab.close()

        # Get frequency info
        sw = pt.table(self.ms_filename+'::SPECTRAL_WINDOW', ack=False)
        self.referencefreq = sw.col('REF_FREQUENCY')[0]
        self.startfreq = np.min(sw.col('CHAN_FREQ')[0])
        self.endfreq = np.max(sw.col('CHAN_FREQ')[0])
        self.numchannels = sw.col('NUM_CHAN')[0]
        self.channelwidth = sw.col('CHAN_WIDTH')[0][0]
        sw.close()

        # Get pointing info
        obs = pt.table(self.ms_filename+'::FIELD', ack=False)
        self.ra = misc.normalize_ra(np.degrees(float(obs.col('REFERENCE_DIR')[0][0][0])))
        self.dec = misc.normalize_dec(np.degrees(float(obs.col('REFERENCE_DIR')[0][0][1])))
        obs.close()

        # Get station names and diameter
        ant = pt.table(self.ms_filename+'::ANTENNA', ack=False)
        self.stations = ant.col('NAME')[:]
        self.diam = float(ant.col('DISH_DIAMETER')[0])
        if 'HBA' in self.stations[0]:
            self.antenna = 'HBA'
        elif 'LBA' in self.stations[0]:
            self.antenna = 'LBA'
        else:
            self.log.warning('Antenna type not recognized (only LBA and HBA data '
                             'are supported at this time)')
        ant.close()

        # Find mean elevation and FOV
        el_values = pt.taql("SELECT mscal.azel1()[1] AS el from "
                            + self.ms_filename + " limit ::10000").getcol("el")
        self.mean_el_rad = np.mean(el_values)

    def set_calibration_parameters(self, parset, ndir, nobs, calibrator_fluxes,
                                   target_fast_timestep, target_slow_timestep_joint,
                                   target_slow_timestep_separate,
                                   target_fulljones_timestep, target_flux=None):
        """
        Sets the calibration parameters

        Parameters
        ----------
        parset : dict
            Parset with processing parameters
        ndir : int
            Number of calibration directions/patches
        nobs : int
            Number of observations in total
        calibrator_fluxes : list
            List of calibrator apparent flux densities in Jy
        target_fast_timestep : float
            Target solution interval for fast solves in sec
        target_slow_timestep_joint : float
            Target solution interval for joint slow solves in sec
        target_slow_timestep_separate : float
            Target solution interval for separate slow solves in sec
        target_fulljones_timestep : float
            Target solution interval for full-Jones solves in sec
        target_flux: float, optional
            Target calibrator flux in Jy. If None, the lowest calibrator flux density
            is used.
        """
        # Get the target solution intervals
        target_fast_freqstep = parset['calibration_specific']['fast_freqstep_hz']
        target_slow_freqstep = parset['calibration_specific']['slow_freqstep_hz']
        target_fulljones_freqstep = parset['calibration_specific']['fulljones_freqstep_hz']

        # Find solution intervals for fast-phase solve. The solve is split into time
        # chunks instead of frequency chunks, since continuous frequency coverage is
        # desirable to recover the expected smooth, TEC-like behavior (phase ~ nu^-1)
        solint_fast_timestep = max(1, int(round(target_fast_timestep / round(self.timepersample))))
        solint_fast_freqstep = max(1, self.get_nearest_freqstep(target_fast_freqstep / self.channelwidth))

        # Adjust the solution interval if needed to fit the fast solve into the
        # available memory and determine how many calibration chunks to make (to allow
        # parallel jobs)
        samplesperchunk, solint_fast_timestep = get_fast_solve_intervals(parset['cluster_specific'],
                                                                         self.numsamples, nobs,
                                                                         solint_fast_timestep,
                                                                         self.antenna, ndir)
        chunksize = samplesperchunk * self.timepersample
        mystarttime = self.starttime
        myendtime = self.endtime
        if (myendtime - mystarttime) > chunksize:
            # Divide up the total duration into chunks of chunksize or smaller
            nchunks = int(np.ceil(float(self.numsamples) * self.timepersample / chunksize))
        else:
            nchunks = 1
        starttimes = [mystarttime+(chunksize * i) for i in range(nchunks)]
        if starttimes[-1] >= myendtime:
            # Make sure the last start time does not equal or exceed the end time
            starttimes.pop(-1)
            nchunks -= 1
        self.ntimechunks = nchunks
        self.log.debug('Using {0} time chunk{1} for fast-phase '
                       'calibration'.format(self.ntimechunks, "s" if self.ntimechunks > 1 else ""))
        self.parameters['timechunk_filename'] = [self.ms_filename] * self.ntimechunks
        self.parameters['starttime'] = [misc.convert_mjd2mvt(t) for t in starttimes]
        self.parameters['ntimes'] = [samplesperchunk] * self.ntimechunks

        # Set last entry in ntimes list to extend to end of observation
        if self.goesto_endofms:
            self.parameters['ntimes'][-1] = 0
        else:
            self.parameters['ntimes'][-1] += int(self.numsamples - (samplesperchunk * self.ntimechunks))

        # Set the fast solve solution intervals
        self.parameters['solint_fast_timestep'] = [solint_fast_timestep] * self.ntimechunks
        self.parameters['solint_fast_freqstep'] = [solint_fast_freqstep] * self.ntimechunks

        # Find solution intervals for the gain solves. In contrast to the fast-phase
        # solve, the gain solves are split into frequency chunks, since continuous
        # frequency coverage is not needed
        solve_max_factor = int(parset['calibration_specific']['dd_interval_factor'])
        target_slow_timestep_joint = max(1, int(round(target_slow_timestep_joint * solve_max_factor / round(self.timepersample))))
        target_slow_timestep_separate = max(1, int(round(target_slow_timestep_separate * solve_max_factor / round(self.timepersample))))
        solint_slow_freqstep = max(1, self.get_nearest_freqstep(target_slow_freqstep / self.channelwidth))
        target_timestep_fulljones = max(1, int(round(target_fulljones_timestep / round(self.timepersample))))
        solint_freqstep_fulljones = max(1, self.get_nearest_freqstep(target_fulljones_freqstep / self.channelwidth))

        # Adjust the solution intervals if needed to fit the gain solves into the
        # available memory and determine how many calibration chunks to make (to allow
        # parallel jobs). For simplicity, we do this for all potential types of gain
        # solve, even if they are not all required by the current strategy
        target_timesteps = {'joint': target_slow_timestep_joint,
                            'separate': target_slow_timestep_separate,
                            'fulljones': target_timestep_fulljones}
        solint_freqsteps = {'joint': solint_slow_freqstep,
                            'separate': solint_slow_freqstep,
                            'fulljones': solint_freqstep_fulljones}
        for solve_type in ['joint', 'separate', 'fulljones']:
            solint_freqstep = solint_freqsteps[solve_type]
            target_timestep = target_timesteps[solve_type]
            if solve_type == 'fulljones':
                ndirections = 1
                freqchunk_filename = self.ms_predict_di_filename
            else:
                ndirections = ndir
                freqchunk_filename = self.ms_filename
            samplesperchunk, solint_timestep = get_slow_solve_intervals(parset['cluster_specific'],
                                                                        self.numchannels, nobs,
                                                                        solint_freqstep,
                                                                        target_timestep,
                                                                        self.antenna, ndirections)
            freqchunksize = samplesperchunk * self.channelwidth
            if (self.endfreq-self.startfreq) > freqchunksize:
                # Divide up the bandwidth into chunks of freqchunksize or smaller
                nfreqchunks = int(np.ceil(float(self.numchannels) * self.channelwidth / freqchunksize))
            else:
                nfreqchunks = 1
            setattr(self, f'nfreqchunks_{solve_type}', nfreqchunks)
            self.log.debug('Using {0} frequency chunk{1} for the {2} gain '
                           'calibration (if done)'.format(nfreqchunks, "s" if nfreqchunks > 1 else "", solve_type))
            self.parameters[f'freqchunk_filename_{solve_type}'] = [freqchunk_filename] * nfreqchunks
            self.parameters[f'startchan_{solve_type}'] = [samplesperchunk * i for i in range(nfreqchunks)]
            self.parameters[f'nchan_{solve_type}'] = [samplesperchunk] * nfreqchunks
            self.parameters[f'nchan_{solve_type}'][-1] = 0  # set last entry to extend until end
            self.parameters[f'slow_starttime_{solve_type}'] = [misc.convert_mjd2mvt(self.starttime)] * nfreqchunks
            self.parameters[f'slow_ntimes_{solve_type}'] = [self.numsamples] * nfreqchunks
            self.parameters[f'solint_slow_timestep_{solve_type}'] = [solint_timestep] * nfreqchunks
            self.parameters[f'solint_slow_freqstep_{solve_type}'] = [solint_freqstep] * nfreqchunks

        # Define the direction-dependent solution interval list for the fast and
        # slow solves (the full-Jones solve is direction-independent so is not included).
        # The list values are defined as the number of solutions that will be obtained for
        # each base solution interval, with one entry per direction
        input_solint_keys = {'slow_joint': 'solint_slow_timestep_joint',
                             'slow_separate': 'solint_slow_timestep_separate',
                             'fast': 'solint_fast_timestep'}
        if target_flux is None:
            target_flux = min(calibrator_fluxes)
        for solve_type in ['fast', 'slow_joint', 'slow_separate']:
            solint = self.parameters[input_solint_keys[solve_type]][0]  # number of time slots
            nchunks = len(self.parameters[input_solint_keys[solve_type]])  # number of time or frequency chunks

            if solve_max_factor > 1:
                # Find the initial estimate for the number of solutions, relative to that
                # for a source with a flux equal to the target flux and at most
                # solve_max_factor, with the brighter calibrators getting a larger number
                # and the fainter ones a smaller number (smaller numbers give longer
                # solution intervals)
                interval_factors = np.round(np.array(calibrator_fluxes) / target_flux)
                n_solutions = [min(solve_max_factor, max(1, int(factor))) for
                               factor in interval_factors]

                # Increase the base solution interval (passed to DDECal as solint). We use
                # solve_max_factor (instead of, e.g., max(n_solutions)) because the length
                # of the observation (if chunking was done) is tuned to this value (see
                # the field.chunk_observations() method)
                base_solint = solint * solve_max_factor

                # Calculate the final number per direction, making sure each is a divisor
                # of the new base solution interval. We choose the lower number that
                # satisfies this requirement, as it will result in a longer solution
                # interval (and therefore a higher SNR) and so is generally safer than
                # going the other way (towards low SNRs)
                solutions_per_direction = []
                for n_sols in n_solutions:
                    while base_solint % n_sols:
                        n_sols -= 1
                    solutions_per_direction.append(n_sols)
                self.parameters[f'solutions_per_direction_{solve_type}'] = [solutions_per_direction] * nchunks

                # Update the solution interval to the new one
                self.parameters[input_solint_keys[solve_type]] = [base_solint] * nchunks
            else:
                self.parameters[f'solutions_per_direction_{solve_type}'] = [[1] * len(calibrator_fluxes)] * nchunks

        # Set the number of segments to split the h5parm files into for screen fitting.
        # Try to split so that each file gets at least two solutions
        solint_slow_timestep = max(self.parameters['solint_slow_timestep_joint'][0],
                                   self.parameters['solint_slow_timestep_separate'][0])
        self.parameters['nsplit_fast'] = [max(1, int(self.numsamples / solint_fast_timestep / 2))]
        self.parameters['nsplit_slow'] = [max(1, int(self.numsamples / solint_slow_timestep / 2))]

        # Set the smoothnessreffrequency for the fast solves, if not set by the user
        fast_smoothnessreffrequency = parset['calibration_specific']['fast_smoothnessreffrequency']
        if fast_smoothnessreffrequency is None:
            if self.antenna == 'HBA':
                fast_smoothnessreffrequency = 144e6
            elif self.antenna == 'LBA':
                # Select a frequency at the midpoint of the frequency coverage of this observation
                fast_smoothnessreffrequency = (self.startfreq + self.endfreq) / 2.0
        self.parameters['fast_smoothnessreffrequency'] = [fast_smoothnessreffrequency] * self.ntimechunks

    def set_prediction_parameters(self, sector_name, patch_names, scratch_dir):
        """
        Sets the prediction parameters

        Parameters
        ----------
        sector_name : str
            Name of sector for which predict is to be done
        patch_names : list
            List of patch names to predict
        scratch_dir : str
            Scratch directory path
        """
        self.parameters['ms_filename'] = self.ms_filename

        # The filename of the sector's model data (from predict)
        root_filename = os.path.basename(self.ms_filename)
        ms_model_filename = '{0}{1}.{2}_modeldata'.format(root_filename, self.infix,
                                                          sector_name)
        self.parameters['ms_model_filename'] = ms_model_filename

        # The filename of the sector's data with all non-sector sources peeled off
        # and/or with the weights adjusted (i.e., the data used as input for the
        # imaging operation)
        self.ms_subtracted_filename = '{0}{1}.{2}'.format(root_filename, self.infix,
                                                          sector_name)
        self.parameters['ms_subtracted_filename'] = self.ms_subtracted_filename

        # The filename of the field data (after subtraction of outlier sources)
        self.ms_field = '{0}{1}_field'.format(root_filename, self.infix)

        # The filename of the model data for direction-independent calibration
        self.ms_predict_di = self.ms_subtracted_filename + '_di.ms'

        # The sky model patch names
        self.parameters['patch_names'] = patch_names

        # The start time and number of times (since an observation can be a part of its
        # associated MS file)
        self.parameters['predict_starttime'] = misc.convert_mjd2mvt(self.starttime)
        if self.goesto_endofms:
            self.parameters['predict_ntimes'] = 0
        else:
            self.parameters['predict_ntimes'] = self.numsamples

    def set_imaging_parameters(self, sector_name, cellsize_arcsec, max_peak_smearing, width_ra,
                               width_dec, solve_fast_timestep, solve_slow_freqstep,
                               use_screens):
        """
        Sets the imaging parameters

        Parameters
        ----------
        sector_name : str
            Name of sector for which predict is to be done
        cellsize_arcsec : float
            Pixel size in arcsec for imaging
        width_ra : float
            Width in RA of image in degrees
        width_dec : float
            Width in Dec of image in degrees
        solve_fast_timestep : float
            Solution interval in sec for fast solve
        solve_slow_freqstep : float
            Solution interval in Hz for slow solve
        use_screens : bool
            If True, use setup appropriate for screens
        """
        mean_freq_mhz = self.referencefreq / 1e6
        peak_smearing_rapthor = np.sqrt(1.0 - max_peak_smearing)
        chan_width_hz = self.channelwidth
        nchan = self.numchannels
        timestep_sec = self.timepersample

        # Set MS filenames for step that prepares the data for imaging
        root_filename = os.path.basename(self.ms_filename)
        ms_prep_filename = '{0}{1}.{2}.prep'.format(root_filename, self.infix,
                                                    sector_name)
        self.parameters['ms_prep_filename'] = ms_prep_filename

        # Get target time and frequency averaging steps.
        #
        # Note: We limit the averaging to be not more than 2 MHz and 120 s to
        # avoid extreme values for very small images. Also, due to a limitation
        # in Dysco, we make sure to have at least 2 time slots after averaging,
        # otherwise the output MS cannot be written with compression
        if self.numsamples == 1:
            raise RuntimeError('Only one time slot is availble for imaging, but at least '
                               'two are required. Please increase the fraction of data '
                               'processed with the selfcal_data_fraction parameter or supply a '
                               'measurement set with more time slots.')
        max_timewidth_sec = min(120, int(self.numsamples / 2) * timestep_sec)
        delta_theta_deg = max(width_ra, width_dec) / 2.0
        resolution_deg = 3.0 * cellsize_arcsec / 3600.0  # assume normal sampling of restoring beam
        target_timewidth_sec = min(max_timewidth_sec, self.get_target_timewidth(delta_theta_deg,
                                   resolution_deg, peak_smearing_rapthor))

        if use_screens:
            # Ensure we don't average more than the solve time step, as we want to
            # preserve the time resolution that matches that of the screens
            target_timewidth_sec = min(target_timewidth_sec, solve_fast_timestep)

        target_bandwidth_mhz = min(2.0, self.get_target_bandwidth(mean_freq_mhz,
                                   delta_theta_deg, resolution_deg, peak_smearing_rapthor))
        target_bandwidth_mhz = min(target_bandwidth_mhz, solve_slow_freqstep/1e6)
        self.log.debug('Target timewidth for imaging is {} s'.format(target_timewidth_sec))
        self.log.debug('Target bandwidth for imaging is {} MHz'.format(target_bandwidth_mhz))

        # Find averaging steps for above target values
        image_freqstep = max(1, min(int(round(target_bandwidth_mhz * 1e6 / chan_width_hz)), nchan))
        self.parameters['image_freqstep'] = self.get_nearest_freqstep(image_freqstep)
        self.parameters['image_timestep'] = max(1, int(round(target_timewidth_sec / timestep_sec)))
        self.log.debug('Using averaging steps of {0} channel{1} and {2} time slot{3} '
                       'for imaging'.format(self.parameters['image_freqstep'],
                                            "s" if self.parameters['image_freqstep'] > 1 else "",
                                            self.parameters['image_timestep'],
                                            "s" if self.parameters['image_timestep'] > 1 else ""))

    def get_nearest_freqstep(self, freqstep):
        """
        Gets the nearest frequency step to the target one

        Parameters
        ----------
        freqstep : int
            Target frequency step

        Returns
        -------
        optimum_step : int
            Optimum frequency step nearest to target step
        """
        # Generate a list of possible values for freqstep
        if not hasattr(self, 'freq_divisors'):
            tmp_divisors = []
            for step in range(self.numchannels, 0, -1):
                if (self.numchannels % step) == 0:
                    tmp_divisors.append(step)
            self.freq_divisors = np.array(tmp_divisors)

        # Find nearest
        idx = np.argmin(np.abs(self.freq_divisors - freqstep))

        return self.freq_divisors[idx]

    def get_target_timewidth(self, delta_theta, resolution, reduction_factor):
        """
        Returns the time width for given peak flux density reduction factor

        Parameters
        ----------
        delta_theta : float
            Distance from phase center
        resolution : float
            Resolution of restoring beam
        reduction_factor : float
            Ratio of pre-to-post averaging peak flux density

        Returns
        -------
        delta_time : float
            Time width in seconds for target reduction_factor

        """
        delta_time = np.sqrt((1.0 - reduction_factor) /
                             (1.22E-9 * (delta_theta / resolution)**2.0))

        return delta_time

    def get_bandwidth_smearing_factor(self, freq, delta_freq, delta_theta, resolution):
        """
        Returns peak flux density reduction factor due to bandwidth smearing

        Parameters
        ----------
        freq : float
            Frequency at which averaging will be done
        delta_freq : float
            Bandwidth over which averaging will be done
        delta_theta : float
            Distance from phase center
        resolution : float
            Resolution of restoring beam

        Returns
        -------
        reduction_facgtor : float
            Ratio of pre-to-post averaging peak flux density

        """
        beta = (delta_freq/freq) * (delta_theta/resolution)
        gamma = 2*(np.log(2)**0.5)
        reduction_factor = ((np.pi**0.5)/(gamma * beta)) * (erf(beta*gamma/2.0))

        return reduction_factor

    def get_target_bandwidth(self, freq, delta_theta, resolution, reduction_factor):
        """
        Returns the bandwidth for given peak flux density reduction factor

        Parameters
        ----------
        freq : float
            Frequency at which averaging will be done
        delta_theta : float
            Distance from phase center
        resolution : float
            Resolution of restoring beam
        reduction_factor : float
            Ratio of pre-to-post averaging peak flux density

        Returns
        -------
        delta_freq : float
            Bandwidth over which averaging will be done
        """
        # Increase delta_freq until we drop below target reduction_factor
        delta_freq = 1e-3 * freq
        while self.get_bandwidth_smearing_factor(freq, delta_freq, delta_theta,
                                                 resolution) > reduction_factor:
            delta_freq *= 1.1

        return delta_freq
