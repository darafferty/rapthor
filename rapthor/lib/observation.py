"""
Definition of the Observation class that holds parameters for each measurement set
"""
import os
import logging
import casacore.tables as pt
import numpy as np
from rapthor.lib.cluster import get_chunk_size
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
        self.ms_predict_nc_filename = None
        self.name = os.path.basename(self.ms_filename)
        self.log = logging.getLogger('rapthor:{}'.format(self.name))
        self.starttime = starttime
        self.endtime = endtime
        self.data_fraction = 1.0
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
        self.log, obs_log = None, self.log
        obs_copy = copy.deepcopy(self)
        obs_copy.log = logging.getLogger('rapthor:{}'.format(self.name))
        self.log = obs_log

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
        self.channelfreqs = sw.col('CHAN_FREQ')[0]
        self.startfreq = np.min(sw.col('CHAN_FREQ')[0])
        self.endfreq = np.max(sw.col('CHAN_FREQ')[0])
        self.numchannels = sw.col('NUM_CHAN')[0]
        self.channelwidths = sw.col('CHAN_WIDTH')[0]
        self.channelwidth = sw.col('CHAN_WIDTH')[0][0]
        sw.close()

        # Check that the channels are evenly spaced, as the use of baseline-dependent
        # averaging (BDA) during calibration requires it. If the channels are not evenly
        # spaced, BDA will not be used in DDECal steps even if activated in the parset
        #
        # Note: the code is based on that used in DP3
        # (see https://git.astron.nl/RD/DP3/-/blob/master/base/DPInfo.cc)
        self.channels_are_regular = True
        if self.numchannels > 1:
            freqstep0 = self.channelfreqs[1] - self.channelfreqs[0]
            atol = 1e3  # use an absolute tolerance of 1 kHz
            for i in range(1, self.numchannels):
                if (
                    (self.channelfreqs[i] - self.channelfreqs[i-1] - freqstep0 >= atol) or
                    (self.channelwidths[i] - self.channelwidths[0] >= atol)
                ):
                    self.channels_are_regular = False
        if not self.channels_are_regular:
            self.log.warning('Irregular spacing of channel frequencies found. Baseline-'
                             'dependent averaging (if any) will be disabled.')

        # Get pointing info
        obs = pt.table(self.ms_filename+'::FIELD', ack=False)
        self.ra, self.dec = misc.normalize_ra_dec(np.degrees(float(obs.col('REFERENCE_DIR')[0][0][0])),
                                                  np.degrees(float(obs.col('REFERENCE_DIR')[0][0][1])))
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
            # Set antenna to HBA to at least let Rapthor proceed.
            self.antenna = "HBA"
            self.log.warning(
                "Antenna type not recognized (only LBA and HBA data "
                "are supported at this time)"
            )
        ant.close()

        # Find mean elevation and time range for periods where the elevation
        # falls below the lowest 20% of all values. We sample every 10000 entries to
        # avoid very large arrays (note that for each time slot, there is an entry for
        # each baseline, so a typical HBA LOFAR observation would have around 1500 entries
        # per time slot)
        el_values = pt.taql("SELECT mscal.azel1()[1] AS el from "
                            + self.ms_filename + " limit ::10000").getcol("el")
        self.mean_el_rad = np.mean(el_values)
        times = pt.taql("select TIME from "
                        + self.ms_filename + " limit ::10000").getcol("TIME")
        low_indices = np.sort(el_values.argpartition(len(el_values)//5)[:len(el_values)//5])
        if len(low_indices):
            # At least one element is needed for the start and end time check. The check
            # assumes that the elevation either increases smoothly to a maximum and then
            # decreases with time or that it simply increases (or decreases)
            # monotonically. These cases should cover almost all observations. For any
            # other behavior, just fall back to the untrimmed time
            diff = np.diff(low_indices)
            low_at_start = low_indices[0] == 0  # first elevation is low
            low_at_end = low_indices[-1] == len(el_values) - 1  # last elevation is low
            starttime_index = 0  # default to no trim at start
            endtime_index = -1  # default to no trim at end
            if np.all(diff == 1):
                # No gap found (so a single continuous period of low elevations)
                if low_at_start:
                    # Trim the start
                    starttime_index = low_indices[-1]
                elif low_at_end:
                    # Trim the end
                    endtime_index = low_indices[0]
            elif len(np.where(diff != 1)[0]) == 1 and low_at_start and low_at_end:
                # Single gap found (with low elevations at start and end, due to a period
                # of higher elevations in between), so trim both start and end
                #
                # Note: low_indices will look something like:
                #   array([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9, 637, 638,
                #          639, 640, 641, 642, 643, 644, 645, 646, 647])
                # and diff like:
                #   array([  1,   1,   1,   1,   1,   1,   1,   1,   1, 545,   1,   1,
                #            1,   1,   1,   1,   1,   1,   1,   1])
                starttime_index = low_indices[np.argmax(diff)]
                endtime_index = low_indices[np.argmax(diff) + 1]
            self.high_el_starttime = times[starttime_index]
            self.high_el_endtime = times[endtime_index]
        else:
            # Too few times (or none) at lower elevations, so just ignore them
            self.high_el_starttime = self.starttime
            self.high_el_endtime = self.endtime

    def set_calibration_parameters(self, parset, ndir, nobs, calibrator_fluxes,
                                   target_fast_timestep, target_slow_timestep,
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
        target_medium_timestep : float
            Target solution interval for medium solves in sec
        target_slow_timestep : float
            Target solution interval for slow solves in sec
        target_fulljones_timestep : float
            Target solution interval for full-Jones solves in sec
        target_flux: float, optional
            Target calibrator flux in Jy. If None, the lowest calibrator flux density
            is used.
        """
        # Get the target solution intervals and maximum factor by which they can
        # be increased when using direction-dependent solution intervals
        target_fast_freqstep = parset['calibration_specific']['fast_freqstep_hz']
        target_medium_freqstep = parset['calibration_specific']['medium_freqstep_hz']
        target_slow_freqstep = parset['calibration_specific']['slow_freqstep_hz']
        target_fulljones_freqstep = parset['calibration_specific']['fulljones_freqstep_hz']
        solve_max_factor = parset['calibration_specific']['dd_interval_factor']
        smoothness_max_factor = parset['calibration_specific']['dd_smoothness_factor']

        # Find the maximum solution interval in time that can be used in any solve
        max_timestep = max(target_fast_timestep, target_medium_freqstep, target_slow_timestep, target_fulljones_timestep)
        solint_max_timestep = max(1, int(round(max_timestep / round(self.timepersample)) * solve_max_factor))

        # Determine how many calibration chunks to make (to allow parallel jobs)
        samplesperchunk = get_chunk_size(parset['cluster_specific'], self.numsamples, nobs, solint_max_timestep)

        # Calculate start times, etc.
        chunksize = samplesperchunk * self.timepersample
        mystarttime = self.starttime
        myendtime = self.endtime
        if (myendtime - mystarttime) > chunksize:
            # Divide up the total duration into chunks of chunksize
            nchunks = int(np.ceil(self.numsamples * self.timepersample / chunksize))
        else:
            nchunks = 1
        starttimes = [mystarttime+(chunksize * i) for i in range(nchunks)]
        if starttimes[-1] >= myendtime:
            # Make sure the last start time does not equal or exceed the end time
            starttimes.pop(-1)
            nchunks -= 1
        self.ntimechunks = nchunks
        self.log.debug('Using {0} time chunk{1} for '
                       'calibration'.format(self.ntimechunks, "s" if self.ntimechunks > 1 else ""))
        if self.antenna == 'LBA':
            # For LBA, use the MS files with non-calibrator sources subtracted
            self.parameters['timechunk_filename'] = [self.ms_predict_nc_filename] * self.ntimechunks
        else:
            # For other data, use the primary MS files
            self.parameters['timechunk_filename'] = [self.ms_filename] * self.ntimechunks
        self.parameters['starttime'] = [misc.convert_mjd2mvt(t) for t in starttimes]
        self.parameters['ntimes'] = [samplesperchunk] * self.ntimechunks

        # Set last entry in ntimes list to extend to end of observation
        if self.goesto_endofms:
            self.parameters['ntimes'][-1] = 0
        else:
            self.parameters['ntimes'][-1] += int(self.numsamples - (samplesperchunk * self.ntimechunks))

        # Find solution intervals for fast-phase solve. The solve is split into time
        # chunks instead of frequency chunks, since continuous frequency coverage is
        # desirable to recover the expected smooth, TEC-like behavior (phase ~ nu^-1)
        #
        # Note: we don't explicitly check that the resulting solution intervals fit
        # within the observation's size, as this is handled by DP3
        solint_fast_timestep = max(1, int(round(target_fast_timestep / round(self.timepersample)) * solve_max_factor))
        solint_fast_freqstep = max(1, self.get_nearest_freqstep(target_fast_freqstep / self.channelwidth))
        solint_medium_timestep = max(1, int(round(target_medium_timestep / round(self.timepersample)) * solve_max_factor))
        solint_medium_freqstep = max(1, self.get_nearest_freqstep(target_medium_freqstep / self.channelwidth))

        # Set the fast solve solution intervals
        self.parameters['solint_fast_timestep'] = [solint_fast_timestep] * self.ntimechunks
        self.parameters['solint_fast_freqstep'] = [solint_fast_freqstep] * self.ntimechunks
        self.parameters['solint_medium_timestep'] = [solint_medium_timestep] * self.ntimechunks
        self.parameters['solint_medium_freqstep'] = [solint_medium_freqstep] * self.ntimechunks

        # Find solution intervals for the gain solves
        #
        # Note: as with the fast-phase solve, we don't explicitly check that the resulting
        # solution intervals fit within the observation's size, as this is handled by DP3
        solint_slow_timestep = max(1, int(round(target_slow_timestep / round(self.timepersample)) * solve_max_factor))
        solint_slow_freqstep = max(1, self.get_nearest_freqstep(target_slow_freqstep / self.channelwidth))
        self.parameters['solint_slow_timestep'] = [solint_slow_timestep] * self.ntimechunks
        self.parameters['solint_slow_freqstep'] = [solint_slow_freqstep] * self.ntimechunks
        solint_fulljones_timestep = max(1, int(round(target_fulljones_timestep / round(self.timepersample))))
        solint_fulljones_freqstep = max(1, self.get_nearest_freqstep(target_fulljones_freqstep / self.channelwidth))
        self.parameters['solint_fulljones_timestep'] = [solint_fulljones_timestep] * self.ntimechunks
        self.parameters['solint_fulljones_freqstep'] = [solint_fulljones_freqstep] * self.ntimechunks

        # Define the BDA (baseline-dependent averaging) max interval constraints. They
        # are set to the solution intervals *before* adjusting for the DD intervals
        # to ensure that they match the smallest interval used in the solves (since
        # maxinterval cannot exceed solint in DDECal)
        self.parameters['bda_maxinterval'] = [max(1.0, int(min(solint_fast_timestep, solint_slow_timestep) / solve_max_factor) * self.timepersample)] * self.ntimechunks  # sec
        self.parameters['bda_minchannels'] = [max(1, int(self.numchannels / min(solint_fast_freqstep, solint_slow_freqstep)))] * self.ntimechunks  # channels

        # Define the direction-dependent solution interval list for the fast and
        # slow solves (the full-Jones solve is direction-independent so is not included).
        # The list values are defined as the number of solutions that will be obtained for
        # each base solution interval, with one entry per direction
        input_solint_keys = {'slow': 'solint_slow_timestep',
                             'medium': 'solint_medium_timestep',
                             'fast': 'solint_fast_timestep'}
        if target_flux is None:
            target_flux = min(calibrator_fluxes)
        if smoothness_max_factor > 1:
            smoothness_dd_factors = target_flux / np.array(calibrator_fluxes)
            smoothness_dd_factors /= max(smoothness_dd_factors)
            smoothness_dd_factors[smoothness_dd_factors < 1 / smoothness_max_factor] = 1 / smoothness_max_factor
        else:
            smoothness_dd_factors = [1] * len(calibrator_fluxes)
        for solve_type in ['fast', 'medium', 'slow']:
            solint = self.parameters[input_solint_keys[solve_type]][0]  # number of time slots

            if solve_max_factor > 1:
                # Find the initial estimate for the number of solutions, relative to that
                # for a source with a flux equal to the target flux and at most
                # solve_max_factor, with the brighter calibrators getting a larger number
                # and the fainter ones a smaller number (smaller numbers give longer
                # solution intervals)
                interval_factors = np.round(np.array(calibrator_fluxes) / target_flux)
                n_solutions = [min(solve_max_factor, max(1, int(factor))) for
                               factor in interval_factors]

                # Calculate the final number per direction, making sure each is a divisor
                # of the solution interval. We choose the lower number that satisfies this
                # requirement, as it will result in a longer solution interval (and
                # therefore a higher SNR) and so is generally safer than going the other
                # way (towards low SNRs)
                solutions_per_direction = []
                for n_sols in n_solutions:
                    while solint % n_sols:
                        n_sols -= 1
                    solutions_per_direction.append(n_sols)
                self.parameters[f'{solve_type}_solutions_per_direction'] = [solutions_per_direction] * self.ntimechunks

            else:
                self.parameters[f'{solve_type}_solutions_per_direction'] = [[1] * len(calibrator_fluxes)] * self.ntimechunks

            # Set the smoothness_dd_factors so that brighter sources have smaller
            # smoothing factors
            self.parameters[f'{solve_type}_smoothness_dd_factors'] = [smoothness_dd_factors] * self.ntimechunks

        # Set the smoothnessreffrequency for the fast solves, if not set by the user
        fast_smoothnessreffrequency = parset['calibration_specific']['fast_smoothnessreffrequency']
        if fast_smoothnessreffrequency is None:
            if self.antenna == 'HBA':
                fast_smoothnessreffrequency = 144e6
            elif self.antenna == 'LBA':
                # Select a frequency at the midpoint of the frequency coverage of this observation
                fast_smoothnessreffrequency = (self.startfreq + self.endfreq) / 2.0
        self.parameters['fast_smoothnessreffrequency'] = [fast_smoothnessreffrequency] * self.ntimechunks
        medium_smoothnessreffrequency = parset['calibration_specific']['medium_smoothnessreffrequency']
        if medium_smoothnessreffrequency is None:
            medium_smoothnessreffrequency = fast_smoothnessreffrequency
        self.parameters['medium_smoothnessreffrequency'] = [medium_smoothnessreffrequency] * self.ntimechunks

    def set_prediction_parameters(self, sector_name, patch_names):
        """
        Sets the prediction parameters

        Parameters
        ----------
        sector_name : str
            Name of sector for which predict is to be done
        patch_names : list
            List of patch names to predict
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
                               width_dec, solve_fast_timestep, solve_slow_timestep,
                               solve_slow_freqstep, preapply_dde_solutions):
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
        solve_slow_timestep : float
            Solution interval in sec for slow solve
        solve_slow_freqstep : float
            Solution interval in Hz for slow solve
        preapply_dde_solutions : bool
            If True, use setup appropriate for case in which all DDE
            solutions are preapplied before imaging is done
        """
        mean_freq_mhz = self.referencefreq / 1e6
        peak_smearing_rapthor = np.sqrt(1.0 - max_peak_smearing)
        chan_width_hz = self.channelwidth
        nchan = self.numchannels
        timestep_sec = self.timepersample

        # Set MS filenames for step that prepares the data for imaging
        if 'ms_filename' not in self.parameters:
            self.parameters['ms_filename'] = self.ms_filename
        root_filename = os.path.basename(self.ms_filename)
        ms_prep_filename = '{0}{1}.{2}.prep'.format(root_filename, self.infix,
                                                    sector_name)
        self.parameters['ms_prep_filename'] = ms_prep_filename

        # Get target time and frequency averaging steps.
        #
        # Note: We limit the averaging to be not more than 2 MHz and 120 s to avoid
        # extreme values for very small images (when solutions are preapplied). Also, due
        # to a limitation in Dysco, we make sure to have at least 2 time slots after
        # averaging, otherwise the output MS cannot be written with compression
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

        if not preapply_dde_solutions:
            # Ensure we don't average more than the solve time step, as we want to
            # preserve the time resolution so that the soltuions can be applied
            # properly during imaging
            target_timewidth_sec = min(target_timewidth_sec, solve_fast_timestep)

        target_bandwidth_mhz = min(2.0, self.get_target_bandwidth(mean_freq_mhz,
                                   delta_theta_deg, resolution_deg, peak_smearing_rapthor))
        target_bandwidth_mhz = min(target_bandwidth_mhz, solve_slow_freqstep/1e6)
        self.log.debug('Target timewidth for imaging is {0:.1f} s'.format(target_timewidth_sec))
        self.log.debug('Target bandwidth for imaging is {0:.1f} MHz'.format(target_bandwidth_mhz))

        # Find averaging steps for above target values
        image_freqstep = max(1, min(int(round(target_bandwidth_mhz * 1e6 / chan_width_hz)), nchan))
        self.parameters['image_freqstep'] = self.get_nearest_freqstep(image_freqstep)
        self.parameters['image_timestep'] = max(1, int(round(target_timewidth_sec / timestep_sec)))
        self.log.debug('Using averaging steps of {0} channel{1} and {2} time slot{3} '
                       'for imaging'.format(self.parameters['image_freqstep'],
                                            "s" if self.parameters['image_freqstep'] > 1 else "",
                                            self.parameters['image_timestep'],
                                            "s" if self.parameters['image_timestep'] > 1 else ""))

        # Find BDA maxinterval: the max time interval in time slots over which to average
        # (for the shortest baselines). We set this to be the slow solve time step to ensure
        # we don't average more than the timescale of the slow corrections
        target_maxinterval = min(self.numsamples, int(round(solve_slow_timestep / timestep_sec)))  # time slots
        self.parameters['image_bda_maxinterval'] = max(1, target_maxinterval)
        self.log.debug('Using BDA with maxinterval = {0:.1f} s for '
                       'imaging'.format(self.parameters['image_bda_maxinterval'] * timestep_sec))

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
