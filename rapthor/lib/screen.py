"""
Module that holds screen-related classes and functions
"""
import logging
import numpy as np
import os
from rapthor.lib import miscellaneous as misc
from rapthor.lib import cluster
from astropy.coordinates import Angle
from losoto.h5parm import h5parm
import scipy.interpolate as si
from astropy.io import fits as pyfits
import scipy.ndimage as ndimage
from losoto.operations import reweight, stationscreen
from astropy import wcs
import lsmtool
from shapely.geometry import Point
from scipy.spatial import Voronoi
import shapely.geometry
import shapely.ops
import multiprocessing


class Screen(object):
    """
    Master class for a-term screens

    Parameters
    ----------
    name : str
        Name of screen
    h5parm_filename : str
        Filename of h5parm containing the input solutions
    skymodel_filename : str
        Filename of input sky model
    ra : float
        RA in degrees of screen center
    dec : float
        Dec in degrees of screen center
    width_ra : float
        Width of screen in RA in degrees, corrected to Dec = 0
    width_dec : float
        Width of screen in Dec in degrees
    solset_name: str, optional
        Name of solset of the input h5parm to use
    phase_soltab_name: str, optional
        Name of the phase soltab of the input h5parm to use
    amplitude_soltab_name: str, optional
        Name of amplitude soltab of the input h5parm to use
    """
    def __init__(self, name, h5parm_filename, skymodel_filename, ra, dec, width_ra, width_dec,
                 solset_name='sol000', phase_soltab_name='phase000', amplitude_soltab_name=None):
        self.name = name
        self.log = logging.getLogger('rapthor:{}'.format(self.name))
        self.input_h5parm_filename = h5parm_filename
        self.input_skymodel_filename = skymodel_filename
        self.input_solset_name = solset_name
        self.input_phase_soltab_name = phase_soltab_name
        self.input_amplitude_soltab_name = amplitude_soltab_name
        if self.input_amplitude_soltab_name is not None:
            self.phase_only = False
        else:
            self.phase_only = True
        if type(ra) is str:
            ra = Angle(ra).to('deg').value
        if type(dec) is str:
            dec = Angle(dec).to('deg').value
        self.ra = ra
        self.dec = dec
        width = max(width_ra, width_dec)  # force square image until rectangular ones are supported by IDG
        self.width_ra = width
        self.width_dec = width

    def fit(self):
        """
        Fits screens to the input solutions

        This method should be defined in the subclasses
        """
        pass

    def interpolate(self, interp_kind='nearest'):
        """
        Interpolate the slow amplitude values to the fast-phase time and frequency grid

        Parameters
        ----------
        interp_kind : str, optional
            Kind of interpolation to use
        """
        if self.phase_only:
            return

        if len(self.times_amp) == 1:
            # If only a single time, we just repeat the values as needed
            new_shape = list(self.vals_amp.shape)
            new_shape[0] = self.vals_ph.shape[0]
            new_shape[1] = self.vals_ph.shape[1]
            self.vals_amp = np.resize(self.vals_amp, new_shape)
        else:
            # Interpolate amplitudes (in log space)
            logvals = np.log10(self.vals_amp)
            if self.vals_amp.shape[0] != self.vals_ph.shape[0]:
                f = si.interp1d(self.times_amp, logvals, axis=0, kind=interp_kind, fill_value='extrapolate')
                logvals = f(self.times_ph)
            if self.vals_amp.shape[1] != self.vals_ph.shape[1]:
                f = si.interp1d(self.freqs_amp, logvals, axis=1, kind=interp_kind, fill_value='extrapolate')
                logvals = f(self.freqs_ph)
            self.vals_amp = 10**(logvals)

    def make_fits_file(self, outfile, cellsize_deg, t_start_index,
                       t_stop_index, aterm_type='gain'):
        """
        Makes a FITS data cube and returns the Header Data Unit

        Parameters
        ----------
        outfile : str
            Filename of output FITS file
        cellsize_deg : float
            Pixel size of image in degrees
        t_start_index : int
            Index of first time
        t_stop_index : int
            Index of last time
        aterm_type : str, optional
            Type of a-term solutions
        """
        ximsize = int(self.width_ra / cellsize_deg)  # pix
        yimsize = int(self.width_dec / cellsize_deg)  # pix
        misc.make_template_image(outfile, self.ra, self.dec, ximsize=ximsize,
                                 yimsize=yimsize, cellsize_deg=cellsize_deg, freqs=self.freqs_ph,
                                 times=self.times_ph[t_start_index:t_stop_index],
                                 antennas=self.station_names, aterm_type=aterm_type)
        hdu = pyfits.open(outfile, memmap=False)
        return hdu

    def make_matrix(self, t_start_index, t_stop_index, freq_ind, stat_ind, cellsize_deg,
                    out_dir, ncpu):
        """
        Makes the matrix of values for the given time, frequency, and station indices

        This method should be defined in the subclasses, but should conform to the inputs
        below.

        Parameters
        ----------
        t_start_index : int
            Index of first time
        t_stop_index : int
            Index of last time
        t_start_index : int
            Index of frequency
        t_stop_index : int
            Index of station
        cellsize_deg : float
            Size of one pixel in degrees
        out_dir : str
            Full path to the output directory (needed for template file generation)
        ncpu : int, optional
            Number of CPUs to use (0 means all)
        """
        pass

    def get_memory_usage(self, cellsize_deg):
        """
        Returns memory usage per time slot in GB

        This method should be defined in the subclasses, but should conform to the inputs
        below.

        Parameters
        ----------
        cellsize_deg : float
            Size of one pixel in degrees
        """
        pass

    def write(self, out_dir, cellsize_deg, smooth_pix=0, interp_kind='nearest', ncpu=0):
        """
        Write the a-term screens to a FITS data cube

        Parameters
        ----------
        out_dir : str
            Output directory
        cellsize_deg : float
            Size of one pixel in degrees
        smooth_pix : int, optional
            Size of Gaussian in pixels to smooth with
        interp_kind : str, optional
            Kind of interpolation to use
        ncpu : int, optional
            Number of CPUs to use (0 means all)
        """
        self.ncpu = ncpu

        # Identify any gaps in time (frequency gaps are not allowed), as we need to
        # output a separate FITS file for each time chunk
        if len(self.times_ph) > 2:
            delta_times = self.times_ph[1:] - self.times_ph[:-1]  # time at center of solution interval
            timewidth = np.min(delta_times)
            gaps = np.where(delta_times > timewidth*1.2)
            gaps_ind = gaps[0] + 1
            gaps_ind = np.append(gaps_ind, np.array([len(self.times_ph)]))
        else:
            gaps_ind = np.array([len(self.times_ph)])

        # Add additional breaks to gaps_ind to keep memory usage within that available
        if len(self.times_ph) > 2:
            tot_mem_gb = cluster.get_total_memory()
            max_ntimes = max(1, int(tot_mem_gb / (self.get_memory_usage(cellsize_deg))))
            check_gaps = True
            while check_gaps:
                check_gaps = False
                g_start = 0
                gaps_ind_copy = gaps_ind.copy()
                for gnum, g_stop in enumerate(gaps_ind_copy):
                    if g_stop - g_start > max_ntimes:
                        new_gap = g_start + int((g_stop - g_start) / 2)
                        gaps_ind = np.insert(gaps_ind, gnum, np.array([new_gap]))
                        check_gaps = True
                        break
                    g_start = g_stop

        # Input data are [time, freq, ant, dir, pol] for slow amplitudes
        # and [time, freq, ant, dir] for fast phases (scalarphase).
        # Output data are [RA, DEC, MATRIX, ANTENNA, FREQ, TIME].T
        # Loop over stations, frequencies, and times and fill in the correct
        # matrix values (matrix dimension has 4 elements: real XX, imaginary XX,
        # real YY and imaginary YY)
        outroot = self.name
        outfiles = []
        g_start = 0
        for gnum, g_stop in enumerate(gaps_ind):
            ntimes = g_stop - g_start
            outfile = os.path.join(out_dir, '{0}_{1}.fits'.format(outroot, gnum))
            hdu = self.make_fits_file(outfile, cellsize_deg, g_start, g_stop, aterm_type='gain')
            data = hdu[0].data
            for f, freq in enumerate(self.freqs_ph):
                for s, stat in enumerate(self.station_names):
                    data[:, f, s, :, :, :] = self.make_matrix(g_start, g_stop, f, s,
                                                              cellsize_deg, out_dir, self.ncpu)

                    # Smooth if desired
                    if smooth_pix > 0:
                        for t in range(ntimes):
                            data[t, f, s, :, :, :] = ndimage.gaussian_filter(data[t, f, s, :, :, :],
                                                                             sigma=(0, smooth_pix,
                                                                                    smooth_pix),
                                                                             order=0)

            # Ensure there are no NaNs in the images, as WSClean will produced uncorrected,
            # uncleaned images if so. We replace NaNs with 1.0 and 0.0 for real and
            # imaginary parts, respectively
            # Note: we iterate over time to reduce memory usage
            for t in range(ntimes):
                for p in range(4):
                    if p % 2:
                        # Imaginary elements
                        nanval = 0.0
                    else:
                        # Real elements
                        nanval = 1.0
                    data[t, :, :, p, :, :][np.isnan(data[t, :, :, p, :, :])] = nanval

            # Write FITS file
            hdu[0].data = data
            hdu.writeto(outfile, overwrite=True)
            outfiles.append(outfile)
            hdu = None
            data = None

            # Update start time index before starting next loop
            g_start = g_stop

        # Write list of filenames to a text file for later use
        list_file = open(os.path.join(out_dir, '{0}.txt'.format(outroot)), 'w')
        list_file.writelines([o+'\n' for o in outfiles])
        list_file.close()

    def process(self, ncpu=0):
        """
        Makes a-term images

        Parameters
        ----------
        ncpu : int, optional
            Number of CPUs to use (0 means all)
        """
        self.ncpu = ncpu

        # Fit screens to input solutions
        self.fit()

        # Interpolate best-fit parameters to common time and frequency grid
        self.interpolate()


class KLScreen(Screen):
    """
    Class for KL (Karhunen-Lo`eve) screens
    """
    def __init__(self, name, h5parm_filename, skymodel_filename, ra, dec, width_ra, width_dec,
                 solset_name='sol000', phase_soltab_name='phase000', amplitude_soltab_name=None):
        super(KLScreen, self).__init__(name, h5parm_filename, skymodel_filename, ra, dec, width_ra, width_dec,
                                       solset_name=solset_name, phase_soltab_name=phase_soltab_name,
                                       amplitude_soltab_name=amplitude_soltab_name)

    def fit(self):
        """
        Fits screens to the input solutions
        """
        # Open solution tables
        H = h5parm(self.input_h5parm_filename, readonly=False)
        solset = H.getSolset(self.input_solset_name)
        soltab_ph = solset.getSoltab(self.input_phase_soltab_name)
        if not self.phase_only:
            soltab_amp = solset.getSoltab(self.input_amplitude_soltab_name)

        # Set the position of the calibration patches to those of
        # the input sky model, as the patch positions written to the h5parm
        # file by DPPP may be different
        skymod = lsmtool.load(self.input_skymodel_filename)
        source_dict = skymod.getPatchPositions()
        source_positions = []
        for source in soltab_ph.dir:
            radecpos = source_dict[source.strip('[]')]
            source_positions.append([radecpos[0].value, radecpos[1].value])
        source_positions = np.array(source_positions)
        ra_deg = source_positions.T[0]
        dec_deg = source_positions.T[1]
        sourceTable = solset.obj._f_get_child('source')
        vals = [[ra*np.pi/180.0, dec*np.pi/180.0] for ra, dec in zip(ra_deg, dec_deg)]
        sourceTable = list(zip(*(soltab_ph.dir, vals)))

        # Now call LoSoTo's stationscreen operation to do the fitting
        adjust_order_amp = True
        adjust_order_ph = True
        screen_order = min(20, len(source_positions)-1)
        stationscreen.run(soltab_ph, 'phase_screen000', order=screen_order,
                          scale_order=True, adjust_order=adjust_order_ph, ncpu=self.ncpu)
        soltab_ph_screen = solset.getSoltab('phase_screen000')
        if not self.phase_only:
            stationscreen.run(soltab_amp, 'amplitude_screen000', order=screen_order,
                              scale_order=False, adjust_order=adjust_order_amp, ncpu=self.ncpu)
            soltab_amp_screen = solset.getSoltab('amplitude_screen000')
        else:
            soltab_amp_screen = None

        # Read in the screen solutions and parameters
        self.vals_ph = soltab_ph_screen.val
        self.times_ph = soltab_ph_screen.time
        self.freqs_ph = soltab_ph_screen.freq
        if not self.phase_only:
            self.vals_amp = 10**(soltab_amp_screen.val)
            self.times_amp = soltab_amp_screen.time
            self.freqs_amp = soltab_amp_screen.freq
        self.source_names = soltab_ph_screen.dir
        self.source_dict = solset.getSou()
        self.source_positions = []
        for source in self.source_names:
            self.source_positions.append(self.source_dict[source])
        self.station_names = soltab_ph_screen.ant
        self.station_dict = solset.getAnt()
        self.station_positions = []
        for station in self.station_names:
            self.station_positions.append(self.station_dict[station])
        self.height = soltab_ph_screen.obj._v_attrs['height']
        self.beta_val = soltab_ph_screen.obj._v_attrs['beta']
        self.r_0 = soltab_ph_screen.obj._v_attrs['r_0']
        self.pp = np.array(soltab_ph_screen.obj.piercepoint)
        self.midRA = soltab_ph_screen.obj._v_attrs['midra']
        self.midDec = soltab_ph_screen.obj._v_attrs['middec']

    def get_memory_usage(self, cellsize_deg):
        """
        Returns memory usage per time slot in GB

        Parameters
        ----------
        cellsize_deg : float
            Size of one pixel in degrees
        """
        ncpu = self.ncpu
        if ncpu == 0:
            ncpu = multiprocessing.cpu_count()

        # Make a test array and find its memory usage
        ximsize = int(self.width_ra / cellsize_deg)  # pix
        yimsize = int(self.width_dec / cellsize_deg)  # pix
        test_array = np.zeros([1, len(self.freqs_ph), len(self.station_names), 4,
                               yimsize, ximsize])
        mem_per_timeslot_gb = test_array.nbytes/1024**3 * 10  # include factor of 10 overhead

        # Multiply by the number of CPUs, since each gets a copy
        mem_per_timeslot_gb *= ncpu

        return mem_per_timeslot_gb

    def make_matrix(self, t_start_index, t_stop_index, freq_ind, stat_ind, cellsize_deg,
                    out_dir, ncpu):
        """
        Makes the matrix of values for the given time, frequency, and station indices

        Parameters
        ----------
        t_start_index : int
            Index of first time
        t_stop_index : int
            Index of last time
        t_start_index : int
            Index of frequency
        t_stop_index : int
            Index of station
        cellsize_deg : float
            Size of one pixel in degrees
        out_dir : str
            Full path to the output directory
        ncpu : int, optional
            Number of CPUs to use (0 means all)
        """
        # Define various parameters
        N_sources = len(self.source_names)
        N_times = t_stop_index - t_start_index
        N_piercepoints = N_sources

        # Make arrays of pixel coordinates for screen
        # We need to convert the FITS cube pixel coords to screen pixel coords. The FITS cube
        # has self.ra, self.dec at (xsize/2, ysize/2)
        ximsize = int(self.width_ra / cellsize_deg)  # pix
        yimsize = int(self.width_dec / cellsize_deg)  # pix
        w = wcs.WCS(naxis=2)
        w.wcs.crpix = [ximsize/2.0, yimsize/2.0]
        w.wcs.cdelt = np.array([-cellsize_deg, cellsize_deg])
        w.wcs.crval = [self.ra, self.dec]
        w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        w.wcs.set_pv([(2, 1, 45.0)])

        x_fits = list(range(ximsize))
        y_fits = list(range(yimsize))
        ra = []
        dec = []
        for xf, yf in zip(x_fits, y_fits):
            x_y = np.array([[xf, yf]])
            ra.append(w.wcs_pix2world(x_y, 0)[0][0])
            dec.append(w.wcs_pix2world(x_y, 0)[0][1])
        xy, _, _ = stationscreen._getxy(ra, dec, midRA=self.midRA, midDec=self.midDec)
        x = xy[0].T
        y = xy[1].T
        Nx = len(x)
        Ny = len(y)

        # Select input data and reorder the axes to get axis order of [dir, time]
        # Input data are [time, freq, ant, dir, pol] for slow amplitudes
        # and [time, freq, ant, dir] for fast phases (scalarphase).
        time_axis = 0
        dir_axis = 1
        screen_ph = np.array(self.vals_ph[t_start_index:t_stop_index, freq_ind, stat_ind, :])
        screen_ph = screen_ph.transpose([dir_axis, time_axis])
        if not self.phase_only:
            screen_amp_xx = np.array(self.vals_amp[t_start_index:t_stop_index, freq_ind, stat_ind, :, 0])
            screen_amp_xx = screen_amp_xx.transpose([dir_axis, time_axis])
            screen_amp_yy = np.array(self.vals_amp[t_start_index:t_stop_index, freq_ind, stat_ind, :, 1])
            screen_amp_yy = screen_amp_yy.transpose([dir_axis, time_axis])

        # Process phase screens
        val_phase = np.zeros((Nx, Ny, N_times))
        mpm = misc.multiprocManager(ncpu, calculate_kl_screen)
        for k in range(N_times):
            mpm.put([screen_ph[:, k], self.pp, N_piercepoints, k,
                     x, y, self.beta_val, self.r_0])
        mpm.wait()
        for (k, scr) in mpm.get():
            val_phase[:, :, k] = scr

        # Process amplitude screens
        if not self.phase_only:
            # XX amplitudes
            val_amp_xx = np.zeros((Nx, Ny, N_times))
            mpm = misc.multiprocManager(ncpu, calculate_kl_screen)
            for k in range(N_times):
                mpm.put([np.log10(screen_amp_xx[:, k]), self.pp, N_piercepoints, k,
                         x, y, self.beta_val, self.r_0])
            mpm.wait()
            for (k, scr) in mpm.get():
                val_amp_xx[:, :, k] = scr

            # YY amplitudes
            val_amp_yy = np.zeros((Nx, Ny, N_times))
            mpm = misc.multiprocManager(ncpu, calculate_kl_screen)
            for k in range(N_times):
                mpm.put([np.log10(screen_amp_yy[:, k]), self.pp, N_piercepoints, k,
                         x, y, self.beta_val, self.r_0])
            mpm.wait()
            for (k, scr) in mpm.get():
                val_amp_yy[:, :, k] = scr

        # Output data are [RA, DEC, MATRIX, ANTENNA, FREQ, TIME].T
        data = np.zeros((N_times, 4, Ny, Nx))
        if self.phase_only:
            data[:, 0, :, :] = np.cos(val_phase.T)
            data[:, 2, :, :] = np.cos(val_phase.T)
            data[:, 1, :, :] = np.sin(val_phase.T)
            data[:, 3, :, :] = np.sin(val_phase.T)
        else:
            data[:, 0, :, :] = val_amp_xx.T * np.cos(val_phase.T)
            data[:, 2, :, :] = val_amp_yy.T * np.cos(val_phase.T)
            data[:, 1, :, :] = val_amp_xx.T * np.sin(val_phase.T)
            data[:, 3, :, :] = val_amp_yy.T * np.sin(val_phase.T)

        return data


class VoronoiScreen(Screen):
    """
    Class for Voronoi screens
    """
    def __init__(self, name, h5parm_filename, skymodel_filename, ra, dec, width_ra, width_dec,
                 solset_name='sol000', phase_soltab_name='phase000', amplitude_soltab_name=None):
        super(VoronoiScreen, self).__init__(name, h5parm_filename, skymodel_filename, ra, dec, width_ra, width_dec,
                                            solset_name=solset_name, phase_soltab_name=phase_soltab_name, amplitude_soltab_name=amplitude_soltab_name)
        self.data_rasertize_template = None

    def fit(self):
        """
        Fitting is not needed: the input solutions are used directly
        """
        # Open solution tables
        H = h5parm(self.input_h5parm_filename)
        solset = H.getSolset(self.input_solset_name)
        soltab_ph = solset.getSoltab(self.input_phase_soltab_name)
        if not self.phase_only:
            soltab_amp = solset.getSoltab(self.input_amplitude_soltab_name)

        # Input data are [time, freq, ant, dir, pol] for slow amplitudes
        # and [time, freq, ant, dir] for fast phases (scalarphase).
        self.vals_ph = soltab_ph.val
        self.times_ph = soltab_ph.time
        self.freqs_ph = soltab_ph.freq
        if not self.phase_only:
            self.vals_amp = soltab_amp.val
            self.times_amp = soltab_amp.time
            self.freqs_amp = soltab_amp.freq
        else:
            self.vals_amp = np.ones_like(self.vals_ph)
            self.times_amp = self.times_ph
            self.freqs_amp = self.freqs_ph

        self.source_names = soltab_ph.dir
        self.source_dict = solset.getSou()
        self.source_positions = []
        for source in self.source_names:
            self.source_positions.append(self.source_dict[source])
        self.station_names = soltab_ph.ant
        self.station_dict = solset.getAnt()
        self.station_positions = []
        for station in self.station_names:
            self.station_positions.append(self.station_dict[station])

    def get_memory_usage(self, cellsize_deg):
        """
        Returns memory usage per time slot in GB

        Parameters
        ----------
        cellsize_deg : float
            Size of one pixel in degrees
        """
        # Make a test array and find its memory usage
        ximsize = int(self.width_ra / cellsize_deg)  # pix
        yimsize = int(self.width_dec / cellsize_deg)  # pix
        test_array = np.zeros([1, len(self.freqs_ph), len(self.station_names), 4,
                               yimsize, ximsize])
        mem_per_timeslot_gb = test_array.nbytes/1024**3 * 10  # include factor of 10 overhead

        return mem_per_timeslot_gb

    def make_matrix(self, t_start_index, t_stop_index, freq_ind, stat_ind, cellsize_deg,
                    out_dir, ncpu):
        """
        Makes the matrix of values for the given time, frequency, and station indices

        Parameters
        ----------
        t_start_index : int
            Index of first time
        t_stop_index : int
            Index of last time
        t_start_index : int
            Index of frequency
        t_stop_index : int
            Index of station
        cellsize_deg : float
            Size of one pixel in degrees
        out_dir : str
            Full path to the output directory
        ncpu : int, optional
            Number of CPUs to use (0 means all)
        """
        # Make the template that converts polynomials to a rasterized 2-D image.
        # This only needs to be done once
        if self.data_rasertize_template is None:
            self.make_rasertize_template(cellsize_deg, out_dir)

        # Fill the output data array
        data = np.zeros((t_stop_index-t_start_index, 4, self.data_rasertize_template.shape[0],
                         self.data_rasertize_template.shape[1]))
        for p, poly in enumerate(self.polygons):
            ind = np.where(self.data_rasertize_template == poly.index+1)
            if not self.phase_only:
                val_amp_xx = self.vals_amp[t_start_index:t_stop_index, freq_ind, stat_ind, poly.index, 0]
                val_amp_yy = self.vals_amp[t_start_index:t_stop_index, freq_ind, stat_ind, poly.index, 1]
            else:
                val_amp_xx = self.vals_amp[t_start_index:t_stop_index, freq_ind, stat_ind, poly.index]
                val_amp_yy = val_amp_xx
            val_phase = self.vals_ph[t_start_index:t_stop_index, freq_ind, stat_ind, poly.index]
            for t in range(t_stop_index-t_start_index):
                data[t, 0, ind[0], ind[1]] = val_amp_xx[t] * np.cos(val_phase[t])
                data[t, 2, ind[0], ind[1]] = val_amp_yy[t] * np.cos(val_phase[t])
                data[t, 1, ind[0], ind[1]] = val_amp_xx[t] * np.sin(val_phase[t])
                data[t, 3, ind[0], ind[1]] = val_amp_yy[t] * np.sin(val_phase[t])

        return data

    def make_rasertize_template(self, cellsize_deg, out_dir):
        """
        Makes the template that is used to fill the output FITS cube

        Parameters
        ----------
        cellsize_deg : float
            Size of one pixel in degrees
        out_dir : str
            Full path to the output directory
        """
        temp_image = os.path.join(out_dir, '{}_template.fits'.format(self.name))
        hdu = self.make_fits_file(temp_image, cellsize_deg, 0, 1, aterm_type='gain')
        data = hdu[0].data
        w = wcs.WCS(hdu[0].header)
        RAind = w.axis_type_names.index('RA')
        Decind = w.axis_type_names.index('DEC')

        # Get x, y coords for directions in pixels. We use the input calibration sky
        # model for this, as the patch positions written to the h5parm file by DPPP may
        # be different
        skymod = lsmtool.load(self.input_skymodel_filename)
        source_dict = skymod.getPatchPositions()
        source_positions = []
        for source in self.source_names:
            radecpos = source_dict[source.strip('[]')]
            source_positions.append([radecpos[0].value, radecpos[1].value])
        source_positions = np.array(source_positions)
        ra_deg = source_positions.T[0]
        dec_deg = source_positions.T[1]

        xy = []
        for RAvert, Decvert in zip(ra_deg, dec_deg):
            ra_dec = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
            ra_dec[0][RAind] = RAvert
            ra_dec[0][Decind] = Decvert
            xy.append((w.wcs_world2pix(ra_dec, 0)[0][RAind], w.wcs_world2pix(ra_dec, 0)[0][Decind]))

        # Get boundary of tessellation region in pixels
        bounds_deg = [self.ra+self.width_ra/2.0, self.dec-self.width_dec/2.0,
                      self.ra-self.width_ra/2.0, self.dec+self.width_dec/2.0]
        ra_dec = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        ra_dec[0][RAind] = max(bounds_deg[0], np.max(ra_deg)+0.1)
        ra_dec[0][Decind] = min(bounds_deg[1], np.min(dec_deg)-0.1)
        field_minxy = (w.wcs_world2pix(ra_dec, 0)[0][RAind], w.wcs_world2pix(ra_dec, 0)[0][Decind])
        ra_dec[0][RAind] = min(bounds_deg[2], np.min(ra_deg)-0.1)
        ra_dec[0][Decind] = max(bounds_deg[3], np.max(dec_deg)+0.1)
        field_maxxy = (w.wcs_world2pix(ra_dec, 0)[0][RAind], w.wcs_world2pix(ra_dec, 0)[0][Decind])

        if len(xy) == 1:
            # If there is only a single direction, just make a single rectangular polygon
            box = [field_minxy, (field_minxy[0], field_maxxy[1]), field_maxxy, (field_maxxy[0], field_minxy[1]), field_minxy]
            polygons = [shapely.geometry.Polygon(box)]
        else:
            # For more than one direction, tessellate
            # Generate array of outer points used to constrain the facets
            nouter = 64
            means = np.ones((nouter, 2)) * np.array(xy).mean(axis=0)
            offsets = []
            angles = [np.pi/(nouter/2.0)*i for i in range(0, nouter)]
            for ang in angles:
                offsets.append([np.cos(ang), np.sin(ang)])
            radius = 2.0*np.sqrt( (field_maxxy[0]-field_minxy[0])**2 + (field_maxxy[1]-field_minxy[1])**2 )
            scale_offsets = radius * np.array(offsets)
            outer_box = means + scale_offsets

            # Tessellate and clip
            points_all = np.vstack([xy, outer_box])
            vor = Voronoi(points_all)
            lines = [
                shapely.geometry.LineString(vor.vertices[line])
                for line in vor.ridge_vertices
                if -1 not in line
            ]
            polygons = [poly for poly in shapely.ops.polygonize(lines)]

        # Index polygons to directions
        for i, xypos in enumerate(xy):
            for poly in polygons:
                if poly.contains(Point(xypos)):
                    poly.index = i

        # Rasterize the polygons to an array, with the value being equal to the
        # polygon's index+1
        data_template = np.ones(data[0, 0, 0, 0, :, :].shape)
        data_rasertize_template = np.zeros(data[0, 0, 0, 0, :, :].shape)
        for poly in polygons:
            verts_xy = poly.exterior.xy
            verts = []
            for x, y in zip(verts_xy[0], verts_xy[1]):
                verts.append((x, y))
            poly_raster = misc.rasterize(verts, data_template.copy()) * (poly.index+1)
            filled = np.where(poly_raster > 0)
            data_rasertize_template[filled] = poly_raster[filled]
        zeroind = np.where(data_rasertize_template == 0)
        if len(zeroind[0]) > 0:
            nonzeroind = np.where(data_rasertize_template != 0)
            data_rasertize_template[zeroind] = si.griddata((nonzeroind[0], nonzeroind[1]), data_rasertize_template[nonzeroind],
                                                           (zeroind[0], zeroind[1]), method='nearest')
        self.data_rasertize_template = data_rasertize_template
        self.polygons = polygons


def calculate_kl_screen(inscreen, pp, N_piercepoints, k, x, y, beta_val, r_0, outQueue):
    """
    Calculates screen images

    Parameters
    ----------
    inscreen : array
        Array of screen values at the piercepoints
    pp : array
        Array of piercepoint locations
    N_piercepoints : int
        Number of pierce points
    k : int
        Time index
    x : int
        X coordinate for pixels in screen
    y : int
        Y coordinate for pixels in screen
    beta_val : float
        power-law index for phase structure function (5/3 =>
        pure Kolmogorov turbulence)
    r_0 : float
        scale size of phase fluctuations
    outQueue : queue
        Queue to add results to
    """
    Nx = len(x)
    Ny = len(y)
    screen = np.zeros((Nx, Ny))
    f = inscreen.reshape(N_piercepoints)
    for i, xi in enumerate(x):
        for j, yi in enumerate(y):
            p = np.array([xi, yi, 0.0])
            d2 = np.sum(np.square(pp - p), axis=1)
            c = -(d2 / ( r_0**2 ))**(beta_val / 2.0) / 2.0
            screen[i, j] = np.dot(c, f)

    outQueue.put([k, screen])
