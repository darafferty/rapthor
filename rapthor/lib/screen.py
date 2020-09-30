"""
Module that holds screen-related classes and functions
"""
import logging
from numpy import concatenate, newaxis
from numpy.linalg import norm
import numpy as np
import os
from rapthor.lib import miscellaneous as misc
import sys
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


class Screen(object):
    """
    Master class for a-term screens

    Parameters
    ----------
    phase_soltab : soltab
        Solution table of phases
    amplitude_soltab : soltab
        Solution table of amplitudes
    ra : float
        RA in degrees of screen center
    dec : float
        Dec in degrees of screen center
    width_ra : float
        Width of screen in RA degrees
    width_dec : float
        Width of screen in Dec in degrees
    skymodel: str
        Filename of sky model from which screen solutions were generated
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
        self.width_ra = width_ra
        self.width_dec = width_dec

        # Do some checking
        H = h5parm(self.input_h5parm_filename)
        if self.input_solset_name not in H.getSolsetNames():
            self.log.critical('Solset {} not found in input h5parm! '
                              'Exiting!'.format(self.input_solset_name))
            sys.exit(1)
        solset = H.getSolset(self.input_solset_name)
        if self.input_phase_soltab_name not in solset.getSoltabNames():
            self.log.critical('Soltab {} not found in input solset! '
                              'Exiting!'.format(self.input_phase_soltab_name))
            sys.exit(1)
        if not self.phase_only:
            if self.input_amplitude_soltab_name not in solset.getSoltabNames():
                self.log.critical('Soltab {} not found in input solset! '
                                  'Exiting!'.format(self.input_amplitude_soltab_name))
                sys.exit(1)
        H.close()

    def fit(self):
        """
        Fits screens to the input solutions

        This should be defined in the subclasses
        """
        pass

    def interpolate(self, interp_kind='nearest'):
        """
        Interpolate the slow amplitude values to the fast-phase time and frequency grid
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
        cellsize_deg : float
            Pixel size of image in degrees
        """
        ximsize = int(self.width_ra / cellsize_deg)  # pix
        yimsize = int(self.width_dec / cellsize_deg)  # pix
        ximsize = yimsize  # force square image until rectangular ones are supported by IDG
        misc.make_template_image(outfile, self.ra, self.dec, ximsize=ximsize,
                                 yimsize=yimsize, cellsize_deg=cellsize_deg, freqs=self.freqs_ph,
                                 times=self.times_ph[t_start_index:t_stop_index],
                                 antennas=self.station_names, aterm_type=aterm_type)
        hdu = pyfits.open(outfile, memmap=False)
        return hdu

    def make_matrix(self, t_start_index, t_stop_index, freq_ind, stat_ind, cellsize_deg, out_dir):
        """
        Makes the matrix of values for the given time, frequency, and station indices

        This should be defined in the subclasses
        """
        pass

    def write(self, out_dir, cellsize_deg, smooth_pix=0, interp_kind='nearest'):
        """
        Write the a-term screens to a FITS data cube

        Parameters
        ----------
        out_dir : str
            Output directory
        cellsize_deg : float
            Pixel size of image in degrees
        """
        # Identify any gaps in time (frequency gaps are not allowed), as we need to
        # output a separate FITS file for each time chunk
        delta_times = self.times_ph[1:] - self.times_ph[:-1]  # time at center of solution interval
        timewidth = np.min(delta_times)
        gaps = np.where(delta_times > timewidth*1.2)
        gaps_ind = gaps[0] + 1
        gaps_ind = np.append(gaps_ind, np.array([len(self.times_ph)]))

        # Add additional breaks to gaps_ind to keep memory use within that available
        # From experience, making a (30, 46, 62, 4, 146, 146) aterm image needs around
        # 30 GB of memory
        max_ntimes = 15
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
                    data[:, f, s, :, :, :] = self.make_matrix(g_start, g_stop, f, s, cellsize_deg, out_dir)

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

        # Write filenames to a text file for later use
        outfile = open(outroot+'.txt', 'w')
        outfile.writelines([o+'\n' for o in outfiles])
        outfile.close()

    def process(self):
        """
        Makes a-term images
        """
        # Fit screens to input solutions
        self.fit()

        # Interpolate best-fit parameters to common time and frequency grid
        self.interpolate()


class KLScreen(Screen):
    """
    Screen class for KL (Karhunen-Lo`eve) screens

    Parameters
    ----------
    phase_soltab : soltab
        Solution table of phases
    amplitude_soltab : soltab
        Solution table of amplitudes
    ra : float
        RA in degrees of screen center
    dec : float
        Dec in degrees of screen center
    width_ra : float
        Width of screen in RA degrees
    width_dec : float
        Width of screen in Dec in degrees
    """
    def __init__(self, name, h5parm_filename, skymodel_filename, ra, dec, width_ra, width_dec,
                 solset_name='sol000', phase_soltab_name='phase000', amplitude_soltab_name=None):
        super(KLScreen, self).__init__(name, h5parm_filename, skymodel_filename, ra, dec, width_ra, width_dec,
                                       solset_name=solset_name, phase_soltab_name=phase_soltab_name, amplitude_soltab_name=amplitude_soltab_name)

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
        for source in self.source_names:
            radecpos = source_dict[source.strip('[]')]
            source_positions.append([radecpos[0].value, radecpos[1].value])
        source_positions = np.array(source_positions)
        ra_deg = source_positions.T[0]
        dec_deg = source_positions.T[1]
        sourceTable = solset.obj._f_get_child('source')
        vals = [[ra, dec] for ra, dec in zip(ra_deg, dec_deg)]
        sourceTable.append(list(zip(*(self.source_names, vals))))

        # Reweight the input solutions by the scatter after detrending
#        reweight.run(soltab_ph, mode='window', nmedian=3, nstddev=251)
#        if self.input_amplitude_soltab_name is not None:
#            reweight.run(soltab_amp, mode='window', nmedian=3, nstddev=21)

        # Now call LoSoTo's stationscreen operation to do the fitting
        stationscreen.run(soltab_ph, 'phase_screen000')
        soltab_ph_screen = solset.getSoltab('phase_screen000')
        if not self.phase_only:
            stationscreen.run(soltab_amp, 'amplitude_screen000')
            soltab_amp_screen = solset.getSoltab('amplitude_screen000')
        else:
            soltab_amp_screen = None

        # Read in the screen solutions and parameters
        self.vals_ph = soltab_ph_screen.val
        self.times_ph = soltab_ph_screen.time
        self.freqs_ph = soltab_ph_screen.freq
        if not self.phase_only:
            self.vals_amp = soltab_amp_screen.val
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

    def make_matrix(self, t_start_index, t_stop_index, freq_ind, stat_ind, cellsize_deg, out_dir, ncpu=0):
        """
        Makes the matrix of values for the given time, frequency, and station indices
        """
        # Define various parameters
        N_sources = len(self.source_names)
        N_times = t_stop_index - t_start_index
        N_piercepoints = N_sources
        xp, yp, zp = self.station_positions[0]
        east = np.array([-yp, xp, 0])
        east = east / norm(east)
        north = np.array([-xp, -yp, (xp*xp + yp*yp)/zp])
        north = north / norm(north)
        up = np.array([xp, yp, zp])
        up = up / norm(up)
        T = concatenate([east[:, newaxis], north[:, newaxis]], axis=1)

        ra_deg = self.source_positions.T[0]
        dec_deg = self.source_positions.T[1]
        xy, _, _ = stationscreen._getxy(ra_deg, dec_deg, midRA=self.midRA, midDec=self.midDec)

        # Get boundary of screen region in pixels
        bounds_deg = [self.ra+self.width_ra/2.0, self.dec-self.width_dec/2.0,
                      self.ra-self.width_ra/2.0, self.dec+self.width_dec/2.0]
        ra_max = [max(bounds_deg[0], np.max(ra_deg)+0.1)]
        dec_min = [min(bounds_deg[1], np.min(dec_deg)-0.1)]
        field_minxy = stationscreen._getxy(ra_max, dec_min, midRA=self.midRA, midDec=self.midDec)[0]
        ra_min = [min(bounds_deg[2], np.min(ra_deg)-0.1)]
        dec_max = [max(bounds_deg[3], np.max(dec_deg)+0.1)]
        field_maxxy = stationscreen._getxy(ra_min, dec_max, midRA=self.midRA, midDec=self.midDec)[0]

        # Make arrays of pixel coordinates for screen
        pix1 = stationscreen._getxy(self.midRA, self.midDec, midRA=self.midRA, midDec=self.midDec)[0]
        pix2 = stationscreen._getxy(self.midRA, self.midDec+cellsize_deg, midRA=self.midRA, midDec=self.midDec)[0]
        cellsize_pix = pix2[1] - pix1[1]
        x = np.arange(field_minxy[0], field_maxxy[0], cellsize_pix)
        y = np.arange(field_minxy[1], field_maxxy[1], cellsize_pix)
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
                     east, north, up, T, x, y, self.beta_val, self.r_0])
        mpm.wait()
        for (k, scr) in mpm.get():
            val_phase[:, :, k] = scr

        # Process amplitude screens
        if not self.phase_only:
            # XX amplitudes
            val_amp_xx = np.zeros((Nx, Ny, N_times))
            mpm = misc.multiprocManager(ncpu, calculate_kl_screen)
            for k in range(N_times):
                mpm.put([screen_amp_xx[:, k], self.pp, N_piercepoints, k,
                         east, north, up, T, x, y, self.beta_val, self.r_0])
            mpm.wait()
            for (k, scr) in mpm.get():
                val_amp_xx[:, :, k] = scr

            # YY amplitudes
            val_amp_yy = np.zeros((Nx, Ny, N_times))
            mpm = misc.multiprocManager(ncpu, calculate_kl_screen)
            for k in range(N_times):
                mpm.put([screen_amp_yy[:, k], self.pp, N_piercepoints, k,
                         east, north, up, T, x, y, self.beta_val, self.r_0])
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
    Screen class for Voronoi screens

    Parameters
    ----------
    phase_soltab : soltab
        Solution table of phases
    amplitude_soltab : soltab
        Solution table of amplitudes
    ra : float
        RA in degrees of screen center
    dec : float
        Dec in degrees of screen center
    width_ra : float
        Width of screen in RA degrees
    width_dec : float
        Width of screen in Dec in degrees
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

    def make_matrix(self, t_start_index, t_stop_index, freq_ind, stat_ind, cellsize_deg, out_dir):
        """
        Makes the matrix of values for the given time, frequency, and station indices
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


def calculate_kl_screen(inscreen, pp, N_piercepoints, k, east, north, up,
                        T, x, y, beta_val, r_0, outQueue):
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
    east : array
        East array
    north : array
        North array
    up : array
        Up array
    T : array
        T array
    x : int
        X coordinate for pixels in screen
    y : int
        Y coordinate for pixels in screen
    beta_val : float
        power-law index for phase structure function (5/3 =>
        pure Kolmogorov turbulence)
    r_0 : float
        scale size of phase fluctuations
    """
    Nx = len(x)
    Ny = len(y)
    screen = np.zeros((Nx, Ny))

    D = np.resize(pp, (N_piercepoints, N_piercepoints, 3))
    D = np.transpose(D, (1, 0, 2)) - D
    f = inscreen.reshape(N_piercepoints)
    for i, xi in enumerate(x):
        for j, yi in enumerate(y):
            p = np.array([xi, yi, 0.0])
            d2 = np.sum(np.square(pp - p), axis=1)
            c = -(d2 / ( r_0**2 ))**(beta_val / 2.0) / 2.0
            screen[i, j] = np.dot(c, f)

    outQueue.put([k, screen])
