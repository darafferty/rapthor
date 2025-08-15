"""
Definition of the Sector class that holds parameters for an image or predict sector
"""
import copy
import logging
import os
import pickle

import astropy.units as u
import lsmtool
import numpy as np
from astropy.coordinates import Angle, SkyCoord
from matplotlib import patches
from shapely.geometry import Polygon

from rapthor.lib import cluster, facet
from rapthor.lib import miscellaneous as misc


class Sector(object):
    """
    The Sector object contains various parameters for a sector of the field. Sectors
    are used only in image and predict operations

    Parameters
    ----------
    name : str
        Name of sector
    ra : float
        RA in degrees of sector center
    dec : float
        Dec in degrees of sector center
    width_ra : float
        Width of sector in RA degrees
    width_dec : float
        Width of sector in Dec in degrees
    field : Field object
        Field object
    """
    def __init__(self, name, ra, dec, width_ra, width_dec, field):
        self.name = name
        self.log = logging.getLogger('rapthor:{0}'.format(self.name))
        if type(ra) is str:
            ra = Angle(ra).to('deg').value
        if type(dec) is str:
            dec = Angle(dec).to('deg').value
        self.ra, self.dec = misc.normalize_ra_dec(ra, dec)
        self.width_ra = width_ra
        self.width_dec = width_dec
        self.field = field
        self.vertices_file = os.path.join(field.working_dir, 'regions',
                                          '{}_vertices.pkl'.format(self.name))
        self.region_file = None
        self.I_image_file_true_sky = None  # set by the Image operation
        self.I_image_file_apparent_sky = None  # set by the Image operation
        self.I_mask_file = None   # set by the Image operation
        self.image_skymodel_file_apparent_sky = None  # set by the Image operation
        self.image_skymodel_file_true_sky = None  # set by the Image operation
        self.max_wsclean_nchannels = None  # set by the Image operation
        self.is_outlier = False
        self.is_bright_source = False
        self.is_predict = False
        self.imsize = None  # set to None to force calculation in set_imaging_parameters()
        self.wsclean_image_padding = 1.2  # the WSClean default value, used in the workflows
        self.diagnostics = []  # list to hold dicts of image diagnostics
        self.calibration_skymodel = None  # set by Field.update_skymodel()
        self.max_nmiter = None  # set by the strategy
        self.normalize_h5parm = None  # set by the ImageNormalize operation

        # Make copies of the observation objects, as each sector may have its own
        # observation-specific settings
        self.observations = []
        for obs in field.observations:
            obs.log = None  # deepcopy cannot copy the log object
            cobs = copy.deepcopy(obs)
            obs.log = logging.getLogger('rapthor:{}'.format(obs.name))
            cobs.log = logging.getLogger('rapthor:{}'.format(cobs.name))
            self.observations.append(cobs)

        # Define the initial sector polygon vertices
        self.intialize_vertices()

    def set_prediction_parameters(self):
        """
        Sets the predict parameters
        """
        for obs in self.observations:
            obs.set_prediction_parameters(self.name, self.patches)

    def set_imaging_parameters(self, do_multiscale=False, recalculate_imsize=False,
                               imaging_parameters=None, preapply_dde_solutions=False):
        """
        Sets the parameters needed for the imaging operation

        Parameters
        ----------
        do_multiscale : bool, optional
            If True, multiscale clean is done
        recalculate_imsize : bool, optional
            If True, the image size is recalculated based on the current sector region
        imaging_parameters : dict, optional
            Dict of imaging parameters to use instead of those defined by the field's
            parset. If supplied, the following keys are expected to be present:
                'cellsize_arcsec': cell (pixel) size in arcsec
                'robust': Briggs robust value
                'taper_arcsec': taper in arcsec
                'local_rms_strength': local RMS strength factor
                'local_rms_window': local RMS window size
                'local_rms_method': local RMS method
                'min_uv_lambda': minimum uv distance cut in lambda
                'max_uv_lambda': maximum uv distance cut in lambda
                'mgain': cleaning gain
                'idg_mode': IDG processing mode
                'mem_gb': maximum memory in GB
                'reweight': reweighting flag
                'dd_psf_grid': DD PSF grid
                'max_peak_smearing': maximum allowed peak smearing
        preapply_dde_solutions : bool, optional
            If True, use setup appropriate for case in which all DDE
            solutions are preapplied before imaging is done
        """
        if imaging_parameters is None:
            imaging_parameters = self.field.parset['imaging_specific']
        self.cellsize_arcsec = imaging_parameters['cellsize_arcsec']
        self.cellsize_deg = self.cellsize_arcsec / 3600.0
        self.robust = imaging_parameters['robust']
        self.taper_arcsec = imaging_parameters['taper_arcsec']
        self.local_rms_strength = imaging_parameters['local_rms_strength']
        self.local_rms_window = imaging_parameters['local_rms_window']
        self.local_rms_method = imaging_parameters['local_rms_method']
        self.min_uv_lambda = imaging_parameters['min_uv_lambda']
        self.max_uv_lambda = imaging_parameters['max_uv_lambda']
        self.mgain = imaging_parameters['mgain']
        self.idg_mode = imaging_parameters['idg_mode']
        self.mem_limit_gb = imaging_parameters['mem_gb']
        slurm_limit_gb = self.field.parset['cluster_specific']['mem_per_node_gb']
        if slurm_limit_gb > 0:
            # Obey the Slurm limit if it's set and is more restrictive than the
            # WSClean-specific limit
            if self.mem_limit_gb > 0:
                # WSClean-specific limit set, so take the lower limit
                self.mem_limit_gb = min(self.mem_limit_gb, slurm_limit_gb)
            else:
                # WSClean-specific limit not set (i.e., use all available memory), so
                # take Slurm limit
                self.mem_limit_gb = slurm_limit_gb
        if self.mem_limit_gb == 0:
            # If no limit is set at this point, use the memory of the current machine
            self.mem_limit_gb = cluster.get_available_memory()
        self.reweight = imaging_parameters['reweight']
        self.target_fast_timestep = self.field.fast_timestep_sec
        self.target_slow_timstep = self.field.slow_timestep_separate_sec
        self.target_slow_freqstep = self.field.parset['calibration_specific']['slow_freqstep_hz']
        self.apply_screens = self.field.apply_screens

        # Set image size based on current sector polygon
        if recalculate_imsize or self.imsize is None:
            xmin, ymin, xmax, ymax = self.poly.bounds
            self.width_ra = (xmax - xmin) * self.field.wcs_pixel_scale  # deg
            self.width_dec = (ymax - ymin) * self.field.wcs_pixel_scale  # deg
            self.imsize = [int(self.width_ra / self.cellsize_deg),
                           int(self.width_dec / self.cellsize_deg)]

            if self.apply_screens:
                # IDG does not yet support rectangular images, so ensure image
                # is square
                self.imsize = [max(self.imsize), max(self.imsize)]

                # IDG has problems with small images, so set minimum size to 500 pixels
                # and adjust padded polygon
                minsize = 500
                if max(self.imsize) < minsize:
                    dec_width_pix = self.width_dec / abs(self.field.wcs.wcs.cdelt[1])
                    padding_pix = dec_width_pix * (self.wsclean_image_padding - 1.0)
                    padding_pix *= minsize / max(self.imsize)  # scale padding to new imsize
                    self.poly_padded = self.poly.buffer(padding_pix)
                    self.imsize = [minsize, minsize]

            # Lastly, make sure the image size is an even number (odd sizes cause the
            # peak to lie not necessarily in the img center)
            if self.imsize[0] % 2:
                self.imsize[0] += 1
            if self.imsize[1] % 2:
                self.imsize[1] += 1

        self.wsclean_imsize = "'{0} {1}'".format(self.imsize[0], self.imsize[1])
        self.log.debug('Image size is {0} x {1} pixels'.format(
                       self.imsize[0], self.imsize[1]))

        # Set the direction-dependent PSF grid (defined as [# in RA, # in Dec]):
        #   [0, 0] => scale automatically with image size
        #   [1, 1] => direction-independent
        #   [X, Y] => user-defined
        self.dd_psf_grid = imaging_parameters['dd_psf_grid']
        if self.dd_psf_grid == [0, 0]:
            # Set the grid based on the image size, with ~ 1 PSF per square deg
            # of imaged area
            self.dd_psf_grid = [max(1, int(np.round(self.width_ra))),
                                max(1, int(np.round(self.width_dec)))]

        # Set number of output channels to get ~ 4 MHz per channel equivalent at 120 MHz
        # (the maximum averaging allowed for typical dTEC values of -0.5 < dTEC < 0.5)
        min_freq = np.min([obs.startfreq for obs in self.observations])
        target_bandwidth = 4e6 * min_freq / 120e6
        max_nchannels = np.max([obs.numchannels for obs in self.observations])
        min_nchannels = 4
        tot_bandwidth = 0.0
        for obs in self.observations:
            # Find observation with largest bandwidth
            obs_bandwidth = obs.numchannels * obs.channelwidth
            if obs_bandwidth > tot_bandwidth:
                tot_bandwidth = obs_bandwidth
        self.wsclean_nchannels = max(min_nchannels, min(max_nchannels, int(np.ceil(tot_bandwidth / target_bandwidth))))
        if self.max_wsclean_nchannels is not None:
            self.wsclean_nchannels = min(self.wsclean_nchannels, self.max_wsclean_nchannels)

        # Set number of channels to use in spectral fitting. We set this to the
        # number of channels, up to a maximum of 4 (and the fit spectral order to
        # one less)
        self.wsclean_deconvolution_channels = min(4, self.wsclean_nchannels)
        self.wsclean_spectral_poly_order = max(1, self.wsclean_deconvolution_channels-1)

        # Set number of iterations. We scale the number of iterations depending on the
        # integration time and the distance of the sector center to the phase center, to
        # account for the reduced sensitivity of the image, assuming a Gaussian primary
        # beam. Lastly, we also reduce them if bright sources are peeled
        total_time_hr = 0.0
        for obs in self.observations:
            # Find total observation time in hours
            total_time_hr += (obs.endtime - obs.starttime) / 3600.0
        scaling_factor = np.sqrt(float(tot_bandwidth / 2e6) * total_time_hr / 16.0)
        min_dist_deg, max_dist_deg = self.get_distance_to_obs_center()
        sens_factor = np.e**(-4.0 * np.log(2.0) * min_dist_deg**2 / self.field.fwhm_deg**2)
        self.wsclean_niter = int(1e7)  # set to high value and just use nmiter to limit clean
        self.wsclean_nmiter = min(self.max_nmiter, max(2, int(round(8 * scaling_factor * sens_factor))))
        if self.field.peel_bright_sources:
            # If bright sources are peeled, reduce nmiter by 25% (since they no longer
            # need to be cleaned)
            self.wsclean_niter = int(1e7)  # set to high value and just use nmiter to limit clean
            self.wsclean_nmiter = max(2, int(round(self.wsclean_nmiter * 0.75)))

        # Set multiscale clean
        self.multiscale = do_multiscale
        if self.multiscale:
            self.wsclean_niter = int(self.wsclean_niter/1.5)  # fewer iterations are needed
            self.log.debug("Will do multiscale cleaning.")

        # Set the observation-specific parameters
        max_peak_smearing = imaging_parameters['max_peak_smearing']
        for obs in self.observations:
            # Set imaging parameters
            obs.set_imaging_parameters(self.name, self.cellsize_arcsec, max_peak_smearing,
                                       self.width_ra, self.width_dec,
                                       self.target_fast_timestep, self.target_slow_timstep,
                                       self.target_slow_freqstep, preapply_dde_solutions)

        # Set BL-dependent averaging parameters
        do_bl_averaging = False  # does not yet work with IDG
        if do_bl_averaging:
            timestep_sec = (self.observations[0].timepersample *
                            self.observations[0].parameters['image_timestep'])
            self.wsclean_nwavelengths = self.get_nwavelengths(self.cellsize_deg,
                                                              timestep_sec)
        else:
            self.wsclean_nwavelengths = 0

    def get_nwavelengths(self, cellsize_deg, timestep_sec):
        """
        Returns nwavelengths for WSClean BL-based averaging

        The value depends on the integration time given the specified maximum
        allowed smearing. We scale it from the imaging cell size assuming normal
        sampling as:

        max baseline in nwavelengths = 1 / theta_rad ~= 1 / (cellsize_deg * 3 * pi / 180)
        nwavelengths = max baseline in nwavelengths * 2 * pi * integration time in seconds / (24 * 60 * 60) / 4

        Parameters
        ----------
        cellsize_deg : float
            Pixel size of image in degrees
        timestep_sec : float
            Length of one timestep in seconds

        """
        max_baseline = 1 / (3 * cellsize_deg * np.pi / 180)
        wsclean_nwavelengths_time = int(max_baseline * 2*np.pi * timestep_sec /
                                        (24 * 60 * 60) / 4)
        return wsclean_nwavelengths_time

    def make_skymodel(self, index):
        """
        Makes predict sky model

        Parameters
        ----------
        index : int
            Processing cycle index
        """
        # First check whether sky model already exists due to a previous run and attempt
        # to load it if so
        dst_dir = os.path.join(self.field.working_dir, 'skymodels', 'predict_{}'.format(index))
        os.makedirs(dst_dir, exist_ok=True)
        self.predict_skymodel_file = os.path.join(dst_dir, '{}_predict_skymodel.txt'.format(self.name))
        if os.path.exists(self.predict_skymodel_file):
            skymodel = lsmtool.load(str(self.predict_skymodel_file))
        else:
            # If sky model does not already exist, make it
            if self.is_outlier or self.is_bright_source or self.is_predict:
                # For outlier, bright-source, and predict sectors, we use the sky model
                # made earlier, with no filtering
                skymodel = self.predict_skymodel
            else:
                # For imaging sectors, we use the full calibration sky model and filter it
                # to keep only sources inside the sector
                skymodel = self.calibration_skymodel.copy()
                skymodel = self.filter_skymodel(skymodel)

            # Remove the bright sources from the sky model if they will be predicted and
            # subtracted separately (so that they aren't subtracted twice)
            if (self.field.peel_bright_sources and
                    not self.is_outlier and
                    not self.is_bright_source and
                    not self.is_predict):
                source_names = skymodel.getColValues('Name')
                bright_source_names = self.field.bright_source_skymodel.getColValues('Name')
                matching_ind = []
                for i, sn in enumerate(source_names):
                    if sn in bright_source_names:
                        matching_ind.append(i)
                if len(matching_ind) > 0:
                    skymodel.remove(np.array(matching_ind))

            # Write filtered sky model to file for later prediction
            if len(skymodel) > 0:
                skymodel.write(self.predict_skymodel_file, clobber=True)
            else:
                # No sources, so just make a dummy sky model with single,
                # very faint source at center
                dummylines = ["Format = Name, Type, Patch, Ra, Dec, I, SpectralIndex, LogarithmicSI, "
                              "ReferenceFrequency='100000000.0', MajorAxis, MinorAxis, Orientation\n"]
                coord_strings = lsmtool.utils.format_coordinates(self.ra, self.dec, precision=6)
                patch = self.calibration_skymodel.getPatchNames()[0]
                dummylines.append(',,{0},{1},{2}\n'.format(patch, *coord_strings))
                dummylines.append('s0c0,POINT,{0},{1},{2},0.00000001,'
                                  '[0.0,0.0],false,100000000.0,,,\n'.format(patch, *coord_strings))
                with open(self.predict_skymodel_file, 'w') as f:
                    f.writelines(dummylines)
                skymodel = lsmtool.load(str(self.predict_skymodel_file))

        # Save list of patches (directions) in the format written by DDECal in the h5parm
        self.patches = ['[{}]'.format(p) for p in skymodel.getPatchNames()]

        # Find nearest patch to flux-weighted center of the sector sky model
        if not self.is_outlier and not self.is_bright_source and not self.is_predict:
            tmp_skymodel = skymodel.copy()
            tmp_skymodel.group('single')
            ra, dec = tmp_skymodel.getPatchPositions(method='wmean', asArray=True)
            patch_dist = skymodel.getDistance(ra[0], dec[0], byPatch=True).tolist()
            patch_names = skymodel.getPatchNames()
            self.central_patch = patch_names[patch_dist.index(min(patch_dist))]

            # Filter the field source sky model and store source sizes
            all_source_names = self.field.source_skymodel.getColValues('Name').tolist()
            source_names = skymodel.getColValues('Name')
            if len(source_names) == 1 and source_names[0] not in all_source_names:
                # This occurs when a dummy sky model was made above, so skip the size
                # determination below
                source_skymodel = []
            else:
                in_sector = np.array([all_source_names.index(sn) for sn in source_names if sn in all_source_names])
                source_skymodel = self.field.source_skymodel.copy()
                source_skymodel.select(in_sector)
            if len(source_skymodel) > 0:
                self.source_sizes = source_skymodel.getPatchSizes(units='degree')
            else:
                self.source_sizes = [0.0]

        # Set the parameters for predict
        self.set_prediction_parameters()

    def filter_skymodel(self, skymodel, invert=False):
        """
        Filters input skymodel to select only sources that lie inside the sector

        Parameters
        ----------
        skymodel : LSMTool skymodel object
            Input sky model
        invert : bool, optional
            If True, invert the selection (so select only sources that lie outside
            the sector)

        Returns
        -------
        filtered_skymodel : LSMTool skymodel object
            Filtered sky model
        """
        return facet.filter_skymodel(self.poly, skymodel, self.field.wcs, invert=invert)

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
            List of parameters, with one entry for each observation
        """
        return [obs.parameters[parameter] for obs in self.observations]

    def intialize_vertices(self):
        """
        Determines the vertices of the sector polygon
        """
        # Define initial polygon as a rectangle
        x_sector_pixels, y_sector_pixels = self.field.wcs.wcs_world2pix(
            self.ra, self.dec, misc.WCS_ORIGIN
        )
        ra_width_pix = self.width_ra / abs(self.field.wcs.wcs.cdelt[0])
        dec_width_pix = self.width_dec / abs(self.field.wcs.wcs.cdelt[1])
        x0 = x_sector_pixels - ra_width_pix / 2.0
        y0 = y_sector_pixels - dec_width_pix / 2.0
        poly_verts = [(x0, y0), (x0, y0+dec_width_pix),
                      (x0+ra_width_pix, y0+dec_width_pix),
                      (x0+ra_width_pix, y0), (x0, y0)]
        poly = Polygon(poly_verts)

        # Save initial polygon, copy of initial polygon (which potentially will be
        # altered later for source avoidance), and buffered version of initial polygon
        # (which includes the padding done by WSClean)
        self.initial_poly = poly
        self.poly = Polygon(poly)
        padding_pix = dec_width_pix*(self.wsclean_image_padding - 1.0)
        self.poly_padded = self.poly.buffer(padding_pix)

    def get_vertices_radec(self):
        """
        Return the vertices as RA, Dec for the sector boundary
        """
        return self.field.wcs.wcs_pix2world(self.poly.exterior.coords.xy[0],
                                            self.poly.exterior.coords.xy[1],
                                            misc.WCS_ORIGIN)

    def make_vertices_file(self):
        """
        Make a vertices file for the sector boundary
        """
        vertices = self.get_vertices_radec()

        with open(self.vertices_file, 'wb') as f:
            pickle.dump(vertices, f)

    def make_region_file(self, outputfile, region_format='ds9'):
        """
        Make a ds9 or CASA region file for the sector boundary

        Parameters
        ----------
        outputfile : str
            Name of output region file
        region_format : str, optional
            Format of region file: 'ds9' or 'casa'
        """
        vertices = self.get_vertices_radec()

        if region_format == 'casa':
            lines = ['#CRTFv0\n\n']
            xylist = []
            RAs = vertices[0][0:-1]  # trim last point, as it is a repeat of the first
            Decs = vertices[1][0:-1]
            for x, y in zip(RAs, Decs):
                xylist.append('[{0}deg, {1}deg]'.format(x, y))
            lines.append('poly[{0}]\n'.format(', '.join(xylist)))

            with open(outputfile, 'w') as f:
                f.writelines(lines)
        elif region_format == 'ds9':
            lines = []
            lines.append('# Region file format: DS9 version 4.0\nglobal color=green '
                         'font="helvetica 10 normal" select=1 highlite=1 edit=1 '
                         'move=1 delete=1 include=1 fixed=0 source=1\nfk5\n')
            xylist = []
            RAs = vertices[0]
            Decs = vertices[1]
            for x, y in zip(RAs, Decs):
                xylist.append('{0}, {1}'.format(x, y))
            lines.append('polygon({0})\n'.format(', '.join(xylist)))
            lines.append('point({0}, {1}) # point=cross width=2 text={{{2}}}\n'.
                         format(self.ra, self.dec, self.name))

            with open(outputfile, 'w') as f:
                f.writelines(lines)
        else:
            self.log.error('Region format not understood.')

    def get_matplotlib_patch(self, wcs=None):
        """
        Returns a matplotlib patch for the sector polygon

        Parameters
        ----------
        wcs : WCS object, optional
            WCS object defining (RA, Dec) <-> (x, y) transformation. If not given,
            the field's transformation is used

        Returns
        -------
        patch : matplotlib patch object
            The patch for the sector polygon
        """
        if wcs is not None:
            vertices = self.field.wcs.wcs_pix2world(self.poly.exterior.coords.xy[0],
                                                    self.poly.exterior.coords.xy[1],
                                                    misc.WCS_ORIGIN)
            x, y = wcs.wcs_world2pix(vertices[0], vertices[1], misc.WCS_ORIGIN)
        else:
            x, y = self.poly.exterior.coords.xy

        xy = np.vstack([x, y]).transpose()
        patch = patches.Polygon(xy=xy, label=self.name, edgecolor='k', facecolor='none',
                                linewidth=2)

        return patch

    def get_distance_to_obs_center(self):
        """
        Return the overall minimum and maximum distance in degrees from any sector vertex
        (and sector center) to the phase center of the observation

        Returns
        -------
        min_dist, max_dist : float, float
            Minimum and maximum distance in degrees
        """
        obs_coord = SkyCoord(self.observations[0].ra, self.observations[0].dec,
                             unit=(u.degree, u.degree), frame='fk5')

        # Calculate the distance from each vertex to the observation phase center
        vertices = self.get_vertices_radec()
        RAs = vertices[0]
        Decs = vertices[1]
        distances = []
        for ra, dec in zip(RAs, Decs):
            ra_norm, dec_norm = misc.normalize_ra_dec(ra, dec)
            coord = SkyCoord(ra_norm, dec_norm, unit=(u.degree, u.degree), frame='fk5')
            distances.append(obs_coord.separation(coord).value)

        # Also calculate the distance to the sector center
        coord = SkyCoord(self.ra, self.dec, unit=(u.degree, u.degree), frame='fk5')
        distances.append(obs_coord.separation(coord).value)

        return np.min(distances), np.max(distances)
