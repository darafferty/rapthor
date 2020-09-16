"""
Definition of the Sector class that holds parameters for an image or predict sector
"""
import logging
import numpy as np
from rapthor.lib import miscellaneous as misc
from astropy.coordinates import Angle
from shapely.geometry import Point, Polygon
from shapely.prepared import prep
from PIL import Image, ImageDraw
import pickle
import os
import copy


class Sector(object):
    """
    The Sector object contains various parameters for a sector of the field. Sectors
    are used only in image and predict pipelines

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
        self.ra = ra
        self.dec = dec
        self.width_ra = width_ra
        self.width_dec = width_dec
        self.field = field
        self.vertices_file = os.path.join(field.working_dir, 'regions', '{}_vertices.pkl'.format(self.name))
        self.region_file = "'[]'"
        self.I_image_file_true_sky = None  # set by the Image operation
        self.I_image_file_apparent_sky = None  # set by the Image operation
        self.I_mask_file = None   # set by the Image operation
        self.image_skymodel_file_apparent_sky = None  # set by the Image operation
        self.image_skymodel_file_true_sky = None  # set by the Image operation
        self.is_outlier = False
        self.is_bright_source = False
        self.imsize = None  # set to None to force calculation in set_imaging_parameters()

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
            obs.set_prediction_parameters(self.name, self.patches,
                                          os.path.join(self.field.working_dir, 'scratch'))

    def set_imaging_parameters(self, do_multiscale=None, recalculate_imsize=False):
        """
        Sets the parameters needed for the imaging pipeline

        Parameters
        ----------
        do_multiscale : bool, optional
            If True, multiscale clean is done. If None, multiscale clean is done only
            when a large source is detected
        recalculate_imsize : bool, optional
            If True, the image size is recalculated based on the current sector region
        """
        self.cellsize_arcsec = self.field.parset['imaging_specific']['cellsize_arcsec']
        self.cellsize_deg = self.cellsize_arcsec / 3600.0
        self.robust = self.field.parset['imaging_specific']['robust']
        self.taper_arcsec = self.field.parset['imaging_specific']['taper_arcsec']
        self.min_uv_lambda = self.field.parset['imaging_specific']['min_uv_lambda']
        self.max_uv_lambda = self.field.parset['imaging_specific']['max_uv_lambda']
        self.idg_mode = self.field.parset['imaging_specific']['idg_mode']
        self.reweight = self.field.parset['imaging_specific']['reweight']
        self.wsclean_image_padding = self.field.parset['imaging_specific']['wsclean_image_padding']
        self.flag_abstime = self.field.parset['flag_abstime']
        self.flag_baseline = self.field.parset['flag_baseline']
        self.flag_freqrange = self.field.parset['flag_freqrange']
        self.flag_expr = self.field.parset['flag_expr']
        self.target_fast_timestep = self.field.parset['calibration_specific']['fast_timestep_sec']
        self.target_slow_freqstep = self.field.parset['calibration_specific']['slow_freqstep_hz']
        self.use_screens = self.field.use_screens
        self.nmiter = 10
        self.auto_mask = 3.0
        self.threshisl = 4.0
        self.threshpix = 5.0

        # Set image size based on current sector polygon
        if recalculate_imsize or self.imsize is None:
            xmin, ymin, xmax, ymax = self.poly.bounds
            self.width_ra = (xmax - xmin) * self.field.wcs_pixel_scale  # deg
            self.width_dec = (ymax - ymin) * self.field.wcs_pixel_scale  # deg
            self.imsize = [int(self.width_ra / self.cellsize_deg * 1.1),
                           int(self.width_dec / self.cellsize_deg * 1.1)]

            # IDG does not yet support rectangular images, so ensure image is square
            self.imsize = [max(self.imsize), max(self.imsize)]

            # IDG has problems with small images, so set minimum size to 500 pixels and adjust
            # padded polygon
            minsize = 500
            if max(self.imsize) < minsize:
                dec_width_pix = self.width_dec / abs(self.field.wcs.wcs.cdelt[1])
                padding_pix = dec_width_pix * (self.wsclean_image_padding - 1.0)
                padding_pix *= minsize / max(self.imsize)  # scale padding to new imsize
                self.poly_padded = self.poly.buffer(padding_pix)
                self.imsize = [minsize, minsize]
        self.wsclean_imsize = "'{0} {1}'".format(self.imsize[0], self.imsize[1])
        self.log.debug('Image size is {0} x {1} pixels'.format(
                       self.imsize[0], self.imsize[1]))

        # Set number of output channels to get ~ 4 MHz per channel equivalent at 120 MHz
        # (the maximum averaging allowed for typical dTEC values of -0.5 < dTEC < 0.5)
        min_freq = np.min([obs.startfreq for obs in self.observations])
        target_bandwidth = 4e6 * min_freq / 120e6
        min_nchannels = 4
        tot_bandwidth = 0.0
        for obs in self.observations:
            # Find observation with largest bandwidth
            obs_bandwidth = obs.numchannels * obs.channelwidth
            if obs_bandwidth > tot_bandwidth:
                tot_bandwidth = obs_bandwidth
        self.wsclean_nchannels = max(min_nchannels, int(np.ceil(tot_bandwidth / target_bandwidth)))

        # Set number of iterations. We scale the number of iterations depending on the
        # integration time and the distance of the sector center to the phase center, to
        # account for the reduced sensitivity of the image, assuming a Gaussian primary
        # beam. Lastly, we also reduce them if bright sources are peeled
        total_time_hr = 0.0
        for obs in self.observations:
            # Find total observation time in hours
            total_time_hr += (obs.endtime - obs.starttime) / 3600.0
        scaling_factor = np.sqrt(np.float(tot_bandwidth / 2e6) * total_time_hr / 8.0)
        dist_deg = np.min(self.get_distance_to_obs_center())
        sens_factor = np.e**(-4.0 * np.log(2.0) * dist_deg**2 / self.field.fwhm_deg**2)
        self.wsclean_niter = int(round(12000 * scaling_factor * sens_factor))
        self.wsclean_nmiter = min(12, max(2, int(round(8 * scaling_factor * sens_factor))))
        if self.field.peel_bright_sources:
            self.wsclean_niter = int(round(self.wsclean_niter * 0.75))
            self.wsclean_nmiter = min(12, max(2, int(round(self.wsclean_nmiter * 0.75))))

        # Set multiscale: get source sizes and check for large sources
        self.multiscale = do_multiscale
        if self.multiscale is None:
            # TODO: figure out good way to determine whether multiscale should be used
            # and the scales, maybe using the presence of Gaussians on larger wavelet
            # scales? For now, force it to off, as it takes a long time and so should
            # only be used when necessary
            self.multiscale = False

#             largest_scale = np.max(self.source_sizes) / self.cellsize_deg / 3.0
#             large_size_arcmin = 4.0  # threshold source size for multiscale to be activated
#             sizes_arcmin = self.source_sizes * 60.0
#             if sizes_arcmin is not None and any([s > large_size_arcmin for s in sizes_arcmin]):
#                 self.multiscale = True
#             else:
#                 self.multiscale = False
        if self.multiscale:
            self.multiscale_scales_pixel = self.field.parset['imaging_specific']['multiscale_scales_pixel']
#             if self.multiscale_scales_pixel is None:
#                 largest_scale = np.max(self.source_sizes) / self.cellsize_deg / 3.0
#                 if largest_scale < 3:
#                     self.multiscale_scales_pixel = [0]
#                 elif largest_scale < 5:
#                     self.multiscale_scales_pixel = [0, 3]
#                 elif largest_scale < 15:
#                     self.multiscale_scales_pixel = [0, 6, 12]
#                 else:
#                     self.multiscale_scales_pixel = None  # let WSClean decide
            self.wsclean_niter = int(self.wsclean_niter/1.5)  # fewer iterations are needed
            self.log.debug("Will do multiscale cleaning.")
        else:
            self.multiscale_scales_pixel = 0

        # Set the observation-specific parameters
        max_peak_smearing = self.field.parset['imaging_specific']['max_peak_smearing']
        for obs in self.observations:
            # Set imaging parameters
            obs.set_imaging_parameters(self.cellsize_arcsec, max_peak_smearing,
                                       self.width_ra, self.width_dec,
                                       self.target_fast_timestep, self.target_slow_freqstep,
                                       self.use_screens)

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

    def make_skymodel(self, iter):
        """
        Makes predict sky model

        Parameters
        ----------
        iter : int
            Iteration index
        """
        if self.is_outlier or self.is_bright_source:
            # For outlier and bright-source sectors, we use the sky model made earlier,
            # with no filtering
            skymodel = self.predict_skymodel
        else:
            # For imaging sectors, we use the full calibration sky model and filter it
            # to keep only sources inside the sector
            skymodel = self.calibration_skymodel.copy()
            skymodel = self.filter_skymodel(skymodel)

        # Remove the bright sources from the sky model if they will be predicted and
        # subtracted separately (so that they aren't subtracted twice)
        if self.field.peel_bright_sources and not self.is_outlier and not self.is_bright_source:
            source_names = skymodel.getColValues('Name')
            bright_source_names = self.field.bright_source_skymodel.getColValues('Name')
            matching_ind = []
            for i, sn in enumerate(source_names):
                if sn in bright_source_names:
                    matching_ind.append(i)
            if len(matching_ind) > 0:
                skymodel.remove(np.array(matching_ind))

        # Write filtered sky model to file for later prediction
        dst_dir = os.path.join(self.field.working_dir, 'skymodels', 'predict_{}'.format(iter))
        misc.create_directory(dst_dir)
        self.predict_skymodel_file = os.path.join(dst_dir, '{}_predict_skymodel.txt'.format(self.name))
        skymodel.write(self.predict_skymodel_file, clobber=True)

        # Save list of patches (directions) in the format written by DDECal in the h5parm
        self.patches = ['[{}]'.format(p) for p in skymodel.getPatchNames()]

        # Find nearest patch to flux-weighted center of the sector sky model
        if not self.is_outlier and not self.is_bright_source:
            tmp_skymodel = skymodel.copy()
            tmp_skymodel.group('single')
            ra, dec = tmp_skymodel.getPatchPositions(method='wmean', asArray=True)
            patch_dist = skymodel.getDistance(ra[0], dec[0], byPatch=True).tolist()
            patch_names = skymodel.getPatchNames()
            self.central_patch = patch_names[patch_dist.index(min(patch_dist))]

            # Filter the field source sky model and store source sizes
            all_source_names = self.field.source_skymodel.getColValues('Name').tolist()
            source_names = skymodel.getColValues('Name')
            in_sector = np.array([all_source_names.index(sn) for sn in source_names])
            source_skymodel = self.field.source_skymodel.copy()
            source_skymodel.select(in_sector)
            if len(source_skymodel) > 0:
                self.source_sizes = source_skymodel.getPatchSizes(units='degree')
            else:
                self.source_sizes = [0.0]

        # Set the parameters for predict
        self.set_prediction_parameters()

    def filter_skymodel(self, skymodel):
        """
        Filters input skymodel to select only sources that lie inside the sector

        Parameters
        ----------
        skymodel : LSMTool skymodel object
            Input sky model

        Returns
        -------
        filtered_skymodel : LSMTool skymodel object
            Filtered sky model
        """
        # Make list of sources
        RA = skymodel.getColValues('Ra')
        Dec = skymodel.getColValues('Dec')
        x, y = self.field.radec2xy(RA, Dec)
        x = np.array(x)
        y = np.array(y)

        # Keep only those sources inside the sector bounding box
        inside = np.zeros(len(skymodel), dtype=bool)
        xmin, ymin, xmax, ymax = self.poly.bounds
        inside_ind = np.where((x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax))
        inside[inside_ind] = True
        skymodel.select(inside)
        if len(skymodel) == 0:
            return skymodel
        RA = skymodel.getColValues('Ra')
        Dec = skymodel.getColValues('Dec')
        x, y = self.field.radec2xy(RA, Dec)
        x = np.array(x)
        y = np.array(y)

        # Now check the actual sector boundary against filtered sky model
        xpadding = max(int(0.1 * (max(x) - min(x))), 3)
        ypadding = max(int(0.1 * (max(y) - min(y))), 3)
        xshift = int(min(x)) - xpadding
        yshift = int(min(y)) - ypadding
        xsize = int(np.ceil(max(x) - min(x))) + 2*xpadding
        ysize = int(np.ceil(max(y) - min(y))) + 2*ypadding
        x -= xshift
        y -= yshift
        prepared_polygon = prep(self.poly)

        # Unmask everything outside of the polygon + its border (outline)
        inside = np.zeros(len(skymodel), dtype=bool)
        mask = Image.new('L', (xsize, ysize), 0)
        verts = [(xv-xshift, yv-yshift) for xv, yv in zip(self.poly.exterior.coords.xy[0],
                                                          self.poly.exterior.coords.xy[1])]
        ImageDraw.Draw(mask).polygon(verts, outline=1, fill=1)
        inside_ind = np.where(np.array(mask).transpose()[(x.astype(int), y.astype(int))])
        inside[inside_ind] = True

        # Now check sources in the border precisely
        mask = Image.new('L', (xsize, ysize), 0)
        ImageDraw.Draw(mask).polygon(verts, outline=1, fill=0)
        border_ind = np.where(np.array(mask).transpose()[(x.astype(int), y.astype(int))])
        points = [Point(xs, ys) for xs, ys in zip(x[border_ind], y[border_ind])]
        for i, p in enumerate(points):
            p.index = border_ind[0][i]
        outside_points = [v for v in points if not prepared_polygon.contains(v)]
        for outside_point in outside_points:
            inside[outside_point.index] = False
        skymodel.select(inside)
        return skymodel

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
        sx, sy = self.field.radec2xy([self.ra], [self.dec])
        ra_width_pix = self.width_ra / abs(self.field.wcs.wcs.cdelt[0])
        dec_width_pix = self.width_dec / abs(self.field.wcs.wcs.cdelt[1])
        x0 = sx[0] - ra_width_pix / 2.0
        y0 = sy[0] - dec_width_pix / 2.0
        poly_verts = [(x0, y0), (x0, y0+dec_width_pix),
                      (x0+ra_width_pix, y0+dec_width_pix),
                      (x0+ra_width_pix, y0), (x0, y0)]
        poly = Polygon(poly_verts)

        # Save initial polygon, copy of initial polygon (which potentially will be
        # altered later for source avoidance), and buffered version of initial polygon
        # (which includes the padding done by WSClean, needed for aterm generation)
        self.initial_poly = poly
        self.poly = Polygon(poly)
        padding_pix = dec_width_pix*(self.field.parset['imaging_specific']['wsclean_image_padding'] - 1.0)
        self.poly_padded = self.poly.buffer(padding_pix)

    def get_vertices_radec(self):
        """
        Return the vertices as RA, Dec for the sector boundary
        """
        ra, dec = self.field.xy2radec(self.poly.exterior.coords.xy[0].tolist(),
                                      self.poly.exterior.coords.xy[1].tolist())
        vertices = [np.array(ra), np.array(dec)]

        return vertices

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

    def get_distance_to_obs_center(self):
        """
        Return the distance in degrees to the phase center of the observation(s)

        Returns
        -------
        distance : list
            List of distances: [center, lower-left corner, upper-left corner,
                                lower-right corner, upper-right corner]
        """
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        coordc = SkyCoord(self.ra, self.dec, unit=(u.degree, u.degree), frame='fk5')
        coordll = SkyCoord(self.ra+self.width_ra/2.0, self.dec-self.width_dec/2.0, unit=(u.degree, u.degree), frame='fk5')
        coordul = SkyCoord(self.ra+self.width_ra/2.0, self.dec+self.width_dec/2.0, unit=(u.degree, u.degree), frame='fk5')
        coordlr = SkyCoord(self.ra-self.width_ra/2.0, self.dec-self.width_dec/2.0, unit=(u.degree, u.degree), frame='fk5')
        coordur = SkyCoord(self.ra-self.width_ra/2.0, self.dec+self.width_dec/2.0, unit=(u.degree, u.degree), frame='fk5')
        coord2 = SkyCoord(self.observations[0].ra, self.observations[0].dec, unit=(u.degree, u.degree), frame='fk5')

        return [coordc.separation(coord2).value, coordll.separation(coord2).value,
                coordul.separation(coord2).value, coordlr.separation(coord2).value,
                coordur.separation(coord2).value]
