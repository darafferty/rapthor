"""
Module that holds functions and classes related to faceting
"""
import numpy as np
import scipy as sp
import scipy.spatial
from shapely.geometry import Point, Polygon
from shapely.prepared import prep
from PIL import Image, ImageDraw
from astropy.coordinates import Angle
import ast
import logging
from rapthor.lib import miscellaneous as misc
from matplotlib import patches
import lsmtool
import tempfile


class Facet(object):
    """
    Base Facet class

    Parameters
    ----------
    name : str
        Name of facet
    ra : float
        RA in degrees of reference coordinate
    dec : float
        Dec in degrees of reference coordinate
    vertices : list of tuples
        List of (RA, Dec) tuples, one for each vertex of the facet
    """
    def __init__(self, name, ra, dec, vertices):
        self.name = name
        self.log = logging.getLogger('rapthor:{0}'.format(self.name))
        if type(ra) is str:
            ra = Angle(ra).to('deg').value
        if type(dec) is str:
            dec = Angle(dec).to('deg').value
        self.ra = misc.normalize_ra(ra)
        self.dec = misc.normalize_dec(dec)
        self.vertices = np.array(vertices)

        # Convert input (RA, Dec) vertices to (x, y) polygon
        self.wcs = make_wcs(self.ra, self.dec)
        ra_values = [ra for ra in self.vertices[::2]]
        dec_values = [dec for dec in vertices[1::2]]
        x_values, y_values = radec2xy(self.wcs, ra_values, dec_values)
        polygon_vertices = [(x, y) for x, y in zip(x_values, y_values)]
        self.polygon = Polygon(polygon_vertices)

        # Find the size and center coordinates of the facet
        xmin, ymin, xmax, ymax = self.polygon.bounds
        self.size = min(0.5, 1.2 * max(xmax-xmin, ymax-ymin) *
                        abs(self.wcs.wcs.cdelt[0]))  # degrees
        self.x_center = xmin + (xmax - xmin)/2
        self.y_center = ymin + (ymax - ymin)/2
        ra_center, dec_center = xy2radec(self.wcs, [self.x_center], [self.y_center])
        self.ra_center = ra_center[0]
        self.dec_center = dec_center[0]

    def set_skymodel(self, skymodel):
        """
        Sets the facet's sky model

        The input sky model is filtered to contain only those sources that lie
        inside the facet's polygon. The filtered sky model is stored in
        self.skymodel

        Parameters
        ----------
        skymodel : LSMTool skymodel object
            Input sky model
        """
        self.skymodel = filter_skymodel(self.polygon, skymodel, self.wcs)

    def download_panstarrs(self, max_search_cone_radius=0.5):
        """
        Returns a Pan-STARRS sky model for the area around the facet

        Note: the resulting sky model may contain sources outside the facet's
        polygon

        Parameters
        ----------
        max_search_cone_radius : float, optional
            The maximum radius in degrees to use in the cone search. The smaller
            of this radius and the minimum radius that covers the facet is used

        Returns
        -------
        skymodel : LSMTool skymodel object
            The Pan-STARRS sky model
        """
        try:
            with tempfile.NamedTemporaryFile() as fp:
                misc.download_skymodel(self.ra_center, self.dec_center, fp.name,
                                       radius=min(max_search_cone_radius, self.size/2),
                                       overwrite=True, source='PANSTARRS')
                skymodel = lsmtool.load(fp.name)
        except IOError:
            # Comparison catalog not downloaded successfully
            skymodel = lsmtool.makeEmptyTable()

        return skymodel

    def find_astrometry_offsets(self, comparison_skymodel=None, min_number=10):
        """
        Finds the astrometry offsets for sources in the facet

        The offsets are stored in self.astrometry_diagnostics, a dict with
        the following keys (see LSMTool's compare operation for details of the
        diagnostics):

            'meanRAOffsetDeg', 'stdRAOffsetDeg', 'meanClippedRAOffsetDeg',
            'stdClippedRAOffsetDeg', 'meanDecOffsetDeg', 'stdDecOffsetDeg',
            'meanClippedDecOffsetDeg', 'stdClippedDecOffsetDeg'

        Note: if the comparison is unsuccessful, self.astrometry_diagnostics is
        an empty dict

        Parameters
        ----------
        comparison_skymodel : LSMTool skymodel object, optional
            Comparison sky model
        min_number : int, optional
            Minimum number of sources required for comparison
        """
        self.astrometry_diagnostics = {}
        if comparison_skymodel is None:
            comparison_skymodel = self.download_panstarrs()

        # Find the astrometry offsets between the facet's sky model and the
        # comparison sky model
        #
        # Note: If there are no successful matches, the compare() method
        # returns None
        if comparison_skymodel and len(comparison_skymodel) >= min_number:
            result = self.skymodel.compare(comparison_skymodel,
                                           radius='5 arcsec',
                                           excludeMultiple=True,
                                           make_plots=False)
            # Save offsets
            if result is not None:
                self.astrometry_diagnostics.update(result)

    def get_matplotlib_patch(self, reference_pos=None):
        """
        Returns a matplotlib patch for the facet polygon

        Parameters
        ----------
        reference_pos : list, optional
            The reference [RA, Dec] of the parent field in degrees

        Returns
        -------
        patch : matplotlib patch object
            The patch for the facet polygon
        """
        if reference_pos is not None:
            ref_x, ref_y = radec2xy(self.wcs, [reference_pos[0]], [reference_pos[1]])
            ref_xy = [ref_x[0], ref_y[0]]
        else:
            ref_xy = [0, 0]

        x = self.polygon.exterior.coords.xy[0] - ref_xy[0]
        y = self.polygon.exterior.coords.xy[1] - ref_xy[1]
        xy = np.vstack([x, y]).transpose()
        patch = patches.Polygon(xy=xy)

        return patch


class SquareFacet(Facet):
    """
    Wrapper class for a square facet

    Parameters
    ----------
    name : str
        Name of facet
    ra : float
        RA in degrees of facet center coordinate
    dec : float
        Dec in degrees of facet center coordinate
    width : float
        Width in degrees of facet
    """
    def __init__(self, name, ra, dec, width):
        if type(ra) is str:
            ra = Angle(ra).to('deg').value
        if type(dec) is str:
            dec = Angle(dec).to('deg').value
        ra = misc.normalize_ra(ra)
        dec = misc.normalize_dec(dec)
        wcs = make_wcs(ra, dec)

        # Make the vertices
        xmin = wcs.wcs.crpix[0] - width / 2 / abs(wcs.wcs.cdelt[0])
        xmax = wcs.wcs.crpix[0] + width / 2 / abs(wcs.wcs.cdelt[0])
        ymin = wcs.wcs.crpix[1] - width / 2 / abs(wcs.wcs.cdelt[1])
        ymax = wcs.wcs.crpix[1] + width / 2 / abs(wcs.wcs.cdelt[1])
        vertices = [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)]

        super().__init__(name, ra, dec, vertices)


def make_facet_polygons(ra_cal, dec_cal, ra_mid, dec_mid, width_ra, width_dec):
    """
    Makes a Voronoi tessellation and returns the resulting facet centers
    and polygons

    Parameters
    ----------
    ra_cal : array
        RA values in degrees of calibration directions
    dec_cal : array
        Dec values in degrees of calibration directions
    ra_mid : float
        RA in degrees of bounding box center
    dec_mid : float
        Dec in degrees of bounding box center
    width_ra : float
        Width of bounding box in RA in degrees, corrected to Dec = 0
    width_dec : float
        Width of bounding box in Dec in degrees

    Returns
    -------
    facet_points, facet_polys : list of tuples, list of arrays
        List of facet points (centers) as (RA, Dec) tuples in degrees and
        list of facet polygons (vertices) as [RA, Dec] arrays in degrees
        (each of shape N x 2, where N is the number of vertices in a given
        facet)
    """
    # Build the bounding box corner coordinates
    if width_ra <= 0.0 or width_dec <= 0.0:
        raise ValueError('The RA/Dec width cannot be zero or less')
    wcs_pixel_scale = 20.0 / 3600.0  # 20"/pixel
    wcs = make_wcs(ra_mid, dec_mid, wcs_pixel_scale)
    x_cal, y_cal = radec2xy(wcs, ra_cal, dec_cal)
    x_mid, y_mid = radec2xy(wcs, [ra_mid], [dec_mid])
    width_x = width_ra / wcs_pixel_scale / 2.0
    width_y = width_dec / wcs_pixel_scale / 2.0
    bounding_box = np.array([x_mid[0] - width_x, x_mid[0] + width_x,
                             y_mid[0] - width_y, y_mid[0] + width_y])

    # Tessellate and convert resulting facet polygons from (x, y) to (RA, Dec)
    vor = voronoi(np.stack((x_cal, y_cal)).T, bounding_box)
    facet_polys = []
    for region in vor.filtered_regions:
        vertices = vor.vertices[region + [region[0]], :]
        ra, dec = xy2radec(wcs, vertices[:, 0], vertices[:, 1])
        vertices = np.stack((ra, dec)).T
        facet_polys.append(vertices)
    facet_points = []
    for point in vor.filtered_points:
        ra, dec = xy2radec(wcs, [point[0]], [point[1]])
        facet_points.append((ra[0], dec[0]))

    return facet_points, facet_polys


def radec2xy(wcs, RA, Dec):
    """
    Returns x, y for input RA, Dec

    Parameters
    ----------
    wcs : WCS object
        WCS object defining transformation
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
        x.append(wcs.wcs_world2pix(ra_dec, 0)[0][0])
        y.append(wcs.wcs_world2pix(ra_dec, 0)[0][1])
    return x, y


def xy2radec(wcs, x, y):
    """
    Returns input RA, Dec for input x, y

    Parameters
    ----------
    wcs : WCS object
        WCS object defining transformation
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
        RA.append(wcs.wcs_pix2world(x_y, 0)[0][0])
        Dec.append(wcs.wcs_pix2world(x_y, 0)[0][1])
    return RA, Dec


def make_wcs(ra, dec, wcs_pixel_scale=10.0/3600.0):
    """
    Makes simple WCS object

    Parameters
    ----------
    ra : float
        Reference RA in degrees
    dec : float
        Reference Dec in degrees
    wcs_pixel_scale : float, optional
        Pixel scale in degrees/pixel (default = 10"/pixel)

    Returns
    -------
    w : astropy.wcs.WCS object
        A simple TAN-projection WCS object for specified reference position
    """
    from astropy.wcs import WCS

    w = WCS(naxis=2)
    w.wcs.crpix = [1000, 1000]
    w.wcs.cdelt = np.array([-wcs_pixel_scale, wcs_pixel_scale])
    w.wcs.crval = [ra, dec]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.set_pv([(2, 1, 45.0)])
    return w


def in_box(cal_coords, bounding_box):
    """
    Checks if coordinates are inside the bounding box

    Parameters
    ----------
    cal_coords : array
        Array of x, y coordinates
    bounding_box : array
        Array defining the bounding box as [minx, maxx, miny, maxy]

    Returns
    -------
    inside : array
        Bool array with True for inside and False if not
    """
    return np.logical_and(np.logical_and(bounding_box[0] <= cal_coords[:, 0],
                                         cal_coords[:, 0] <= bounding_box[1]),
                          np.logical_and(bounding_box[2] <= cal_coords[:, 1],
                                         cal_coords[:, 1] <= bounding_box[3]))


def voronoi(cal_coords, bounding_box):
    """
    Produces a Voronoi tessellation for the given coordinates and bounding box

    Parameters
    ----------
    cal_coords : array
        Array of x, y coordinates
    bounding_box : array
        Array defining the bounding box as [minx, maxx, miny, maxy]

    Returns
    -------
    vor : Voronoi object
        The resulting Voronoi object
    """
    eps = 1e-6

    # Select calibrators inside the bounding box
    i = in_box(cal_coords, bounding_box)

    # Mirror points
    points_center = cal_coords[i, :]
    points_left = np.copy(points_center)
    points_left[:, 0] = bounding_box[0] - (points_left[:, 0] - bounding_box[0])
    points_right = np.copy(points_center)
    points_right[:, 0] = bounding_box[1] + (bounding_box[1] - points_right[:, 0])
    points_down = np.copy(points_center)
    points_down[:, 1] = bounding_box[2] - (points_down[:, 1] - bounding_box[2])
    points_up = np.copy(points_center)
    points_up[:, 1] = bounding_box[3] + (bounding_box[3] - points_up[:, 1])
    points = np.append(points_center,
                       np.append(np.append(points_left,
                                           points_right,
                                           axis=0),
                                 np.append(points_down,
                                           points_up,
                                           axis=0),
                                 axis=0),
                       axis=0)

    # Compute Voronoi, sorting the output regions to match the order of the
    # input coordinates
    vor = sp.spatial.Voronoi(points)
    sorted_regions = np.array(vor.regions, dtype=object)[np.array(vor.point_region)]
    vor.regions = sorted_regions.tolist()

    # Filter regions
    regions = []
    for region in vor.regions:
        flag = True
        for index in region:
            if index == -1:
                flag = False
                break
            else:
                x = vor.vertices[index, 0]
                y = vor.vertices[index, 1]
                if not(bounding_box[0] - eps <= x and x <= bounding_box[1] + eps and
                       bounding_box[2] - eps <= y and y <= bounding_box[3] + eps):
                    flag = False
                    break
        if region and flag:
            regions.append(region)
    vor.filtered_points = points_center
    vor.filtered_regions = regions

    return vor


def make_ds9_region_file(center_coords, facet_polygons, outfile, names=None):
    """
    Make a ds9 region file for given polygons and centers

    Parameters
    ----------
    center_coords : list of tuples
        List of (RA, Dec) of the facet center coordinates
    facet_polygons : list of arrays
        List of [RA, Dec] arrays of the facet vertices (each of shape N x 2, where N
        is the number of vertices in a given facet)
    outfile : str
        Name of output region file
    names : list, optional
        List of names of the facets
    """
    lines = []
    lines.append('# Region file format: DS9 version 4.0\nglobal color=green '
                 'font="helvetica 10 normal" select=1 highlite=1 edit=1 '
                 'move=1 delete=1 include=1 fixed=0 source=1\nfk5\n')
    if names is None:
        names = [None] * len(center_coords)
    if not (len(names) == len(center_coords) == len(facet_polygons)):
        raise ValueError('Input lists of facet coordinates, vertices, and names '
                         'must have the same length')
    for name, center_coord, vertices in zip(names, center_coords, facet_polygons):
        radec_list = []
        RAs = vertices.T[0]
        Decs = vertices.T[1]
        for ra, dec in zip(RAs, Decs):
            radec_list.append('{0}, {1}'.format(ra, dec))
        lines.append('polygon({0})\n'.format(', '.join(radec_list)))
        if name is None:
            lines.append('point({0}, {1})\n'.format(center_coord[0], center_coord[1]))
        else:
            lines.append('point({0}, {1}) # text={2}\n'.format(center_coord[0], center_coord[1], name))

    with open(outfile, 'w') as f:
        f.writelines(lines)


def read_ds9_region_file(region_file):
    """
    Read a ds9 facet region file and return facets

    Parameters
    ----------
    region_file : str
        Filename of input ds9 region file

    Returns
    -------
    facets : list
        List of Facet objects
    """
    facets = []
    with open(region_file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        # Each facet in the region file is defined by two consecutive lines:
        #   - the first starts with 'polygon' and gives the (RA, Dec) vertices
        #   - the second starts with 'point' and gives the reference (RA, Dec)
        #     and the facet name
        if line.startswith('polygon'):
            vertices = ast.literal_eval(line.split('polygon')[1])
        if line.startswith('point'):
            ra, dec = ast.literal_eval(line.split('point')[1])
            if 'text' in line:
                name = line.split('text=')[1].strip()
            else:
                name = f'facet_{ra}_{dec}'
            facets.append(Facet(name, ra, dec, vertices))

    return facets


def filter_skymodel(polygon, skymodel, wcs):
    """
    Filters input skymodel to select only sources that lie inside the input facet

    Parameters
    ----------
    polygon : Shapely polygon object
        Polygon object to use for filtering
    skymodel : LSMTool skymodel object
        Input sky model to be filtered
    wcs : WCS object
        WCS object defining image to sky transformations

    Returns
    -------
    filtered_skymodel : LSMTool skymodel object
        Filtered sky model
    """
    # Make list of sources
    RA = skymodel.getColValues('Ra')
    Dec = skymodel.getColValues('Dec')
    x, y = radec2xy(wcs, RA, Dec)
    x = np.array(x)
    y = np.array(y)

    # Keep only those sources inside the bounding box
    inside = np.zeros(len(skymodel), dtype=bool)
    xmin, ymin, xmax, ymax = polygon.bounds
    inside_ind = np.where((x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax))
    inside[inside_ind] = True
    skymodel.select(inside)
    if len(skymodel) == 0:
        return skymodel
    RA = skymodel.getColValues('Ra')
    Dec = skymodel.getColValues('Dec')
    x, y = radec2xy(wcs, RA, Dec)
    x = np.array(x)
    y = np.array(y)

    # Now check the actual boundary against filtered sky model
    xpadding = max(int(0.1 * (max(x) - min(x))), 3)
    ypadding = max(int(0.1 * (max(y) - min(y))), 3)
    xshift = int(min(x)) - xpadding
    yshift = int(min(y)) - ypadding
    xsize = int(np.ceil(max(x) - min(x))) + 2*xpadding
    ysize = int(np.ceil(max(y) - min(y))) + 2*ypadding
    x -= xshift
    y -= yshift
    prepared_polygon = prep(polygon)

    # Unmask everything outside of the polygon + its border (outline)
    inside = np.zeros(len(skymodel), dtype=bool)
    mask = Image.new('L', (xsize, ysize), 0)
    verts = [(xv-xshift, yv-yshift) for xv, yv in zip(polygon.exterior.coords.xy[0],
                                                      polygon.exterior.coords.xy[1])]
    ImageDraw.Draw(mask).polygon(verts, outline=1, fill=1)
    inside_ind = np.where(np.array(mask).transpose()[(x.astype(int), y.astype(int))])
    inside[inside_ind] = True

    # Now check sources in the border precisely
    mask = Image.new('L', (xsize, ysize), 0)
    ImageDraw.Draw(mask).polygon(verts, outline=1, fill=0)
    border_ind = np.where(np.array(mask).transpose()[(x.astype(int), y.astype(int))])
    points = [Point(xs, ys) for xs, ys in zip(x[border_ind], y[border_ind])]
    indexes = []
    for i in range(len(points)):
        indexes.append(border_ind[0][i])
    i_points = zip(indexes, points)
    i_outside_points = [(i, p) for (i, p) in i_points if not prepared_polygon.contains(p)]
    for idx, _ in i_outside_points:
        inside[idx] = False
    skymodel.select(inside)

    return skymodel
