"""
Module that holds functions and classes related to faceting
"""
import numpy as np
import scipy.spatial
from shapely.geometry import Point, Polygon
from shapely.prepared import prep
from PIL import Image, ImageDraw
from astropy.coordinates import SkyCoord, Angle
import astropy.units as u
import ast
import logging
from rapthor.lib import miscellaneous as misc
from matplotlib import patches
import lsmtool
from lsmtool import tableio
import tempfile
import re


class Facet(object):
    """
    Base Facet class

    Parameters
    ----------
    name : str
        Name of facet
    ra : float or str
        RA of reference coordinate in degrees (if float) or as a string in a
        format supported by astropy.coordinates.Angle
    dec : float or str
        Dec of reference coordinate in degrees (if float) or as a string in a
        format supported by astropy.coordinates.Angle
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
        self.ra, self.dec = misc.normalize_ra_dec(ra, dec)
        self.vertices = np.array(vertices)

        # Convert input (RA, Dec) vertices to (x, y) polygon
        self.wcs = misc.make_wcs(self.ra, self.dec)
        self.polygon_ras = [radec[0] for radec in self.vertices]
        self.polygon_decs = [radec[1] for radec in self.vertices]
        x_values, y_values = misc.radec2xy(self.wcs, self.polygon_ras, self.polygon_decs)
        polygon_vertices = [(x, y) for x, y in zip(x_values, y_values)]
        self.polygon = Polygon(polygon_vertices)

        # Find the size and center coordinates of the facet
        xmin, ymin, xmax, ymax = self.polygon.bounds
        self.size = min(0.5, max(xmax-xmin, ymax-ymin) *
                        abs(self.wcs.wcs.cdelt[0]))  # degrees
        self.x_center = xmin + (xmax - xmin)/2
        self.y_center = ymin + (ymax - ymin)/2
        self.ra_center, self.dec_center = misc.xy2radec(self.wcs, self.x_center, self.y_center)

        # Find the centroid of the facet
        self.ra_centroid, self.dec_centroid = misc.xy2radec(self.wcs, self.polygon.centroid.x,
                                                            self.polygon.centroid.y)

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
                skymodel.group('every')
        except IOError:
            # Comparison catalog not downloaded successfully
            self.log.warning('The Pan-STARRS catalog could not be successfully '
                             'downloaded')
            skymodel = tableio.makeEmptyTable()

        return skymodel

    def find_astrometry_offsets(self, comparison_skymodel=None, min_number=5):
        """
        Finds the astrometry offsets for sources in the facet

        The offsets are calculated as (LOFAR model value) - (comparison model
        value); e.g., a positive Dec offset indicates that the LOFAR sources
        are on average North of the comparison source positions.

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
            Comparison sky model. If not given, the Pan-STARRS catalog is
            used
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
        if len(comparison_skymodel) >= min_number:
            result = self.skymodel.compare(comparison_skymodel,
                                           radius='5 arcsec',
                                           excludeMultiple=True,
                                           make_plots=False)
            # Save offsets
            if result is not None:
                self.astrometry_diagnostics.update(result)
        else:
            self.log.warning('Too few matches to determine astrometry offsets '
                             '(min_number = {0} but number of matches '
                             '= {1})'.format(min_number, len(comparison_skymodel)))

    def get_matplotlib_patch(self, wcs=None):
        """
        Returns a matplotlib patch for the facet polygon

        Parameters
        ----------
        wcs : WCS object, optional
            WCS object defining (RA, Dec) <-> (x, y) transformation. If not given,
            the facet's transformation is used

        Returns
        -------
        patch : matplotlib patch object
            The patch for the facet polygon
        """
        if wcs is not None:
            x, y = misc.radec2xy(wcs, self.polygon_ras, self.polygon_decs)
        else:
            x = self.polygon.exterior.coords.xy[0]
            y = self.polygon.exterior.coords.xy[1]
        xy = np.vstack([x, y]).transpose()
        patch = patches.Polygon(xy=xy, edgecolor='black', facecolor='white')

        return patch


class SquareFacet(Facet):
    """
    Wrapper class for a square facet

    Parameters
    ----------
    name : str
        Name of facet
    ra : float or str
        RA of reference coordinate in degrees (if float) or as a string in a
        format supported by astropy.coordinates.Angle
    dec : float or str
        Dec of reference coordinate in degrees (if float) or as a string in a
        format supported by astropy.coordinates.Angle
    width : float
        Width in degrees of facet
    """
    def __init__(self, name, ra, dec, width):
        if type(ra) is str:
            ra = Angle(ra).to('deg').value
        if type(dec) is str:
            dec = Angle(dec).to('deg').value
        ra, dec = misc.normalize_ra_dec(ra, dec)
        wcs = misc.make_wcs(ra, dec)

        # Make the vertices
        xmin = wcs.wcs.crpix[0] - width / 2 / abs(wcs.wcs.cdelt[0])
        xmax = wcs.wcs.crpix[0] + width / 2 / abs(wcs.wcs.cdelt[0])
        ymin = wcs.wcs.crpix[1] - width / 2 / abs(wcs.wcs.cdelt[1])
        ymax = wcs.wcs.crpix[1] + width / 2 / abs(wcs.wcs.cdelt[1])
        ra_llc, dec_llc = misc.xy2radec(wcs, xmin, ymin)  # (RA, Dec) of lower-left corner
        ra_tlc, dec_tlc = misc.xy2radec(wcs, xmin, ymax)  # (RA, Dec) of top-left corner
        ra_trc, dec_trc = misc.xy2radec(wcs, xmax, ymax)  # (RA, Dec) of top-right corner
        ra_lrc, dec_lrc = misc.xy2radec(wcs, xmax, ymin)  # (RA, Dec) of lower-right corner
        vertices = [(ra_llc, dec_llc), (ra_tlc, dec_tlc), (ra_trc, dec_trc), (ra_lrc, dec_lrc)]

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
    wcs = misc.make_wcs(ra_mid, dec_mid, wcs_pixel_scale)
    x_cal, y_cal = misc.radec2xy(wcs, ra_cal, dec_cal)
    x_mid, y_mid = misc.radec2xy(wcs, ra_mid, dec_mid)
    width_x = width_ra / wcs_pixel_scale / 2.0
    width_y = width_dec / wcs_pixel_scale / 2.0
    bounding_box = np.array([x_mid - width_x, x_mid + width_x,
                             y_mid - width_y, y_mid + width_y])

    # Tessellate and convert resulting facet polygons from (x, y) to (RA, Dec)
    vor = voronoi(np.stack((x_cal, y_cal)).T, bounding_box)
    facet_polys = []
    for region in vor.filtered_regions:
        vertices = vor.vertices[region + [region[0]], :]
        ra, dec = misc.xy2radec(wcs, vertices[:, 0], vertices[:, 1])
        vertices = np.stack((ra, dec)).T
        facet_polys.append(vertices)
    facet_points = []
    for point in vor.filtered_points:
        ra, dec = misc.xy2radec(wcs, point[0], point[1])
        facet_points.append((ra, dec))

    return facet_points, facet_polys


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
    vor = scipy.spatial.Voronoi(points)
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
                if not (bounding_box[0] - eps <= x and x <= bounding_box[1] + eps and
                        bounding_box[2] - eps <= y and y <= bounding_box[3] + eps):
                    flag = False
                    break
        if region and flag:
            regions.append(region)
    vor.filtered_points = points_center
    vor.filtered_regions = regions

    return vor


def make_ds9_region_file(facets, outfile, enclose_names=True):
    """
    Make a ds9 region file for given polygons and centers

    Parameters
    ----------
    facets : list of Facet objects
        List of Facet objects to include
    outfile : str
        Name of output region file
    enclose_names : bool, optional
        If True, enclose patch names in curly brackets for full
        compatibility with ds9. Curly brackets may cause issues with
        other tools that use the region file, such as DP3, in which
        case they can be excluded by setting this option to False
    """
    lines = []
    lines.append('# Region file format: DS9 version 4.0\nglobal color=green '
                 'font="helvetica 10 normal" select=1 highlite=1 edit=1 '
                 'move=1 delete=1 include=1 fixed=0 source=1\nfk5\n')

    for facet in facets:
        radec_list = []
        RAs = facet.polygon_ras
        Decs = facet.polygon_decs
        for ra, dec in zip(RAs, Decs):
            radec_list.append('{0}, {1}'.format(ra, dec))
        lines.append('polygon({0})\n'.format(', '.join(radec_list)))
        if enclose_names:
            lines.append('point({0}, {1}) # text={{{2}}}\n'.format(facet.ra, facet.dec, facet.name))
        else:
            lines.append('point({0}, {1}) # text={2}\n'.format(facet.ra, facet.dec, facet.name))

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

    indx = 0
    for line in lines:
        # Each facet in the region file is defined by a polygon line that starts
        # with 'polygon' and gives the (RA, Dec) vertices
        #
        # Each facet polygon line may be followed by a line giving the reference
        # point that starts with 'point' and gives the reference (RA, Dec)
        #
        # The facet name may be set in the text property of either line
        # (see https://wsclean.readthedocs.io/en/latest/ds9_facet_file.html)
        if line.startswith('polygon'):
            # New facet definition begins
            indx += 1
            vertices = ast.literal_eval(line.split('polygon')[1])
            polygon_ras = [ra for ra in vertices[::2]]
            polygon_decs = [dec for dec in vertices[1::2]]
            vertices = [(ra, dec) for ra, dec in zip(polygon_ras, polygon_decs)]

            # Make a temporary facet to get centroid and make new facet with
            # reference point at centroid (this point may be overridden by
            # a following 'point' line)
            facet_tmp = Facet('temp', polygon_ras[0], polygon_decs[0], vertices)
            ra = facet_tmp.ra_centroid
            dec = facet_tmp.dec_centroid

        elif line.startswith('point'):
            # Facet definition continues
            if not len(facets):
                raise ValueError(f'Error parsing region file "{region_file}": "point" '
                                 'line found without a preceding "polygon" line')
            facet_tmp = facets.pop()
            vertices = facet_tmp.vertices
            ra, dec = ast.literal_eval(line.split('point')[1])

        else:
            continue

        # Read the facet name, if any. The name is defined using the 'text'
        # property. E.g.:
        #     'polygon(309.6, 60.9, 310.4, 58.9, 309.1, 59.2) # text = {Patch_1} width = 2'
        #     'point(0.1, 1.2) # text = {Patch_1} width = 2'
        #
        # Note: ds9 format allows strings to be quoted with " or ' or {}
        # (see https://ds9.si.edu/doc/ref/region.html#RegionProperties),
        # so we match everything between "", '', or {}, if the line contains
        # anything like `... # text = ...`
        #
        # Note: if a name is defined for both the facet polygon and the facet
        # reference point, the one for the point takes precedence
        if 'text' in line:
            pattern = r'^[^#]*#\s*text\s*=\s*[{"\']([^}"\']*)[}"\'].*$'
            try:
                facet_name = re.match(pattern, line).group(1)
            except AttributeError:  # raised if `re.match()` returns `None`
                raise ValueError(f'Error parsing region file "{region_file}": '
                                 '"text" property could not be parsed for line: '
                                 f'{line}')

            # Replace characters that are potentially problematic for Rapthor,
            # DP3, etc. with an underscore
            for invalid_char in [' ', '{', '}', '"', "'"]:
                facet_name = facet_name.replace(invalid_char, '_')
        else:
            facet_name = f'facet_{indx}'

        # Lastly, add the facet to the list
        facets.append(Facet(facet_name, ra, dec, vertices))

    return facets


def read_skymodel(skymodel, ra_mid, dec_mid, width_ra, width_dec):
    """
    Reads a sky model file and returns facets

    Parameters
    ----------
    skymodel : str
        Filename of the sky model (must have patches), in makesourcedb format
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
    facets : list
        List of Facet objects
    """
    skymod = lsmtool.load(skymodel)
    if not skymod.hasPatches:
        raise ValueError('The sky model must be grouped into patches')

    # Set the position of the calibration patches to those of
    # the input sky model
    source_dict = skymod.getPatchPositions()
    name_cal = []
    ra_cal = []
    dec_cal = []
    for k, v in source_dict.items():
        name_cal.append(k)
        # Make sure RA is between [0, 360) deg and Dec between [-90, 90]
        ra, dec = misc.normalize_ra_dec(v[0].value, v[1].value)
        ra_cal.append(ra)
        dec_cal.append(dec)
    patch_coords = SkyCoord(ra=np.array(ra_cal)*u.degree, dec=np.array(dec_cal)*u.degree)

    # Do the tessellation
    facet_points, facet_polys = make_facet_polygons(ra_cal, dec_cal, ra_mid, dec_mid, width_ra, width_dec)
    facet_names = []
    for facet_point in facet_points:
        # For each facet, match the correct patch name (i.e., the name of the
        # patch closest to the facet reference point). This step is needed
        # because some patches in the sky model may not appear in the facet list if
        # they lie outside the bounding box
        facet_coord = SkyCoord(ra=facet_point[0]*u.degree, dec=facet_point[1]*u.degree)
        separations = facet_coord.separation(patch_coords)
        facet_names.append(np.array(name_cal)[np.argmin(separations)])

    # Create the facets
    facets = []
    for name, center_coord, vertices in zip(facet_names, facet_points, facet_polys):
        facets.append(Facet(name, center_coord[0], center_coord[1], vertices))

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
    x, y = misc.radec2xy(wcs, RA, Dec)
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
    x, y = misc.radec2xy(wcs, RA, Dec)
    x = np.array(x)
    y = np.array(y)

    # Now check the actual boundary against filtered sky model. We first do a quick (but
    # coarse) check using ImageDraw with a padding of at least a few pixels to ensure the
    # quick check does not remove sources spuriously. We then do a slow (but precise)
    # check using Shapely
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
