"""
Module that holds functions and classes related to faceting
"""

import ast
import re

from astropy.coordinates import SkyCoord
import astropy.units as u
from lsmtool.facet import Facet
from lsmtool.constants import WCS_PIXEL_SCALE
from lsmtool.io import load
from lsmtool.operations_lib import normalize_ra_dec
import numpy as np
from lsmtool.facet import tessellate
from PIL import Image, ImageDraw
from shapely.prepared import prep
from shapely.geometry import Point, Polygon


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
    lines.append(
        "# Region file format: DS9 version 4.0\nglobal color=green "
        'font="helvetica 10 normal" select=1 highlite=1 edit=1 '
        "move=1 delete=1 include=1 fixed=0 source=1\nfk5\n"
    )

    for facet in facets:
        radec_list = []
        RAs = facet.polygon_ras
        Decs = facet.polygon_decs
        for ra, dec in zip(RAs, Decs):
            radec_list.append("{0}, {1}".format(ra, dec))
        lines.append("polygon({0})\n".format(", ".join(radec_list)))
        if enclose_names:
            lines.append("point({0}, {1}) # text={{{2}}}\n".format(facet.ra, facet.dec, facet.name))
        else:
            lines.append("point({0}, {1}) # text={2}\n".format(facet.ra, facet.dec, facet.name))

    with open(outfile, "w") as f:
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
    with open(region_file, "r") as f:
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
        if line.startswith("polygon"):
            # New facet definition begins
            indx += 1
            vertices = ast.literal_eval(line.split("polygon")[1])
            polygon_ras = [ra for ra in vertices[::2]]
            polygon_decs = [dec for dec in vertices[1::2]]
            vertices = [(ra, dec) for ra, dec in zip(polygon_ras, polygon_decs)]

            # Make a temporary facet to get centroid and make new facet with
            # reference point at centroid (this point may be overridden by
            # a following 'point' line)
            facet_tmp = Facet("temp", polygon_ras[0], polygon_decs[0], vertices)
            ra = facet_tmp.ra_centroid
            dec = facet_tmp.dec_centroid

        elif line.startswith("point"):
            # Facet definition continues
            if not len(facets):
                raise ValueError(
                    f'Error parsing region file "{region_file}": "point" '
                    'line found without a preceding "polygon" line'
                )
            facet_tmp = facets.pop()
            vertices = facet_tmp.vertices
            ra, dec = ast.literal_eval(line.split("point")[1])

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
        if "text" in line:
            pattern = r'^[^#]*#\s*text\s*=\s*[{"\']([^}"\']*)[}"\'].*$'
            try:
                facet_name = re.match(pattern, line).group(1)
            except AttributeError:  # raised if `re.match()` returns `None`
                raise ValueError(
                    f'Error parsing region file "{region_file}": '
                    '"text" property could not be parsed for line: '
                    f"{line}"
                )

            # Replace characters that are potentially problematic for Rapthor,
            # DP3, etc. with an underscore
            for invalid_char in [" ", "{", "}", '"', "'"]:
                facet_name = facet_name.replace(invalid_char, "_")
        else:
            facet_name = f"facet_{indx}"

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
    skymod = load(skymodel)
    if not skymod.hasPatches:
        raise ValueError("The sky model must be grouped into patches")

    # Set the position of the calibration patches to those of
    # the input sky model
    source_dict = skymod.getPatchPositions()
    name_cal = []
    ra_cal = []
    dec_cal = []
    for k, v in source_dict.items():
        name_cal.append(k)
        # Make sure RA is between [0, 360) deg and Dec between [-90, 90]
        ra, dec = normalize_ra_dec(v[0].value, v[1].value)
        ra_cal.append(ra)
        dec_cal.append(dec)
    patch_coords = SkyCoord(ra=np.array(ra_cal) * u.degree, dec=np.array(dec_cal) * u.degree)

    # Do the tessellation
    facet_points, facet_polys = tessellate(
        SkyCoord(ra_cal, dec_cal, unit="deg"),
        SkyCoord(ra_mid, dec_mid, unit="deg"),
        [width_ra, width_dec],
        wcs_pixel_scale=WCS_PIXEL_SCALE,
    )
    facet_names = []
    for facet_point in facet_points:
        # For each facet, match the correct patch name (i.e., the name of the
        # patch closest to the facet reference point). This step is needed
        # because some patches in the sky model may not appear in the facet list if
        # they lie outside the bounding box
        facet_coord = SkyCoord(ra=facet_point[0] * u.degree, dec=facet_point[1] * u.degree)
        separations = facet_coord.separation(patch_coords)
        facet_names.append(np.array(name_cal)[np.argmin(separations)])

    # Create the facets
    facets = []
    for name, center_coord, vertices in zip(facet_names, facet_points, facet_polys):
        facets.append(Facet(name, center_coord[0], center_coord[1], vertices))

    return facets


def filter_skymodel(polygon, skymodel, wcs, invert=False):
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
    invert : bool, optional
        If True, invert the selection (so select only sources that lie outside
        the facet)

    Returns
    -------
    filtered_skymodel : LSMTool skymodel object
        Filtered sky model
    """
    # Make list of sources
    RA = skymodel.getColValues("Ra")
    Dec = skymodel.getColValues("Dec")
    x, y = wcs.wcs_world2pix(RA, Dec, misc.WCS_ORIGIN)

    # Keep only those sources inside the bounding box
    inside = np.zeros(len(skymodel), dtype=bool)
    xmin, ymin, xmax, ymax = polygon.bounds
    inside_ind = np.where((x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax))
    inside[inside_ind] = True
    if invert:
        skymodel.remove(inside)
    else:
        skymodel.select(inside)
    if len(skymodel) == 0:
        return skymodel
    RA = skymodel.getColValues("Ra")
    Dec = skymodel.getColValues("Dec")
    x, y = wcs.wcs_world2pix(RA, Dec, misc.WCS_ORIGIN)

    # Now check the actual boundary against filtered sky model. We first do a quick (but
    # coarse) check using ImageDraw with a padding of at least a few pixels to ensure the
    # quick check does not remove sources spuriously. We then do a slow (but precise)
    # check using Shapely
    xpadding = max(int(0.1 * (max(x) - min(x))), 3)
    ypadding = max(int(0.1 * (max(y) - min(y))), 3)
    xshift = int(min(x)) - xpadding
    yshift = int(min(y)) - ypadding
    xsize = int(np.ceil(max(x) - min(x))) + 2 * xpadding
    ysize = int(np.ceil(max(y) - min(y))) + 2 * ypadding
    x -= xshift
    y -= yshift
    prepared_polygon = prep(polygon)

    # Unmask everything outside of the polygon + its border (outline)
    inside = np.zeros(len(skymodel), dtype=bool)
    mask = Image.new("L", (xsize, ysize), 0)
    verts = [
        (xv - xshift, yv - yshift)
        for xv, yv in zip(polygon.exterior.coords.xy[0], polygon.exterior.coords.xy[1])
    ]
    ImageDraw.Draw(mask).polygon(verts, outline=1, fill=1)
    inside_ind = np.where(np.array(mask).transpose()[(x.astype(int), y.astype(int))])
    inside[inside_ind] = True

    # Now check sources in the border precisely
    mask = Image.new("L", (xsize, ysize), 0)
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
    if invert:
        skymodel.remove(inside)
    else:
        skymodel.select(inside)

    return skymodel
