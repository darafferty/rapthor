"""
Module that holds miscellaneous functions and classes
"""
import logging
import multiprocessing
import os
import subprocess
import time
from math import modf

import astropy.units as u
import lsmtool
import mocpy
import numpy as np
import requests
from astropy.coordinates import SkyCoord
from astropy.io import fits as pyfits
from astropy.time import Time
from losoto.h5parm import h5parm
from scipy.interpolate import interp1d


# Always use a 0-based origin in wcs_pix2world and wcs_world2pix calls.
WCS_ORIGIN = 0
# Default WCS pixel scale within Rapthor, which differs from LSMTool (20"/pixel).
WCS_PIXEL_SCALE = 10.0 / 3600.0 # degrees/pixel (= 10"/pixel)


def download_skymodel(ra, dec, skymodel_path, radius=5.0, overwrite=False, source='TGSS',
                      targetname='Patch'):
    """
    Downloads a skymodel for the given position and radius

    Parameters
    ----------
    ra : float
        Right ascension in degrees of the skymodel centre
    dec : float
        Declination in degrees of the skymodel centre
    skymodel_path : str
        Full name (with path) to the output skymodel
    radius : float, optional
        Radius for the cone search in degrees. For Pan-STARRS, the radius must be
        <= 0.5 degrees
    source : str, optional
        Source where to obtain a skymodel from. Can be one of: TGSS, GSM, LOTSS, or
        PANSTARRS. Note: the PANSTARRS sky model is only suitable for use in
        astrometry checks and should not be used for calibration
    overwrite : bool, optional
        Overwrite the existing skymodel pointed to by skymodel_path
    target_name : str, optional
        Give the patch a certain name
    """
    logger = logging.getLogger('rapthor:skymodel')

    file_exists = os.path.isfile(skymodel_path)
    if file_exists and not overwrite:
        logger.warning('Sky model "{}" exists and overwrite is set to False! Not '
                       'downloading sky model. If this is a restart this may be '
                       'intentional.'.format(skymodel_path))
        return

    if not file_exists and os.path.exists(skymodel_path):
        raise ValueError('Path "%s" exists but is not a file!' % skymodel_path)

    # Empty strings are False. Only attempt directory creation if there is a
    # directory path involved.
    if (
        not file_exists and
        os.path.dirname(skymodel_path) and
        not os.path.exists(os.path.dirname(skymodel_path))
    ):
        os.makedirs(os.path.dirname(skymodel_path))

    if file_exists and overwrite:
        logger.warning('Found existing sky model "{}" and overwrite is True. Deleting '
                       'existing sky model!'.format(skymodel_path))
        os.remove(skymodel_path)

    # Check the radius for Pan-STARRS (it must be <= 0.5 degrees)
    source = source.upper().strip()
    if source == 'PANSTARRS' and radius > 0.5:
        raise ValueError('The radius for Pan-STARRS must be <= 0.5 deg')

    # Check if LoTSS has coverage
    if source == 'LOTSS':
        logger.info('Checking LoTSS coverage for the requested centre and radius.')
        mocpath = os.path.join(os.path.dirname(skymodel_path), 'dr2-moc.moc')
        subprocess.run(['wget', 'https://lofar-surveys.org/public/DR2/catalogues/dr2-moc.moc',
                        '-O', mocpath], capture_output=True, check=True)
        moc = mocpy.MOC.from_fits(mocpath)
        covers_centre = moc.contains(ra * u.deg, dec * u.deg)

        # Checking single coordinates, so get rid of the array
        covers_left = moc.contains(ra * u.deg - radius * u.deg, dec * u.deg)[0]
        covers_right = moc.contains(ra * u.deg + radius * u.deg, dec * u.deg)[0]
        covers_bottom = moc.contains(ra * u.deg, dec * u.deg - radius * u.deg)[0]
        covers_top = moc.contains(ra * u.deg, dec * u.deg + radius * u.deg)[0]
        if covers_centre and not (covers_left and covers_right and covers_bottom and covers_top):
            logger.warning('Incomplete LoTSS coverage for the requested centre and radius! '
                           'Please check the field coverage in plots/field_coverage.png!')
        elif not covers_centre and (covers_left or covers_right or covers_bottom or covers_top):
            logger.warning('Incomplete LoTSS coverage for the requested centre and radius! '
                           'Please check the field coverage in plots/field_coverage.png!')
        elif not covers_centre and not (covers_left and covers_right and covers_bottom and covers_top):
            raise ValueError('No LoTSS coverage for the requested centre and radius!')
        else:
            logger.info('Complete LoTSS coverage for the requested centre and radius.')

    logger.info('Downloading skymodel for the target into ' + skymodel_path)
    max_tries = 5
    for tries in range(1, 1 + max_tries):
        retry = False
        if source == 'LOTSS' or source == 'TGSS' or source == 'GSM':
            try:
                skymodel = lsmtool.skymodel.SkyModel(source, VOPosition=[ra, dec], VORadius=radius)
                skymodel.write(skymodel_path)
                if len(skymodel) > 0:
                    break
            except ConnectionError:
                retry = True
        elif source == 'PANSTARRS':
            baseurl = 'https://catalogs.mast.stsci.edu/api/v0.1/panstarrs'
            release = 'dr1'  # the release with the mean data
            table = 'mean'  # the main catalog, with the mean data
            cat_format = 'csv'  # use csv format for the intermediate file
            url = f'{baseurl}/{release}/{table}.{cat_format}'
            search_params = {'ra': ra,
                             'dec': dec,
                             'radius': radius,
                             'nDetections.min': '5',  # require detection in at least 5 epochs
                             'columns': ['objID', 'ramean', 'decmean']  # get only the info we need
                             }
            try:
                result = requests.get(url, params=search_params, timeout=300)
                if result.ok:
                    # Convert the result to makesourcedb format and write to the output file
                    lines = result.text.split('\n')[1:]  # split and remove header line
                    out_lines = ['FORMAT = Name, Ra, Dec, Type, I, ReferenceFrequency=1e6\n']
                    for line in lines:
                        # Add entries for type and Stokes I flux density
                        if line.strip():
                            out_lines.append(line.strip() + ',POINT,0.0,\n')
                    with open(skymodel_path, 'w') as f:
                        f.writelines(out_lines)
                    break
                else:
                    retry = True
            except requests.exceptions.Timeout:
                retry = True
        else:
            raise ValueError('Unsupported sky model source specified! Please use LOTSS, TGSS, '
                             'GSM, or PANSTARRS.')

        if retry:
            if tries == max_tries:
                logger.error('Attempt #{0:d} to download {1} sky model failed.'.format(tries, source))
                raise IOError('Download of {0} sky model failed after {1} attempts.'.format(source, max_tries))
            else:
                suffix = 's' if max_tries - tries > 1 else ''
                logger.error('Attempt #{0:d} to download {1} sky model failed. Attempting '
                             '{2:d} more time{3}.'.format(tries, source, max_tries - tries, suffix))
                time.sleep(5)

    if not os.path.isfile(skymodel_path):
        raise IOError('Sky model file "{}" does not exist after trying to download the '
                      'sky model.'.format(skymodel_path))

    # Treat all sources as one group (direction)
    skymodel = lsmtool.load(skymodel_path)
    skymodel.group('single', root=targetname)
    skymodel.write(clobber=True)


def normalize_ra_dec(ra, dec):
    """
    Normalize RA to be in the range [0, 360) and Dec to be in the
    range [-90, 90].

    Parameters
    ----------
    ra : float
        The RA in degrees to be normalized.
    dec : float
        The Dec in degrees to be normalized.

    Returns
    -------
    normalized_ra, normalized_dec : float, float
        The normalized RA in degrees in the range [0, 360) and the
        Dec in degrees in the range [-90, 90].
    """
    normalized_dec = (dec + 180) % 360 - 180
    normalized_ra = ra % 360
    if abs(normalized_dec) > 90:
        normalized_dec = 180 - normalized_dec
        normalized_ra = normalized_ra + 180
        normalized_dec = (normalized_dec + 180) % 360 - 180
        normalized_ra = normalized_ra % 360

    return normalized_ra, normalized_dec


def read_vertices(filename, wcs):
    """
    Read facet vertices from a file and convert them to pixel coordinates.

    Parameters
    ----------
    wcs : astropy.wcs.WCS object
        WCS object for converting the vertices to pixel coordinates.

    Returns
    -------
    vertices: list of (x, y) tuples of float
        The converted coordinates.
    """
    # The input file always contains vertices as RA,Dec coordinates.
    vertices_celestial = lsmtool.io.read_vertices_ra_dec(filename)

    # Convert to x, y coordinates.
    vertices_x, vertices_y = wcs.wcs_world2pix(*vertices_celestial, WCS_ORIGIN)

    # Convert to a list of (x, y) tuples.
    return list(zip(vertices_x, vertices_y))


def make_template_image(image_name, reference_ra_deg, reference_dec_deg,
                        ximsize=512, yimsize=512, cellsize_deg=0.000417, freqs=None, times=None,
                        antennas=None, aterm_type='tec', fill_val=0):
    """
    Make a blank FITS image and save it to disk

    Parameters
    ----------
    image_name : str
        Filename of output image
    reference_ra_deg : float
        RA for center of output mask image
    reference_dec_deg : float
        Dec for center of output mask image
    imsize : int, optional
        Size of output image
    cellsize_deg : float, optional
        Size of a pixel in degrees
    freqs : list, optional
        Frequencies to use to construct extra axes (for IDG a-term images)
    times : list, optional
        Times to use to construct extra axes (for IDG a-term images)
    antennas : list, optional
        Antennas to use to construct extra axes (for IDG a-term images)
    aterm_type : str, optional
        One of 'tec' or 'gain'
    fill_val : int, optional
        Value with which to fill the data
    """
    if freqs is not None and times is not None and antennas is not None:
        nants = len(antennas)
        ntimes = len(times)
        nfreqs = len(freqs)
        if aterm_type == 'tec':
            # TEC solutions
            # data is [RA, DEC, ANTENNA, FREQ, TIME].T
            shape_out = [ntimes, nfreqs, nants, yimsize, ximsize]
        else:
            # Gain solutions
            # data is [RA, DEC, MATRIX, ANTENNA, FREQ, TIME].T
            shape_out = [ntimes, nfreqs, nants, 4, yimsize, ximsize]
    else:
        # Normal FITS image
        # data is [STOKES, FREQ, DEC, RA]
        shape_out = [1, 1, yimsize, ximsize]
        nfreqs = 1
        freqs = [150e6]

    hdu = pyfits.PrimaryHDU(np.ones(shape_out, dtype=np.float32)*fill_val)
    hdulist = pyfits.HDUList([hdu])
    header = hdulist[0].header

    # Add RA, Dec info
    i = 1
    header['CRVAL{}'.format(i)] = reference_ra_deg
    header['CDELT{}'.format(i)] = -cellsize_deg
    header['CRPIX{}'.format(i)] = ximsize / 2.0
    header['CUNIT{}'.format(i)] = 'deg'
    header['CTYPE{}'.format(i)] = 'RA---SIN'
    i += 1
    header['CRVAL{}'.format(i)] = reference_dec_deg
    header['CDELT{}'.format(i)] = cellsize_deg
    header['CRPIX{}'.format(i)] = yimsize / 2.0
    header['CUNIT{}'.format(i)] = 'deg'
    header['CTYPE{}'.format(i)] = 'DEC--SIN'
    i += 1

    # Add STOKES info or ANTENNA (+MATRIX) info
    if antennas is None:
        # basic image
        header['CRVAL{}'.format(i)] = 1.0
        header['CDELT{}'.format(i)] = 1.0
        header['CRPIX{}'.format(i)] = 1.0
        header['CUNIT{}'.format(i)] = ''
        header['CTYPE{}'.format(i)] = 'STOKES'
        i += 1
    else:
        if aterm_type == 'gain':
            # gain aterm images: add MATRIX info
            header['CRVAL{}'.format(i)] = 0.0
            header['CDELT{}'.format(i)] = 1.0
            header['CRPIX{}'.format(i)] = 1.0
            header['CUNIT{}'.format(i)] = ''
            header['CTYPE{}'.format(i)] = 'MATRIX'
            i += 1

        # dTEC or gain: add ANTENNA info
        header['CRVAL{}'.format(i)] = 0.0
        header['CDELT{}'.format(i)] = 1.0
        header['CRPIX{}'.format(i)] = 1.0
        header['CUNIT{}'.format(i)] = ''
        header['CTYPE{}'.format(i)] = 'ANTENNA'
        i += 1

    # Add frequency info
    ref_freq = freqs[0]
    if nfreqs > 1:
        deltas = freqs[1:] - freqs[:-1]
        del_freq = np.min(deltas)
    else:
        del_freq = 1e8
    header['RESTFRQ'] = ref_freq
    header['CRVAL{}'.format(i)] = ref_freq
    header['CDELT{}'.format(i)] = del_freq
    header['CRPIX{}'.format(i)] = 1.0
    header['CUNIT{}'.format(i)] = 'Hz'
    header['CTYPE{}'.format(i)] = 'FREQ'
    i += 1

    # Add time info
    if times is not None:
        ref_time = times[0]
        if ntimes > 1:
            # Find CDELT as the smallest delta time, but ignore last delta, as it
            # may be smaller due to the number of time slots not being a divisor of
            # the solution interval
            deltas = times[1:] - times[:-1]
            if ntimes > 2:
                del_time = np.min(deltas[:-1])
            else:
                del_time = deltas[0]
        else:
            del_time = 1.0
        header['CRVAL{}'.format(i)] = ref_time
        header['CDELT{}'.format(i)] = del_time
        header['CRPIX{}'.format(i)] = 1.0
        header['CUNIT{}'.format(i)] = 's'
        header['CTYPE{}'.format(i)] = 'TIME'
        i += 1

    # Add equinox
    header['EQUINOX'] = 2000.0

    # Add telescope
    header['TELESCOP'] = 'LOFAR'

    hdulist[0].header = header
    hdulist.writeto(image_name, overwrite=True)
    hdulist.close()


def string2bool(invar):
    """
    Converts a string to a bool

    Parameters
    ----------
    invar : str
        String to be converted

    Returns
    -------
    result : bool
        Converted bool
    """
    if invar is None:
        return None
    if isinstance(invar, bool):
        return invar
    elif isinstance(invar, str):
        if 'TRUE' in invar.upper() or invar == '1':
            return True
        elif 'FALSE' in invar.upper() or invar == '0':
            return False
        else:
            raise ValueError('input2bool: Cannot convert string "'+invar+'" to boolean!')
    elif isinstance(invar, int) or isinstance(invar, float):
        return bool(invar)
    else:
        raise TypeError('Unsupported data type:'+str(type(invar)))


def string2list(invar):
    """
    Converts a string to a list

    Parameters
    ----------
    invar : str
        String to be converted

    Returns
    -------
    result : list
        Converted list
    """
    if invar is None:
        return None
    str_list = None
    if type(invar) is str:
        invar = invar.strip()
        if invar.startswith('[') and invar.endswith(']'):
            str_list = [f.strip(' \'\"') for f in invar.strip('[]').split(',')]
        elif "," in invar:
            str_list = [f.strip(' \'\"') for f in invar.split(',')]
        else:
            str_list = [invar.strip(' \'\"')]
    elif type(invar) is list:
        str_list = [str(f).strip(' \'\"') for f in invar]
    else:
        raise TypeError('Unsupported data type:'+str(type(invar)))
    return str_list


def _float_approx_equal(x, y, tol=None, rel=None):
    if tol is rel is None:
        raise TypeError('cannot specify both absolute and relative errors are None')
    tests = []
    if tol is not None:
        tests.append(tol)
    if rel is not None:
        tests.append(rel*abs(x))
    assert tests
    return abs(x - y) <= max(tests)


def approx_equal(x, y, *args, **kwargs):
    """
    Return True if x and y are approximately equal, otherwise False

    If x and y are floats, return True if y is within either absolute error
    tol or relative error rel of x. You can disable either the absolute or
    relative check by passing None as tol or rel (but not both).

    Parameters
    ----------
    x : float
        First value to be compared
    y : float
        Second value to be compared
    """
    if not (type(x) is type(y) is float):
        # Skip checking for __approx_equal__ in the common case of two floats.
        methodname = '__approx_equal__'
        # Allow the objects to specify what they consider "approximately equal",
        # giving precedence to x. If either object has the appropriate method, we
        # pass on any optional arguments untouched.
        for a, b in ((x, y), (y, x)):
            try:
                method = getattr(a, methodname)
            except AttributeError:
                continue
            else:
                result = method(b, *args, **kwargs)
                if result is NotImplemented:
                    continue
                return bool(result)
    # If we get here without returning, then neither x nor y knows how to do an
    # approximate equal comparison (or are both floats). Fall back to a numeric
    # comparison.
    return _float_approx_equal(x, y, *args, **kwargs)


def ra2hhmmss(deg, as_string=False):
    """
    Convert RA coordinate (in degrees) to HH MM SS.S

    Parameters
    ----------
    deg : float
        The RA coordinate in degrees
    as_string : bool
        If True, return the RA as a string with 'h', 'm', and 's'
        as the separators. E.g.: '12h23m13.4s' If False, return
        a tuple of (HH, MM, SS.S)

    Returns
    -------
    hhmmss : tuple of (int, int, float) or string
        A tuple of (HH, MM, SS.S) or a string as 'HHhMMmSS.Ss'
    """
    deg = deg % 360
    x, hh = modf(deg/15)
    x, mm = modf(x*60)
    ss = x*60

    if as_string:
        return f'{int(hh)}h{int(mm)}m{ss}s'
    else:
        return (int(hh), int(mm), ss)


def dec2ddmmss(deg, as_string=False):
    """
    Convert Dec coordinate (in degrees) to DD MM SS.S

    Parameters
    ----------
    deg : float
        The Dec coordinate in degrees
    as_string : bool
        If True, return the Dec as a string with 'd', 'm', and 's'
        as the separators. E.g.: '12d23m13.4s'. If False, return
        a tuple of (DD, MM, SS.S, sign)

    Returns
    -------
    ddmmss : tuple of (int, int, float, int) or string
        A tuple of (DD, MM, SS.S, sign) or a string as 'DDdMMmSS.Ss'
    """
    sign = -1 if deg < 0 else 1
    x, dd = modf(abs(deg))
    x, mm = modf(x*60)
    ss = x*60

    if as_string:
        return f'{sign*int(dd)}d{int(mm)}m{ss}s'
    else:
        return (int(dd), int(mm), ss, sign)


def convert_mjd2mvt(mjd_sec):
    """
    Converts MJD to casacore MVTime

    Parameters
    ----------
    mjd_sec : float
        MJD time in seconds

    Returns
    -------
    mvtime : str
        Casacore MVTime string
    """
    t = Time(mjd_sec / 3600 / 24, format='mjd', scale='utc')

    return t.strftime("%d%b%Y/%H:%M:%S.%f")


def convert_mvt2mjd(mvt_str):
    """
    Converts casacore MVTime to MJD

    Parameters
    ----------
    mvt_str : str
        MVTime time

    Returns
    -------
    mjdtime : float
        MJD time in seconds
    """
    mjd = Time.strptime(mvt_str, "%d%b%Y/%H:%M:%S.%f", format="mjd")

    return mjd.to_value('mjd') * 3600 * 24


def angular_separation(position1, position2):
    """
    Compute the angular separation between two RADec coordinates.

    Parameters
    ----------
    position1 : tuple of float
        The first (RA, Dec) coordinates in degrees.
    position2 : tuple of float
        The second (RA, Dec) coordinates in degrees.

    Returns
    -------
    separation : astropy.units.Quantity
        The angular separation between the two positions, in degrees by default.
    """
    ra1, dec1 = position1
    ra2, dec2 = position2

    coord1 = SkyCoord(ra=ra1 * u.degree, dec=dec1 * u.degree)
    coord2 = SkyCoord(ra=ra2 * u.degree, dec=dec2 * u.degree)

    return coord1.separation(coord2)


def get_reference_station(soltab, max_ind=None):
    """
    Return the index of the station with the lowest fraction of flagged
    solutions

    Parameters
    ----------
    soltab : losoto solution table object
        The input solution table
    max_ind : int, optional
        The maximum station index to use when choosing the reference
        station. The reference station will be drawn from the first
        max_ind stations. If None, all stations are considered.

    Returns
    -------
    ref_ind : int
        Index of the reference station
    """
    if max_ind is None or max_ind > len(soltab.ant):
        max_ind = len(soltab.ant)

    weights = soltab.getValues(retAxesVals=False, weight=True)
    weights = np.sum(weights, axis=tuple([i for i, axis_name in
                                          enumerate(soltab.getAxesNames())
                                          if axis_name != 'ant']), dtype=float)
    ref_ind = np.where(weights[0:max_ind] == np.max(weights[0:max_ind]))[0][0]

    return ref_ind


def remove_soltabs(solset, soltabnames):
    """
    Remove H5parm soltabs from a solset

    Note: the H5parm must be opened with readonly = False

    Parameters
    ----------
    solset : losoto solution set object
        The solution set from which to remove soltabs
    soltabnames : list
        Names of soltabs to remove
    """
    soltabnames = string2list(soltabnames)
    for soltabname in soltabnames:
        try:
            soltab = solset.getSoltab(soltabname)
            soltab.delete()
        except Exception:
            print('Error: soltab "{}" could not be removed'.format(soltabname))


def calc_theoretical_noise(obs_list, w_factor=1.5):
    """
    Return the expected theoretical image noise for a dataset. For convenience,
    the total unflagged fraction is also returned.

    Note: the calculations follow those of SKA Memo 113 (see
    https://arxiv.org/abs/1308.4267) and assume no tapering.
    International stations are not included.

    Parameters
    ----------
    obs_list : list of Observation objects
        List of the Observation objects that make up the full dataset
    w_factor : float, optional
        Factor for increase of noise due to the weighting scheme used
        in imaging (typically ranges from 1.3 - 2)

    Returns
    -------
    noise : float
        Estimate of the expected theoretical noise in Jy/beam
    unflagged_fraction : float
        The total unflagged fraction of the input data
    """
    nobs = len(obs_list)
    if nobs == 0:
        # If no Observations, just return zero for the noise and unflagged
        # fraction
        return (0.0, 0.0)

    # Find the total time and the average total bandwidth, average frequency,
    # average unflagged fraction, and average number of core and remote stations
    # (for the averages, assume each observation has equal weight)
    total_time = 0
    total_bandwidth = 0
    ncore = 0
    nremote = 0
    mid_freq = 0
    unflagged_fraction = 0
    for obs in obs_list:
        total_time += obs.numsamples * obs.timepersample  # sec
        total_bandwidth += obs.numchannels * obs.channelwidth  # Hz
        ncore += len([stat for stat in obs.stations if stat.startswith('CS')])
        nremote += len([stat for stat in obs.stations if stat.startswith('RS')])
        mid_freq += (obs.endfreq + obs.startfreq) / 2 / 1e6  # MHz
        unflagged_fraction += find_unflagged_fraction(obs.ms_filename, obs.starttime, obs.endtime)
    total_bandwidth /= nobs
    ncore = int(np.round(ncore / nobs))
    nremote = int(np.round(nremote / nobs))
    mean_freq = mid_freq / nobs
    unflagged_fraction /= nobs

    # Define table of system equivalent flux densities and interpolate
    # to get the values at the mean frequency of the input observations.
    # Note: values were taken from Table 9 of SKA Memo 113
    sefd_freq_MHz = np.array([15, 30, 45, 60, 75, 120, 150, 180, 210, 240])
    sefd_core_kJy = np.array([483, 89, 48, 32, 51, 3.6, 2.8, 3.2, 3.7, 4.1])
    sefd_remote_kJy = np.array([483, 89, 48, 32, 51, 1.8, 1.4, 1.6, 1.8, 2.0])
    f_core = interp1d(sefd_freq_MHz, sefd_core_kJy)
    f_remote = interp1d(sefd_freq_MHz, sefd_remote_kJy)
    sefd_core = f_core(mean_freq) * 1e3  # Jy
    sefd_remote = f_remote(mean_freq) * 1e3  # Jy

    # Calculate the theoretical noise, adjusted for the unflagged fraction
    core_term = ncore * (ncore - 1) / 2 / sefd_core**2
    remote_term = nremote * (nremote - 1) / 2 / sefd_remote**2
    mixed_term = ncore * nremote / (sefd_core * sefd_remote)
    noise = w_factor / np.sqrt(2 * (2 * total_time * total_bandwidth) *
                               (core_term + mixed_term + remote_term))  # Jy
    noise /= np.sqrt(unflagged_fraction)

    return (noise, unflagged_fraction)


def find_unflagged_fraction(ms_file, start_time, end_time):
    """
    Finds the fraction of data that is unflagged for an MS file in the given
    time range

    Parameters
    ----------
    ms_file : str
        Filename of input MS
    start_time : float
        MJD time in seconds for start of time range
    end_time : float
        MJD time in seconds for end of time range

    Returns
    -------
    unflagged_fraction : float
        Fraction of unflagged data
    """
    # Call taql
    time_selection = f"[select from {ms_file} where TIME in [{start_time} =:= {end_time}]]"
    sum_nfalse = f"sum([select nfalse(FLAG) from {time_selection}])"
    sum_nelements = f"sum([select nelements(FLAG) from {time_selection}])"
    cmd = f"taql 'CALC {sum_nfalse} / {sum_nelements}'"
    result = subprocess.run(cmd, shell=True, capture_output=True, check=True)
    unflagged_fraction = float(result.stdout)

    return unflagged_fraction


def get_flagged_solution_fraction(h5file, solsetname='sol000'):
    """
    Get flagged fraction for solutions in given H5parm

    Parameters
    ----------
    h5file : str
        Filename of input h5parm file
    solsetname : str, optional
        The solution set name to use. The flagged fraction is calculated over
        all solution tables in the given solution set

    Returns
    -------
    flagged_fraction : float
        Flagged fraction
    """
    with h5parm(h5file) as h5parm_obj:
        solset = h5parm_obj.getSolset(solsetname)
        num_flagged = 0
        num_all = 0
        for soltab in solset.getSoltabs():
            num_flagged += np.count_nonzero(np.logical_or(~np.isfinite(soltab.val),
                                                          soltab.weight == 0.0))
            num_all += soltab.val.size
    if num_all == 0:
        raise ValueError('Cannot calculate flagged fraction: no solutions found in '
                         'solset {0} of h5parm file {1}'.format(solsetname, h5file))

    return num_flagged / num_all


def rename_skymodel_patches(skymodel, order_dec='high_to_low', order_ra='high_to_low',
                            dec_bin_width=2.0):
    """
    Rename the patches in the input sky model according to the given scheme

    Note: the patches are first binned by Dec and then sorted by RA within each bin.
    The patch names start from "Patch_1" and increase first by RA and then by Dec, ordered
    either with increasing or decreasing RA and Dec as given by the order_dec and order_ra
    args

    Parameters
    ----------
    skymodel : LSMTool skymodel.SkyModel object
        Input sky model
    order_dec : str, optional
        The scheme to use for ordering:
            - 'high_to_low': patches increase with decreasing Dec
            - 'low_to_high': patches increase with increasing Dec
    order_ra : str, optional
        Same as order_dec, but for RA
    dec_bin_width : float, optional
        Bin width in degrees for the Dec values
    """
    if not skymodel.hasPatches:
        raise ValueError('Cannot rename patches since the input skymodel is not grouped '
                         'into patches.')
    patch_positions = skymodel.getPatchPositions()
    patch_names = []
    patch_ras = []
    patch_decs = []
    for name, position in patch_positions.items():
        patch_names.append(name)
        patch_ras.append(position[0].value)
        patch_decs.append(position[1].value)

    # Make bins in Dec, ordered from low to high
    dec_bins = np.linspace(min(patch_decs)-0.1, max(patch_decs)+0.1,
                           num=int(max(patch_decs)-min(patch_decs))+1)
    if order_dec == 'high_to_low':
        dec_bins = dec_bins[::-1]
    bin_members = np.digitize(patch_decs, dec_bins)

    # Run through the bins, sorting by RA and renaming the patches
    # accordingly
    patch_index = 1
    patch_col = skymodel.getColValues('Patch')
    patch_dict = {}
    for bin_index in range(1, len(dec_bins)):
        # Sort by RA (high to low)
        in_bin = np.where(bin_members == bin_index)
        ras = np.array(patch_ras)[in_bin]
        names = np.array(patch_names)[in_bin][np.argsort(ras)]  # ordered from low to high
        if order_ra == 'high_to_low':
            names = names[::-1]

        # Rename the patches in the model's table metadata
        for old_name in names:
            new_name = f'Patch_{patch_index}'
            patch_dict[new_name] = patch_positions[old_name]
            patch_col[skymodel.getRowIndex(old_name)] = new_name
            patch_index += 1
    skymodel.setColValues('Patch', patch_col)
    skymodel.setPatchPositions(patch_dict)


def get_max_spectral_terms(skymodel_file):
    """
    Get the maximum number of spectral terms (including the zeroth term) in a sky model

    Parameters
    ----------
    skymodel_file : str
        Input sky model filename

    Returns
    -------
    nterms : int
        Maximum number of spectral terms
    """
    skymodel = lsmtool.load(skymodel_file)
    if 'SpectralIndex' in skymodel.getColNames():
        return skymodel.getColValues('SpectralIndex').shape[1] + 1  # add one for the zeroth term
    else:
        return 1  # the zeroth term is always present


def nproc():
    """
    Return the number of CPU cores _available_ to the current process, similar
    to what the Linux `nproc` command does. This can be less than the total
    number of CPU cores in the machine.
    NOTE: This function uses `os.sched_getaffinity()`, which is not available
    on every OS. Use `multiprocessing.cpu_count()` as fall-back.
    """
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return multiprocessing.cpu_count()
