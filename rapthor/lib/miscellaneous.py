"""
Module that holds miscellaneous functions and classes
"""
import logging
import os
import shutil
import subprocess
import errno
import numpy as np
import pickle
import time
from shapely.geometry import Point, Polygon
from shapely.prepared import prep
from astropy.io import fits as pyfits
from PIL import Image, ImageDraw
import multiprocessing
from math import modf, floor, ceil
from losoto.h5parm import h5parm
import lsmtool
import sys
from scipy.interpolate import interp1d
from rapthor.lib.observation import Observation


def download_skymodel(ra, dec, skymodel_path, radius=5.0, overwrite=False, source='TGSS',
                      targetname='Patch'):
    """
    Download the skymodel for the target field

    Parameters
    ----------
    ra : float
        Right ascension of the skymodel centre.
    dec : float
        Declination of the skymodel centre.
    skymodel_path : str
        Full name (with path) to the skymodel.
    radius : float, optional
        Radius for the TGSS/GSM cone search in degrees.
    source : str, optional
        Source where to obtain a skymodel from. Can be TGSS or GSM. Default is TGSS.
    overwrite : bool, optional
        Overwrite the existing skymodel pointed to by skymodel_path.
    target_name : str, optional
        Give the patch a certain name. Default is "Patch".
    """
    SKY_SERVERS = {'TGSS': 'http://tgssadr.strw.leidenuniv.nl/cgi-bin/gsmv4.cgi?'
                           'coord={ra:f},{dec:f}&radius={radius:f}&unit=deg&deconv=y',
                   'GSM': 'https://lcs165.lofar.eu/cgi-bin/gsmv1.cgi?'
                          'coord={ra:f},{dec:f}&radius={radius:f}&unit=deg&deconv=y',
                    'LOTSS': ''} # Server is empty since we handle this through LSMTool.
    if source.upper() not in SKY_SERVERS.keys():
        raise ValueError('Unsupported sky model source specified! Please use LOTSS, TGSS or GSM.')

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
    if (not file_exists
            and os.path.dirname(skymodel_path)
            and not os.path.exists(os.path.dirname(skymodel_path))):
        os.makedirs(os.path.dirname(skymodel_path))

    if file_exists and overwrite:
        logger.warning('Found existing sky model "{}" and overwrite is True. Deleting '
                       'existing sky model!'.format(skymodel_path))
        os.remove(skymodel_path)

    logger.info('Downloading skymodel for the target into ' + skymodel_path)
    if source.upper().strip() == 'LOTSS':
        lotssmodel = lsmtool.skymodel.SkyModel('lotss', VOPosition=[ra, dec], VORadius=radius)
        lotssmodel.write(skymodel_path)
    elif (source.upper().strip() == 'TGSS') or (source.upper().strip() == 'GSM'):
        max_tries = 5
        for tries in range(1, 1 + max_tries):
            result = subprocess.run(['wget', '-O', skymodel_path,
                                    SKY_SERVERS[source].format(ra=ra, dec=dec, radius=radius)])
            if result.returncode != 0:
                if tries == max_tries:
                    raise IOError('Download of sky model failed after {} '
                                'attempts.'.format(max_tries))
                else:
                    logger.error('Attempt #{0:d} to download sky model failed. Attempting '
                                '{1:d} more times.'.format(tries, max_tries - tries))
                    time.sleep(5)
            else:
                break

    if not os.path.isfile(skymodel_path):
        raise IOError('Sky model file "{}" does not exist after trying to download the '
                      'sky model.'.format(skymodel_path))

    # Treat all sources as one group (direction)
    skymodel = lsmtool.load(skymodel_path)
    skymodel.group('single', root=targetname)
    skymodel.write(clobber=True)


def normalize_ra(num):
    """
    Normalize RA to be in the range [0, 360).

    Based on https://github.com/phn/angles/blob/master/angles.py

    Parameters
    ----------
    num : float
        The RA in degrees to be normalized.

    Returns
    -------
    res : float
        RA in degrees in the range [0, 360).
    """
    lower = 0.0
    upper = 360.0
    res = num
    if num > upper or num == lower:
        num = lower + abs(num + upper) % (abs(lower) + abs(upper))
    if num < lower or num == upper:
        num = upper - abs(num - lower) % (abs(lower) + abs(upper))
    res = lower if num == upper else num

    return res


def normalize_dec(num):
    """
    Normalize Dec to be in the range [-90, 90].

    Based on https://github.com/phn/angles/blob/master/angles.py

    Parameters
    ----------
    num : float
        The Dec in degrees to be normalized.

    Returns
    -------
    res : float
        Dec in degrees in the range [-90, 90].
    """
    lower = -90.0
    upper = 90.0
    res = num
    total_length = abs(lower) + abs(upper)
    if num < -total_length:
        num += ceil(num / (-2 * total_length)) * 2 * total_length
    if num > total_length:
        num -= floor(num / (2 * total_length)) * 2 * total_length
    if num > upper:
        num = total_length - num
    if num < lower:
        num = -total_length - num
    res = num

    return res


def read_vertices(filename):
    """
    Returns facet vertices stored in input file
    """
    with open(filename, 'rb') as f:
        vertices = pickle.load(f)
    return vertices


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


def rasterize(verts, data, blank_value=0):
    """
    Rasterize a polygon into a data array

    Parameters
    ----------
    verts : list of (x, y) tuples
        List of input vertices of polygon to rasterize
    data : 2-D array
        Array into which rasterize polygon
    blank_value : int or float, optional
        Value to use for blanking regions outside the poly

    Returns
    -------
    data : 2-D array
        Array with rasterized polygon
    """
    poly = Polygon(verts)
    prepared_polygon = prep(poly)

    # Mask everything outside of the polygon plus its border (outline) with zeros
    # (inside polygon plus border are ones)
    mask = Image.new('L', (data.shape[1], data.shape[0]), 0)
    ImageDraw.Draw(mask).polygon(verts, outline=1, fill=1)
    data *= mask

    # Now check the border precisely
    mask = Image.new('L', (data.shape[1], data.shape[0]), 0)
    ImageDraw.Draw(mask).polygon(verts, outline=1, fill=0)
    masked_ind = np.where(np.array(mask).transpose())
    points = [Point(xm, ym) for xm, ym in zip(masked_ind[0], masked_ind[1])]
    outside_points = [v for v in points if prepared_polygon.disjoint(v)]
    for outside_point in outside_points:
        data[int(outside_point.y), int(outside_point.x)] = 0

    if blank_value != 0:
        data[data == 0] = blank_value

    return data


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


def create_directory(dirname):
    """
    Recursively create a directory, without failing if it already exists

    Parameters
    ----------
    dirname : str
        Path of directory
    """
    try:
        if dirname:
            os.makedirs(dirname)
    except OSError as failure:
        if failure.errno != errno.EEXIST:
            raise failure


def delete_directory(dirname):
    """
    Recursively delete a directory tree, without failing if it does not exist

    Parameters
    ----------
    dirname : str
        Path of directory
    """
    try:
        shutil.rmtree(dirname)
    except OSError as e:
        if not e.errno == errno.ENOENT:
            raise e


def ra2hhmmss(deg):
    """
    Convert RA coordinate (in degrees) to HH MM SS

    Parameters
    ----------
    deg : float
        The RA coordinate in degrees

    Returns
    -------
    hh : int
        The hour (HH) part
    mm : int
        The minute (MM) part
    ss : float
        The second (SS) part
    """
    deg = deg % 360
    x, hh = modf(deg/15)
    x, mm = modf(x*60)
    ss = x*60

    return (int(hh), int(mm), ss)


def dec2ddmmss(deg):
    """
    Convert Dec coordinate (in degrees) to DD MM SS

    Parameters
    ----------
    deg : float
        The Dec coordinate in degrees

    Returns
    -------
    dd : int
        The degree (DD) part
    mm : int
        The arcminute (MM) part
    ss : float
        The arcsecond (SS) part
    sign : int
        The sign (+/-)
    """
    sign = (-1 if deg < 0 else 1)
    x, dd = modf(abs(deg))
    x, ma = modf(x*60)
    sa = x*60

    return (int(dd), int(ma), sa, sign)


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


def calc_theoretical_noise(mslist, w_factor=1.5):
    """
    Return the expected theoretical image noise for a dataset

    Note: the calculations follow those of SKA Memo 113 (see
    http://www.skatelescope.org/uploaded/59513_113_Memo_Nijboer.pdf) and
    assume no tapering. International stations are not included.

    Parameters
    ----------
    mslist : list of str
        List of the filenames of the input MS files
    w_factor : float, optional
        Factor for increase of noise due to the weighting scheme used
        in imaging (typically ranges from 1.3 - 2)

    Returns
    -------
    noise, unflagged_fraction : tuple of floats
        Estimate of the expected theoretical noise in Jy/beam and the
        unflagged fraction of the input data
    """
    nobs = len(mslist)
    if nobs == 0:
        # If no MS files, just return zero for the noise as we cannot
        # estimate it
        return 0.0

    # Find the total time and the average total bandwidth, average frequency,
    # average unflagged fraction, and average number of core and remote stations
    # (for the averages, assume each observation has equal weight)
    total_time = 0
    total_bandwidth = 0
    ncore = 0
    nremote = 0
    mid_freq = 0
    unflagged_fraction = 0
    for ms in mslist:
        obs = Observation(ms)
        total_time += obs.numsamples * obs.timepersample  # sec
        total_bandwidth += obs.numchannels * obs.channelwidth  # Hz
        ncore += len([stat for stat in obs.stations if stat.startswith('CS')])
        nremote += len([stat for stat in obs.stations if stat.startswith('RS')])
        mid_freq += (obs.endfreq + obs.startfreq) / 2 / 1e6  # MHz
        unflagged_fraction += find_unflagged_fraction(ms)
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


def find_unflagged_fraction(ms_file):
    """
    Finds the fraction of data that is unflagged

    Parameters
    ----------
    ms_file : str
        Filename of input MS

    Returns
    -------
    unflagged_fraction : float
        Fraction of unflagged data
    """
    # Call taql. Note that we do not use pt.taql(), as pt.taql() can cause
    # hanging/locking issues on some systems
    result = subprocess.run("taql 'CALC sum([select nfalse(FLAG) from {0}]) / "
                            "sum([select nelements(FLAG) from {0}])'".format(ms_file),
                            shell=True, capture_output=True, check=True)
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
    h5parm_obj = h5parm(h5file)
    solset = h5parm_obj.getSolset(solsetname)
    num_flagged = 0
    num_all = 0
    for soltab in solset.getSoltabs():
        num_flagged += np.count_nonzero(np.logical_or(~np.isfinite(soltab.val),
                                                      soltab.weight == 0.0))
        num_all += soltab.val.size
    h5parm_obj.close()
    if num_all == 0:
        raise ValueError('Cannot calculate flagged fraction: no solutions found in '
                         'solset {0} of h5parm file {1}'.format(solsetname, h5file))

    return num_flagged / num_all


class multiprocManager(object):

    class multiThread(multiprocessing.Process):
        """
        This class is a working thread which load parameters from a queue and
        return in the output queue
        """

        def __init__(self, inQueue, outQueue, funct):
            multiprocessing.Process.__init__(self)
            self.inQueue = inQueue
            self.outQueue = outQueue
            self.funct = funct

        def run(self):

            while True:
                parms = self.inQueue.get()

                # poison pill
                if parms is None:
                    self.inQueue.task_done()
                    break

                self.funct(*parms, outQueue=self.outQueue)
                self.inQueue.task_done()

    def __init__(self, procs=0, funct=None):
        """
        Manager for multiprocessing
        procs: number of processors, if 0 use all available
        funct: function to parallelize / note that the last parameter of this function must be the outQueue
        and it will be linked to the output queue
        """
        if procs == 0:
            procs = multiprocessing.cpu_count()
        self.procs = procs
        self._threads = []
        self.inQueue = multiprocessing.JoinableQueue()
        self.outQueue = multiprocessing.Queue()
        self.runs = 0

        for proc in range(self.procs):
            t = self.multiThread(self.inQueue, self.outQueue, funct)
            self._threads.append(t)
            t.start()

    def put(self, args):
        """
        Parameters to give to the next jobs sent into queue
        """
        self.inQueue.put(args)
        self.runs += 1

    def get(self):
        """
        Return all the results as an iterator
        """
        # NOTE: do not use queue.empty() check which is unreliable
        # https://docs.python.org/2/library/multiprocessing.html
        for run in range(self.runs):
            yield self.outQueue.get()

    def wait(self):
        """
        Send poison pills to jobs and wait for them to finish
        The join() should kill all the processes
        """
        for t in self._threads:
            self.inQueue.put(None)

        # wait for all jobs to finish
        self.inQueue.join()
