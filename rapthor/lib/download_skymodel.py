#!/usr/bin/env python
import glob
import os
import time

from rapthor.lib import miscellaneous
import casacore.tables as ct
import lsmtool
import numpy as np

import logging
logger = logging.getLogger('rapthor:skymodel')

def get_ms_phasedir(MS):
    """ Read the PHASE_DIR from one Measurement Set and return its RA and DEC in degrees.

    Parameters
    ----------
    MS : str
        Full name (with path) to one MS of the field

    Returns
    -------
    ra, dec : "tuple"
        Coordinates of the field (RA, Dec in deg , J2000)
    """
    ra, dec = ct.table(MS+'::FIELD', readonly=True, ack=False).getcol('PHASE_DIR').squeeze()

    # convert radians to degrees
    ra_deg =  ra / np.pi * 180.
    dec_deg = dec / np.pi * 180.

    # In case RA happens to be negative, fix it to a 0 < ra < 360 value again.
    ra_deg = miscellaneous.normalize_ra(ra_deg)

    # and sending the coordinates in deg
    return (ra_deg, dec_deg)

def download(ms_input, skymodel_path, radius=5.0, overwrite=False, source="TGSS", targetname = "Patch"):
    """
    Download the skymodel for the target field

    Parameters
    ----------
    ms_input : str
        Input Measurement Set to download a skymodel for.
    skymodel_path : str
        Full name (with path) to the skymodel; if YES is true, the skymodel will be downloaded here.
    radius : float
        Radius for the TGSS/GSM cone search in degrees.
    overwrite : bool
        Overwrite the existing skymodel pointed to by skymodel_path.
    targetname : str
        Give the patch a certain name, default: "Patch"
    """
    FileExists = os.path.isfile(skymodel_path)
    if FileExists and not overwrite:
        logger.error('Skymodel "%s" exists and overwrite is set to False!' % skymodel_path)
        raise ValueError('Skymodel "%s" exists and overwrite is set to False!' % skymodel_path)

    if (not FileExists and os.path.exists(skymodel_path)):
        logger.error('Path "%s" exists but is not a file!' % skymodel_path)
        raise ValueError('Path "%s" exists but is not a file!' % skymodel_path)

    if not os.path.exists(os.path.dirname(skymodel_path)):
        os.makedirs(os.path.dirname(skymodel_path))

    if overwrite:
        if FileExists:
            os.remove(skymodel_path)

    logger.info('Downloading skymodel for the target into ' + skymodel_path)

    # Reading a MS to find the coordinate (pyrap)
    RATar, DECTar = get_ms_phasedir(ms_input)

    # Downloading the skymodel, skip after five tries
    errorcode = 1
    tries     = 0
    while errorcode != 0 and tries < 5:
        if source == 'TGSS':
            errorcode = os.system('wget -O ' + skymodel_path + " \'http://tgssadr.strw.leidenuniv.nl/cgi-bin/gsmv4.cgi?coord="+str(RATar)+","+str(DECTar)+"&radius="+str(radius)+"&unit=deg&deconv=y\' ")
        elif source == 'GSM':
            errorcode = os.system('wget -O ' + skymodel_path + " \'https://lcs165.lofar.eu/cgi-bin/gsmv1.cgi?coord="+str(RATar)+","+str(DECTar)+"&radius="+str(radius)+"&unit=deg&deconv=y\' ")
        time.sleep(5)
        tries += 1

    if not os.path.isfile(skymodel_path):
        logger.error('Path: "%s" does not exist after trying to download the skymodel.' % skymodel_path)
        raise IOError('Path: "%s" does not exist after trying to download the skymodel.' % skymodel_path)

    # Treat all sources as one group (direction)
    skymodel = lsmtool.load(skymodel_path)
    skymodel.group('single', root = targetname)
    skymodel.write(clobber=True)
    
    return 0


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=' Download the TGSS or GSM skymodel for the target field')

    parser.add_argument('MSfile', type=str, nargs='+',
                        help='One (or more MSs) for which a TGSS/GSM skymodel will be download.')
    parser.add_argument('skymodel_path', type=str,
                        help='Full name (with path) to the skymodel; the TGSS/GSM skymodel will be downloaded here.')
    parser.add_argument('--radius', type=float, default=5.,
                        help='Radius for the TGSS/GSM cone search in degrees.')
    parser.add_argument('--source', type=str, default='TGSS',
                        help='Choose source for skymodel: TGSS or GSM.')
    parser.add_argument('--overwrite', type=str, default=False,
                        help='Download or not the TGSS skymodel or GSM ("Force" or "True" or "False").')
    parser.add_argument('--targetname', type=str, default='Patch',
                        help='Name of the patch of the skymodel.')

    args = parser.parse_args()

    download(args.MSfile, args.skymodel_path, args.radius, args.overwrite, args.source, args.targetname)
