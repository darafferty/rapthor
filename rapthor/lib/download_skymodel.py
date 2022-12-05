#!/usr/bin/env python
import glob
import os
import sys
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
    ra = miscellaneous.normalize_ra(ra)

    # and sending the coordinates in deg
    return (ra_deg, dec_deg)

def download(ms_input, SkymodelPath, Radius="5.", DoDownload="True", Source="TGSS", targetname = "Patch"):
    """
    Download the skymodel for the target field

    Parameters
    ----------
    ms_input : str
        Input Measurement Set to download a skymodel for.
    SkymodelPath : str
        Full name (with path) to the skymodel; if YES is true, the skymodel will be downloaded here
    Radius : string with float (default = "5.")
        Radius for the TGSS/GSM cone search in degrees
    DoDownload : str ("Force" or "True" or "False")
        Download or not the TGSS skymodel or GSM.
        "Force": download skymodel from TGSS or GSM, delete existing skymodel if needed.
        "True" or "Yes": use existing skymodel file if it exists, download skymodel from
                         TGSS or GSM if it does not.
        "False" or "No": Do not download skymodel, raise an exception if skymodel
                         file does not exist.
    targetname : str
        Give the patch a certain name, default: "Patch"
    """

    FileExists = os.path.isfile(SkymodelPath)
    if (not FileExists and os.path.exists(SkymodelPath)):
        raise ValueError("Path: \"%s\" exists but is not a file!"%(SkymodelPath))
    download_flag = False
    if not os.path.exists(os.path.dirname(SkymodelPath)):
        os.makedirs(os.path.dirname(SkymodelPath))
    if DoDownload.upper() == "FORCE":
        if FileExists:
            os.remove(SkymodelPath)
        download_flag = True
    elif DoDownload.upper() == "TRUE" or DoDownload.upper() == "YES":
        if FileExists:
            logger.info("USING the exising skymodel in "+ SkymodelPath)
            return(0)
        else:
            download_flag = True
    elif DoDownload.upper() == "FALSE" or DoDownload.upper() == "NO":
         if FileExists:
            logger.info("USING the exising skymodel in "+ SkymodelPath)
            return(0)
         else:
            raise ValueError("download_tgss_skymodel_target: Path: \"%s\" does not exist and skymodel download is disabled!"%(SkymodelPath))

    # If we got here, then we are supposed to download the skymodel.
    assert download_flag is True # Jaja, belts and suspenders...
    logger.info("DOWNLOADING skymodel for the target into "+ SkymodelPath)

    # Reading a MS to find the coordinate (pyrap)
    RATar, DECTar = get_ms_phasedir(ms_input)

    # Downloading the skymodel, skip after five tries
    errorcode = 1
    tries     = 0
    while errorcode != 0 and tries < 5:
        if Source == 'TGSS':
            errorcode = os.system("wget -O "+SkymodelPath+ " \'http://tgssadr.strw.leidenuniv.nl/cgi-bin/gsmv4.cgi?coord="+str(RATar)+","+str(DECTar)+"&radius="+str(Radius)+"&unit=deg&deconv=y\' ")
        elif Source == 'GSM':
            errorcode = os.system("wget -O "+SkymodelPath+ " \'https://lcs165.lofar.eu/cgi-bin/gsmv1.cgi?coord="+str(RATar)+","+str(DECTar)+"&radius="+str(Radius)+"&unit=deg&deconv=y\' ")
        time.sleep(5)
        tries += 1

    if not os.path.isfile(SkymodelPath):
        raise IOError("download_tgss_skymodel_target: Path: \"%s\" does not exist after trying to download the skymodel."%(SkymodelPath))

    # Treat all sources as one group (direction)
    skymodel = lsmtool.load(SkymodelPath)
    skymodel.group('single', root = targetname)
    skymodel.write(clobber=True)
    
    return(0)


########################################################################
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=' Download the TGSS or GSM skymodel for the target field')

    parser.add_argument('MSfile', type=str, nargs='+',
                        help='One (or more MSs) for which a TGSS/GSM skymodel will be download.')
    parser.add_argument('SkyTar', type=str,
                        help='Full name (with path) to the skymodel; the TGSS/GSM skymodel will be downloaded here')
    parser.add_argument('--Radius', type=float, default=5.,
                        help='Radius for the TGSS/GSM cone search in degrees')
    parser.add_argument('--Source', type=str, default='TGSS',
                        help='Choose source for skymodel: TGSS or GSM')
    parser.add_argument('--DoDownload', type=str, default="True",
                        help='Download or not the TGSS skymodel or GSM ("Force" or "True" or "False").')
    parser.add_argument('--targetname', type=str, default='Patch',
                        help='Name of the patch of the skymodel')

    args = parser.parse_args()
    radius=5
    if args.Radius:
        radius=args.Radius

    download(args.MSfile, args.SkyTar, str(radius), args.DoDownload, args.Source, args.targetname)
