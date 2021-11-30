#!/usr/bin/env python3
"""
Script to flag on residual gain solutions
"""
import argparse
from argparse import RawTextHelpFormatter
import losoto.operations as operations
from losoto.h5parm import h5parm


def main(h5parmfile, solsetname='sol000', phasesoltabname='phase000', ampsoltabname='amplitude000'):
    """
    Flag residual gains

    Parameters
    ----------
    h5parmfile : str
        Filename of h5parm
    phasesoltabname : str, optional
        Name of phase soltab
    ampsoltabname : str, optional
        Name of amplitude soltab
    """
    # Read in solutions
    H = h5parm(h5parmfile, readonly=False)
    solset = H.getSolset(solsetname)
    phsoltab = solset.getSoltab(phasesoltabname)
    ampsoltab = solset.getSoltab(ampsoltabname)

    # Flag
    operations.flagstation.run(phsoltab, 'resid', nSigma=3.0)
    operations.flagstation.run(ampsoltab, 'resid', nSigma=5.0)

    # Reset phases to 0 and amps to 1, so only the new flags are applied
    operations.reset.run(phsoltab)
    operations.reset.run(ampsoltab)
