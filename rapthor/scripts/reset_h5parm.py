#!/usr/bin/env python3
"""
Script to reset solutions for given stations in an h5parm
"""
import argparse
from argparse import RawTextHelpFormatter
from losoto.h5parm import h5parm
from losoto.operations import reset, reweight
import sys
import os


def main(h5in, baseline, solset='sol000', soltab='tec000', dataval=0.0):
    """
    Reset solutions

    Parameters
    ----------
    h5in : str
        Filename of h5parm
    baseline : str
        Baseline selection used in calibration. E.g.:
            ""[CR]*&&;!RS106HBA;!RS205HBA;!RS208HBA"
    solset : str, optional
        Name of solset
    soltab : str, optional
        Name of soltab
    dataval : float, optional
        Value to set solutions to

    """
    h = h5parm(h5in, readonly=False)
    s = h.getSolset(solset)
    t = s.getSoltab(soltab)

    # Make list of excluded stations and set selection to them
    remotelist = [stat for stat in t.ant if '!{}'.format(stat) in baseline]
    t.setSelection(ant=remotelist)

    # Reset the values
    reset.run(t, dataVal=float(dataval))

    # Reset the weights
    reweight.run(t, weightVal=1.0)

    h.close()


if __name__ == '__main__':
    descriptiontext = "Reset solutions.\n"

    parser = argparse.ArgumentParser(description=descriptiontext, formatter_class=RawTextHelpFormatter)
    parser.add_argument('h5', help='name of input h5')
    parser.add_argument('baseline', help='baseline selection')
    args = parser.parse_args()

    main(args.h5, args.baseline)
