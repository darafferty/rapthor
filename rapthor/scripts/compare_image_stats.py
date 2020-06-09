#! /usr/bin/env python
"""
Script to compare the dynamic range of two images
"""
import argparse
from argparse import RawTextHelpFormatter
import casacore.images as pim
import numpy
import sys
import os


def robust_sigma(in_y, zero=0):
    """
    Calculate a resistant estimate of the dispersion of
    a distribution. For an uncontaminated distribution,
    this is identical to the standard deviation.

    Use the median absolute deviation as the initial
    estimate, then weight points using Tukey Biweight.
    See, for example, Understanding Robust and
    Exploratory Data Analysis, by Hoaglin, Mosteller
    and Tukey, John Wiley and Sons, 1983.

    .. note:: ROBUST_SIGMA routine from IDL ASTROLIB.

    Examples
    --------
    >>> result = robust_sigma(in_y, zero=1)

    Parameters
    ----------
    in_y : array_like
        Vector of quantity for which the dispersion is
        to be calculated

    zero : int
        If set, the dispersion is calculated w.r.t. 0.0
        rather than the central value of the vector. If
        Y is a vector of residuals, this should be set.

    Returns
    -------
    out_val : float
        Dispersion value. If failed, returns -1.

    """
    # Flatten array
    y = in_y.ravel()

    eps = 1.0E-20
    c1 = 0.6745
    c2 = 0.80
    c3 = 6.0
    c4 = 5.0
    c_err = -1.0
    min_points = 3

    if zero:
        y0 = 0.0
    else:
        y0 = numpy.median(y)

    dy    = y - y0
    del_y = abs( dy )

    # First, the median absolute deviation MAD about the median:

    mad = numpy.median( del_y ) / c1

    # If the MAD=0, try the MEAN absolute deviation:
    if mad < eps:
        mad = del_y.mean() / c2
    if mad < eps:
        return 0.0

    # Now the biweighted value:
    u  = dy / (c3 * mad)
    uu = u * u
    q  = numpy.where(uu <= 1.0)
    count = len(q[0])
    if count < min_points:
        print('ROBUST_SIGMA: This distribution is TOO WEIRD! '
                           'Returning {}'.format(c_err))
        return c_err

    numerator = numpy.sum( (y[q] - y0)**2.0 * (1.0 - uu[q])**4.0 )
    n    = y.size
    den1 = numpy.sum( (1.0 - uu[q]) * (1.0 - c4 * uu[q]) )
    siggma = n * numerator / ( den1 * (den1 - 1.0) )

    if siggma > 0:
        out_val = numpy.sqrt( siggma )
    else:
        out_val = 0.0

    return out_val


def meanclip(indata, clipsig=4.0, maxiter=10, converge_num=0.001, verbose=True):
   """
   Computes an iteratively sigma-clipped mean on a
   data set. Clipping is done about median, but mean
   is returned.

   .. note:: MYMEANCLIP routine from ACS library.

   :History:
       * 21/10/1998 Written by RSH, RITSS
       * 20/01/1999 Added SUBS, fixed misplaced paren on float call, improved doc. RSH
       * 24/11/2009 Converted to Python. PLL.

   Examples
   --------
   >>> mean, sigma = meanclip(indata)

   Parameters
   ----------
   indata: array_like
       Input data.

   clipsig: float
       Number of sigma at which to clip.

   maxiter: int
       Ceiling on number of clipping iterations.

   converge_num: float
       If the proportion of rejected pixels is less than
       this fraction, the iterations stop.

   verbose: {0, 1}
       Print messages to screen?

   Returns
   -------
   mean: float
       N-sigma clipped mean.

   sigma: float
       Standard deviation of remaining pixels.

   """
   # Flatten array
   skpix = indata.reshape( indata.size, )

   ct = indata.size
   iter = 0; c1 = 1.0 ; c2 = 0.0

   while (c1 >= c2) and (iter < maxiter):
       lastct = ct
       medval = numpy.median(skpix)
       sig = numpy.std(skpix)
       wsm = numpy.where( abs(skpix-medval) < clipsig*sig )
       ct = len(wsm[0])
       if ct > 0:
           skpix = skpix[wsm]

       c1 = abs(ct - lastct)
       c2 = converge_num * lastct
       iter += 1
   # End of while loop

   mean  = numpy.mean( skpix )
   sigma = robust_sigma( skpix )

   return mean, sigma


def find_imagenoise(imagename):
    """
    Finds noise, dynamic range, and min/max for an image

    Parameters
    ----------
    imagename : str
        Filename of image

    Returns
    -------
    rms : float
        Noise (Jy/beam) of image
    dynamic_range : float
        Dynamic range (max/min) of image
    minmax : float
        Ratio of min/max

    """
    im    = pim.image(imagename)
    image = numpy.copy(im.getdata())
    mean, rms =  meanclip(image)
    minmax = abs(numpy.min(image) / numpy.max(image))

    return rms, numpy.abs(numpy.max(image)/rms), minmax


def main(im1, im2, count=-1, rapthor=1.0125):
    """
    Compare the dynamic range and min/max of two images and check whether:

        dynamic_range1 / rapthor > dynamic_range2

    or

        abs(min1/max1) * rapthor < abs(min2/max2)

    Typically, im1 is the latest image and im2 the previous one.

    Parameters
    ----------
    im1 : str
        Name of image #1
    im2 : str
        Name of image #2
    count : int, optional
        Loop index. If nonzero, the dynamic range check is skipped for count = 0
        only and break is set to False
    rapthor : float
        Required improvement rapthor for success (i.e., break = True)

    Returns
    -------
    result : dict
        Dict with break value (False if dynamic_range1 / rapthor > dynamic_range2;
        True otherwise)

    """
    rapthor = float(rapthor)
    count = int(count)

    rms1, dynamic_range1, minmax1 =  find_imagenoise(im1)
    rms2, dynamic_range2, minmax2 =  find_imagenoise(im2)

    print('Image 1: rms = {0} Jy/beam; dynamic range = {1}, abs(min/max) = {2}'.format(
        rms1, dynamic_range1, minmax1))
    print('Image 2: rms = {0} Jy/beam; dynamic range = {1}, abs(min/max) = {2}'.format(
        rms2, dynamic_range2, minmax2))

    if count == 0:
        # For count = 0 only, always return False so that loop continues
        return {'break': False}
    else:
        # Check whether dynamic range is increasing or minmax is decreasing. If
        # so, continue (return False)
        if (dynamic_range1 / rapthor > dynamic_range2) or (minmax1 * rapthor < minmax2):
            return {'break': False}
        else:
            return {'break': True}


if __name__ == '__main__':
    descriptiontext = "Compare dynamic range of two images.\n"

    parser = argparse.ArgumentParser(description=descriptiontext, formatter_class=RawTextHelpFormatter)
    parser.add_argument('im1', help='name of image #1')
    parser.add_argument('im2', help='name of image #2')
    parser.add_argument('rapthor', help='required improvement rapthor for success')
    args = parser.parse_args()

    main(args.im1, args.im2, rapthor=args.rapthor)
