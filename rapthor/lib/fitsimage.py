"""
Definition of classes for handling of FITS images
"""
import logging
import re
from pathlib import Path

import numpy as np
from astropy.io import fits as pyfits
from astropy.wcs import WCS
from lsmtool.io import read_vertices
from lsmtool.utils import rasterize
from shapely.geometry import Polygon


class FITSImage(object):
    """
    The FITSImage class is used for processing/manipulation of FITS images

    Parameters
    ----------
    imagefile : str
        Filename of the FITS file
    """
    def __init__(self, imagefile):
        self.imagefile = imagefile
        self.name = 'FITS_image'
        self.vertices_file = None
        self.scale = 1.
        self.shift = 0.
        self.noise = 1.

        self.header = pyfits.open(imagefile)[0].header
        self.find_beam()
        self.find_freq()
        self.flatten()
        self.ra = self.img_hdr['CRVAL1']
        self.dec = self.img_hdr['CRVAL2']

    def find_beam(self):
        """
        Find the primary beam headers following AIPS convention
        """
        if ('BMAJ' in self.header) and ('BMIN' in self.header) and ('PA' in self.header):
            pass
        elif 'HISTORY' in self.header:
            for hist in self.header['HISTORY']:
                if 'AIPS   CLEAN BMAJ' in hist:
                    # remove every letter from the string
                    bmaj, bmin, pa = re.sub(' +', ' ', re.sub('[A-Z ]*=', '', hist)).strip().split(' ')
                    self.header['BMAJ'] = float(bmaj)
                    self.header['BMIN'] = float(bmin)
                    self.header['BPA'] = float(pa)
        self.beam = [float(self.header['BMAJ']), float(self.header['BMIN']), float(self.header['BPA'])]

    def find_freq(self):
        """
        Find frequency value in most common places of a fits header
        """
        self.freq = None
        if not self.header.get('RESTFREQ') is None and not self.header.get('RESTFREQ') == 0:
            self.freq = float(self.header.get('RESTFREQ'))
        elif not self.header.get('FREQ') is None and not self.header.get('FREQ') == 0:
            self.freq = float(self.header.get('FREQ'))
        else:
            for i in range(5):
                type_s = self.header.get('CTYPE%i' % i)
                if type_s is not None and type_s[0:4] == 'FREQ':
                    self.freq = float(self.header.get('CRVAL%i' % i))

    def flatten(self):
        """
        Flatten a FITS image so that it becomes a 2D image
        """
        f = pyfits.open(self.imagefile)

        naxis = f[0].header['NAXIS']
        if naxis < 2:
            raise RuntimeError('Can\'t make map from this')
        if naxis == 2:
            self.img_hdr = f[0].header
            self.img_data = f[0].data
        else:
            w = WCS(f[0].header)
            wn = WCS(naxis=2)
            wn.wcs.crpix[0] = w.wcs.crpix[0]
            wn.wcs.crpix[1] = w.wcs.crpix[1]
            wn.wcs.cdelt = w.wcs.cdelt[0:2]
            wn.wcs.crval = w.wcs.crval[0:2]
            wn.wcs.ctype[0] = w.wcs.ctype[0]
            wn.wcs.ctype[1] = w.wcs.ctype[1]

            header = wn.to_header()
            header["NAXIS"] = 2
            header["NAXIS1"] = f[0].header['NAXIS1']
            header["NAXIS2"] = f[0].header['NAXIS2']
            header["FREQ"] = self.freq
            header['RESTFREQ'] = self.freq
            header['BMAJ'] = self.beam[0]
            header['BMIN'] = self.beam[1]
            header['BPA'] = self.beam[2]
            copy = ('EQUINOX', 'EPOCH')
            for k in copy:
                r = f[0].header.get(k)
                if r:
                    header[k] = r

            dataslice = []
            for i in range(naxis, 0, -1):
                if i <= 2:
                    dataslice.append(np.s_[:],)
                else:
                    dataslice.append(0)
            self.img_hdr = header
            self.img_data = f[0].data[tuple(dataslice)]
        self.min_value = float(np.nanmin(self.img_data))
        self.max_value = float(np.nanmax(self.img_data))
        self.mean_value = float(np.nanmean(self.img_data))
        self.median_value = float(np.nanmedian(self.img_data))

    def write(self, filename=None):
        """
        Write the image to a FITS file

        Parameters
        ----------
        filename : str
            Filename of the output FITS file
        """
        if filename is None:
            filename = self.imagefile
        pyfits.writeto(filename, self.img_data, self.img_hdr, overwrite=True)

    def get_beam(self):
        """
        Return the beam for the image
        """
        return [self.img_hdr['BMAJ'], self.img_hdr['BMIN'], self.img_hdr['BPA']]

    def get_wcs(self):
        """
        Return the WCS object for the image
        """
        return WCS(self.img_hdr)

    def blank(self, vertices_file=None):
        """
        Blank pixels (NaN) outside of polygon region
        """
        # Construct polygon
        if vertices_file is None:
            vertices_file = self.vertices_file
        vertices = read_vertices(vertices_file, WCS(self.header))
        poly = Polygon(vertices)
        poly_padded = poly.buffer(2)
        vertices = list(zip(poly_padded.exterior.coords.xy[0].tolist(),
                            poly_padded.exterior.coords.xy[1].tolist()))

        # Blank pixels (= NaN) outside of the polygon
        self.img_data = rasterize(verts, self.img_data, blank_value=np.nan)

    def calc_noise(self, niter=1000, eps=None, sampling=4):
        """
        Calculate the noise (rms) of all the pixels in an image

        Parameters
        ----------
        niter : float, optional
            Maximum number of iterations to perform for robust rms estimation
        eps : float, optional
            Fractional improvement in rms used to determine convergency. If None, a
            value of 0.1% of the initial rms is used
        sampling : int, optional
            Sampling interval to use to speed up the noise calculation (e.g.,
            sampling = 4 means use every forth pixel)
        """
        if eps is None:
            eps = np.nanstd(self.img_data)*1e-3
        sampling = int(sampling)
        if sampling < 1:
            sampling = 1
        data = self.img_data[::sampling]  # sample array
        data = data[np.isfinite(data)]
        oldrms = 1.
        for i in range(niter):
            rms = np.nanstd(data)
            if np.abs(oldrms-rms)/rms < eps:
                self.noise = float(rms)
                logging.debug('%s: Noise: %.3f mJy/b' % (self.imagefile, self.noise*1e3))
                return

            data = data[np.abs(data) < 3*rms]
            oldrms = rms
        raise Exception('Noise estimation failed to converge.')

    def apply_shift(self, dra, ddec):
        """
        Shift header by dra/ddec

        Parameters
        ----------
        dra : float
            Shift in RA in degrees
        ddec : float
            Shift in Dec in degrees
        """
        # correct the dra shift for np.cos(DEC*np.pi/180.) -- only in the log!
        logging.info('%s: Shift %.2f %.2f (arcsec)' % (self.imagefile, dra*3600*np.cos(self.dec*np.pi/180.), ddec*3600))
        dec = self.img_hdr['CRVAL2']
        self.img_hdr['CRVAL1'] += dra/(np.cos(np.pi*dec/180.))
        self.img_hdr['CRVAL2'] += ddec

    def calc_weight(self):
        """
        Calculate the weights for the image
        """
        self.weight_data = np.ones_like(self.img_data)
        self.weight_data[self.img_data == 0] = 0
        self.weight_data /= self.noise * self.scale
        self.weight_data = self.weight_data**2.0


class FITSCube(object):
    """
    The FITSCube class is used for processing/manipulation of FITS image cubes

    Parameters
    ----------
    channel_imagefiles : list of str
        List of filenames of the FITS channel images (one frequency channel per
        image). Note: If more than one frequency channel is present in a given
        input image, the lowest-frequency channel is used and any other channels
        are ignored
    """
    def __init__(self, channel_imagefiles):
        self.channel_imagefiles = channel_imagefiles
        self.name = 'FITS_cube'

        self.channel_images = []
        for channel_imagefile in self.channel_imagefiles:
            self.channel_images.append(FITSImage(channel_imagefile))
        if not self.channel_images:
            raise ValueError('No valid channel images were found')

        self.check_channel_images()
        self.order_channel_images()
        self.make_header()
        self.make_data()

    def check_channel_images(self):
        """
        Check the input channel images for problems
        """
        image_ch0 = self.channel_images[0]
        wcs_ch0 = image_ch0.get_wcs().wcs
        for channel_image in self.channel_images:
            # Check that all channels have the same data shape
            if channel_image.img_data.shape != image_ch0.img_data.shape:
                raise ValueError('Data shape for channel image {0} differs from that of '
                                 '{1}'.format(channel_image.imagefile, image_ch0.imagefile))

            # Check that all channels have the same WCS parameters
            channel_wcs = channel_image.get_wcs().wcs
            for wcs_attr in ['crpix', 'cdelt', 'crval', 'ctype']:
                if wcs_attr == 'ctype':
                    # Check string values
                    values_agree = np.all(np.array(getattr(channel_wcs, wcs_attr)) ==
                                          np.array(getattr(wcs_ch0, wcs_attr)))
                else:
                    # Check float values
                    values_agree = np.all(np.isclose(np.array(getattr(channel_wcs, wcs_attr)),
                                                     np.array(getattr(wcs_ch0, wcs_attr))))
                if not values_agree:
                    raise ValueError('WCS {0} value for channel image {1} differs from that of '
                                     '{2}'.format(wcs_attr, channel_image.imagefile, image_ch0.imagefile))

    def order_channel_images(self):
        """
        Order the input channel images by frequency
        """
        frequencies = np.array([channel_image.freq for channel_image in self.channel_images])
        sort_idx = np.argsort(frequencies)
        self.channel_frequencies = frequencies[sort_idx]
        self.channel_imagefiles = np.array(self.channel_imagefiles)[sort_idx].tolist()
        self.channel_images = np.array(self.channel_images)[sort_idx].tolist()

    def make_header(self):
        """
        Make the cube header
        """
        # Use the header from the first channel image as the template
        self.header = self.channel_images[0].img_hdr

        # Add a frequecy axis to the header
        self.header['NAXIS'] = 3
        self.header['NAXIS3'] = len(self.channel_frequencies)
        self.header['CRPIX3'] = 1
        self.header['CDELT3'] = np.diff(self.channel_frequencies).mean()
        self.header['CTYPE3'] = 'FREQ'
        self.header['CRVAL3'] = self.channel_frequencies[0]
        self.header["CUNIT3"] = "Hz"

    def make_data(self):
        """
        Make the cube data
        """
        # Set the shape to [nchannels, imsize_0, imsize_1]
        cube_shape = [len(self.channel_images)]
        cube_shape.extend(self.channel_images[0].img_data.shape)

        self.data = np.empty(cube_shape)

        for i, channel_image in enumerate(self.channel_images):
            self.data[i, :] = channel_image.img_data

    def write(self, filename=None):
        """
        Write the image cube to a FITS file

        Parameters
        ----------
        filename : str, optional
            Filename of the output FITS file. If None, it is made by adding the
            suffix "_cube.fits" to the filename of the lowest-frequency input
            channel image
        """
        if filename is None:
            filename = f'{Path(self.channel_imagefiles[0]).stem}_cube.fits'

        pyfits.writeto(filename, self.data, self.header, overwrite=True)

    def write_frequencies(self, filename=None):
        """
        Write the channel frequencies to a text file

        Note: the frequencies are written as comma-separated values
        in Hz

        Parameters
        ----------
        filename : str, optional
            Filename of the output text file. If None, it is made by adding the
            suffix "_frequencies.txt" to the filename of the lowest-frequency
            input channel image
        """
        if filename is None:
            filename = f'{Path(self.channel_imagefiles[0]).stem}_frequencies.txt'

        with open(filename, 'w') as f:
            f.writelines(', '.join([f'{channel_frequecy}' for channel_frequecy in
                                    self.channel_frequencies]))

    def write_beams(self, filename=None):
        """
        Write the channel beam parameters to a text file

        Note: each beam is written as a tuple as follows:
            (major axis, minor axis, position angle)
        with all values being in degrees. The beams are written to the
        file as comma-separated tuples

        Parameters
        ----------
        filename : str, optional
            Filename of the output text file. If None, it is made by adding the
            suffix "_beams.txt" to the filename of the lowest-frequency input
            channel image
        """
        if filename is None:
            filename = f'{Path(self.channel_imagefiles[0]).stem}_beams.txt'

        with open(filename, 'w') as f:
            f.writelines(', '.join([f'{tuple(channel_image.beam)}' for channel_image in
                                    self.channel_images]))
