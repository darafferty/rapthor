"""
Definition of the FITSImage class for handling of FITS images
"""
from rapthor.lib import miscellaneous as misc
from astropy.wcs import WCS as pywcs
from astropy.io import fits as pyfits
from shapely.geometry import Polygon
import numpy as np
import logging
import re
from pathlib import Path


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
            w = pywcs(f[0].header)
            wn = pywcs(naxis=2)
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
        return pywcs(self.img_hdr)

    def blank(self, vertices_file=None):
        """
        Blank pixels (NaN) outside of polygon region
        """
        # Construct polygon
        if vertices_file is None:
            vertices_file = self.vertices_file
        vertices = misc.read_vertices(vertices_file)

        w = pywcs(self.header)
        RAind = w.axis_type_names.index('RA')
        Decind = w.axis_type_names.index('DEC')
        RAverts = vertices[0]
        Decverts = vertices[1]
        verts = []
        for RAvert, Decvert in zip(RAverts, Decverts):
            ra_dec = np.array([[0.0, 0.0, 0.0, 0.0]])
            ra_dec[0][RAind] = RAvert
            ra_dec[0][Decind] = Decvert
            verts.append((w.wcs_world2pix(ra_dec, 0)[0][RAind], w.wcs_world2pix(ra_dec, 0)[0][Decind]))
        poly = Polygon(verts)
        poly_padded = poly.buffer(2)
        verts = [(xi, yi) for xi, yi in zip(poly_padded.exterior.coords.xy[0].tolist(),
                                            poly_padded.exterior.coords.xy[1].tolist())]

        # Blank pixels (= NaN) outside of the polygon
        self.img_data = misc.rasterize(verts, self.img_data, blank_value=np.nan)

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
    imagefiles : list of str
        List of filenames of the FITS channel images
    """
    def __init__(self, imagefiles):
        self.imagefiles = imagefiles
        self.name = 'FITS_cube'

        self.channel_images = []
        for imagefile in self.imagefiles:
            self.channel_images.append(FITSImage(imagefile))

        self.check_channel_images()
        self.order_channel_images()
        self.make_header()
        self.make_data()

    def check_channel_images(self):
        """
        Check the input channel images for problems
        """
        image_ch0 = self.channel_images[0]
        wcs_ch0 = image_ch0.get_wcs()
        for image in self.channel_images:
            # Check that all channels have the same data shape
            if image.img_data.shape != image_ch0.img_data.shape:
                raise ValueError('Data shape for channel image {0} differs from that of '
                                 '{1}'.format(image.imagefile, image_ch0.imagefile))

            # Check that all channels have the same WCS parameters
            image_wcs = image.get_wcs()
            for wcs_attr in ['crpix', 'cdelt', 'crval', 'ctype']:
                if not np.all(np.array(getattr(image_wcs.wcs, wcs_attr)) ==
                              np.array(getattr(wcs_ch0.wcs, wcs_attr))):
                    raise ValueError('WCS for channel image {0} differs from that of '
                                     '{1}'.format(image.imagefile, image_ch0.imagefile))

    def order_channel_images(self):
        """
        Order the input channel images by frequency
        """
        frequencies = np.array([image.freq for image in self.channel_images])
        sort_idx = np.argsort(frequencies)
        self.frequencies = frequencies[sort_idx]
        self.imagefiles = np.array(self.imagefiles)[sort_idx].tolist()
        self.channel_images = np.array(self.channel_images)[sort_idx].tolist()

    def make_header(self):
        """
        Make the cube header
        """
        # Use the header from the first channel image as the template
        self.header = self.channel_images[0].img_hdr

        # Add a frequecy axis to the header
        self.header['NAXIS'] = 3
        self.header['NAXIS3'] = len(self.frequencies)
        self.header['CRPIX3'] = 1
        self.header['CDELT3'] = np.diff(self.frequencies).mean().value
        self.header['CTYPE3'] = 'FREQ'
        self.header['CRVAL3'] = self.frequencies[0]
        self.header["CUNIT3"] = "Hz"

    def make_data(self):
        """
        Make the cube data
        """
        # Set the shape to [nchannels, imsize_0, imsize_1]
        cube_shape = [len(self.channel_images)]
        cube_shape.extend(self.channel_images[0].img_data.shape)

        self.data = np.zeros(cube_shape)

        for i, image in enumerate(self.channel_images):
            self.data[i, :] = image.img_data

    def write(self, filename=None):
        """
        Write the image cube to a FITS file

        Parameters
        ----------
        filename : str
            Filename of the output FITS file
        """
        if filename is None:
            filename = f'{Path(self.imagefiles[0]).stem}_cube.fits'

        pyfits.writeto(filename, self.data, self.header, overwrite=True)

    def write_frequencies(self, filename=None):
        """
        Write the channel frequencies to a text file

        Note: the frequencies are written one per line in Hz

        Parameters
        ----------
        filename : str
            Filename of the output text file
        """
        if filename is None:
            filename = f'{Path(self.imagefiles[0]).stem}_frequencies.txt'

        frequencies = []
        for image in self.channel_images:
            frequencies.append(f'{image.freq}')

        with open(filename, 'w') as f:
            f.writelines(', '.join(frequencies))

    def write_beams(self, filename=None):
        """
        Write the channel beam parameters to a text file

        Note: the beams are written one per line as follows:
            (major axis, minor axis, position angle)
        with all values being in degrees

        Parameters
        ----------
        filename : str
            Filename of the output text file
        """
        if filename is None:
            filename = f'{Path(self.imagefiles[0]).stem}_beams.txt'

        beams = []
        for image in self.channel_images:
            beams.append(f'{tuple(image.beam)}')

        with open(filename, 'w') as f:
            f.writelines(', '.join(beams))
