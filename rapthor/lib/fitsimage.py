"""
Definition of the FITSImage class for handling of FITS images
"""
from rapthor.lib import miscellaneous as misc
from astropy.wcs import WCS as pywcs
from astropy.io import fits as pyfits
from shapely.geometry import Polygon
import numpy as np
import sys
import logging
import re


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
        Find the primary beam headers following AIPS convenction
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
        """ Flatten a fits file so that it becomes a 2D image. Return new header and data """
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
        if filename is None:
            filename = self.imagefile
        pyfits.writeto(filename, self.img_data, self.img_hdr, overwrite=True)

    def get_beam(self):
        return [self.img_hdr['BMAJ'], self.img_hdr['BMIN'], self.img_hdr['BPA']]

    def get_wcs(self):
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
        Return the rms of all the pixels in an image
        niter : robust rms estimation
        eps : convergency criterion, if None is 0.1% of initial rms
        sampling : sampling interval to use to speed up the noise calculation (e.g.,
                   sampling = 4 means use every forth pixel)
        """
        if eps is None:
            eps = np.nanstd(self.img_data)*1e-3
        sampling = int(sampling)
        if sampling < 1:
            sampling = 1
        data = self.img_data[::sampling]  # sample array
        data = data[ np.isfinite(data) ]
        oldrms = 1.
        for i in range(niter):
            rms = np.nanstd(data)
            if np.abs(oldrms-rms)/rms < eps:
                self.noise = float(rms)
                logging.debug('%s: Noise: %.3f mJy/b' % (self.imagefile, self.noise*1e3))
                return

            data = data[np.abs(data)<3*rms]
            oldrms = rms
        raise Exception('Noise estimation failed to converge.')

    def convolve(self, target_beam):
        """
        Convolve *to* this rsolution
        beam = [bmaj, bmin, bpa]
        """
        from lib_beamdeconv import deconvolve_ell, EllipticalGaussian2DKernel
        from astropy import convolution

        # if difference between beam is negligible <1%, skip - it mostly happens when beams are exactly the same
        beam = self.get_beam()
        if (np.abs((target_beam[0]/beam[0])-1) < 1e-2) and (np.abs((target_beam[1]/beam[1])-1) < 1e-2) and (np.abs(target_beam[2] - beam[2]) < 1):
            logging.debug('%s: do not convolve. Same beam.' % self.imagefile)
            return
        # first find beam to convolve with
        convolve_beam = deconvolve_ell(target_beam[0], target_beam[1], target_beam[2], beam[0], beam[1], beam[2])
        if convolve_beam[0] is None:
            raise ValueError('Cannot deconvolve this beam.')
        logging.debug('%s: Convolve beam: %.3f" %.3f" (pa %.1f deg)'
                      % (self.imagefile, convolve_beam[0]*3600, convolve_beam[1]*3600, convolve_beam[2]))
        # do convolution on data
        bmaj, bmin, bpa = convolve_beam
        assert abs(self.img_hdr['CDELT1']) == abs(self.img_hdr['CDELT2'])
        pixsize = abs(self.img_hdr['CDELT1'])
        fwhm2sigma = 1./np.sqrt(8.*np.log(2.))
        gauss_kern = EllipticalGaussian2DKernel((bmaj*fwhm2sigma)/pixsize, (bmin*fwhm2sigma)/pixsize, (90+bpa)*np.pi/180.) # bmaj and bmin are in pixels
        self.img_data = convolution.convolve(self.img_data, gauss_kern, boundary=None)
        self.img_data *= (target_beam[0]*target_beam[1])/(beam[0]*beam[1]) # since we are in Jt/b we need to renormalise
        self.set_beam(target_beam) # update beam

    def apply_shift(self, dra, ddec):
        """
        Shift header by dra/ddec
        dra, ddec in degree
        """
        # correct the dra shift for np.cos(DEC*np.pi/180.) -- only in the log!
        logging.info('%s: Shift %.2f %.2f (arcsec)' % (self.imagefile, dra*3600*np.cos(self.dec*np.pi/180.), ddec*3600))
        dec = self.img_hdr['CRVAL2']
        self.img_hdr['CRVAL1'] += dra/(np.cos(np.pi*dec/180.))
        self.img_hdr['CRVAL2'] += ddec

    def calc_weight(self):
        self.weight_data = np.ones_like(self.img_data)
        self.weight_data[self.img_data == 0] = 0
        self.weight_data /= self.noise * self.scale
        self.weight_data = self.weight_data**2.0
