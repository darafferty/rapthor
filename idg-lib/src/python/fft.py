# Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later

import os
import ctypes
import numpy
import numpy.ctypeslib
import idg

_fftlib = idg.load_library('libidg-fft.so')

def fft2f(matrix):
    """
    Perform in-place 2d FFT
    :param matrix: numpy.ndarray(shape=(m,n), dtype=numpy.complex64)
    """
    m = matrix.shape[0]
    n = matrix.shape[1]
    fftlib.fft2f.argtypes = [ctypes.c_int,
                          ctypes.c_int,
                          ctypes.c_void_p]
    fftlib.fft2f(ctypes.c_int(m),
              ctypes.c_int(n),
              matrix.ctypes.data)


def ifft2f(matrix):
    """
    Perform in-place 2d inverse FFT
    :param matrix: numpy.ndarray(shape=(m,n), dtype=numpy.complex64)
    """
    m = matrix.shape[0]
    n = matrix.shape[1]
    fftlib.ifft2f.argtypes = [ctypes.c_int,
                           ctypes.c_int,
                           ctypes.c_void_p]
    fftlib.ifft2f(ctypes.c_int(m),
               ctypes.c_int(n),
               matrix.ctypes.data)


def fft2f_r2c(matrix):
    """
    Perform real to complex 2d FFT
    :param matrix: numpy.ndarray(shape=(m,n), dtype=numpy.float32)
    :returns result: numpy.ndarray(shape=(m,n), dtype=numpy.complex64)
    """
    m = matrix.shape[0]
    n = matrix.shape[1]
    result = numpy.zeros(shape=(m,n), dtype=numpy.complex64)
    fftlib.fft2f_r2c.argtypes = [ctypes.c_int,
                              ctypes.c_int,
                              ctypes.c_void_p,
                              ctypes.c_void_p]
    fftlib.fft2f_r2c(ctypes.c_int(m),
                  ctypes.c_int(n),
                  matrix.ctypes.data,
                  result.ctypes.data)
    return result


def ifft2f_c2r(matrix):
    """
    Perform complex to real 2d inverse FFT
    :param matrix: numpy.ndarray(shape=(m,n), dtype=numpy.complex64)
    :returns result: numpy.ndarray(shape=(m,n), dtype=numpy.float32)
    """
    m = matrix.shape[0]
    n = matrix.shape[1]
    result = numpy.zeros(shape=(m,n), dtype=numpy.float32)
    fftlib.ifft2f_c2r.argtypes = [ctypes.c_int,
                               ctypes.c_int,
                               ctypes.c_void_p,
                               ctypes.c_void_p]
    fftlib.ifft2f_c2r(ctypes.c_int(m),
                   ctypes.c_int(n),
                   matrix.ctypes.data,
                   result.ctypes.data)
    return result


def fftshift2f(matrix):
    """
    Perform in-place 2d FFT shift
    :param matrix: numpy.ndarray(shape=(m,n), dtype=numpy.complex64)
    """
    m = matrix.shape[0]
    n = matrix.shape[1]
    fftlib.fftshift2f.argtypes = [ctypes.c_int,
                               ctypes.c_int,
                               ctypes.c_void_p]
    fftlib.fftshift2f(ctypes.c_int(m),
                   ctypes.c_int(n),
                   matrix.ctypes.data)


def ifftshift2f(matrix):
    """
    Perform in-place 2d FFT shift
    :param matrix: numpy.ndarray(shape=(m,n), dtype=numpy.complex64)
    """
    m = matrix.shape[0]
    n = matrix.shape[1]
    fftlib.ifftshift2f.argtypes = [ctypes.c_int,
                                ctypes.c_int,
                                ctypes.c_void_p]
    fftlib.ifftshift2f(ctypes.c_int(m),
                    ctypes.c_int(n),
                    matrix.ctypes.data)


def resize2f_r2r(image, out_height, out_width):
    """
    Resize two dimesional "image"
    :param image: numpy.ndarray(shape=(m,n), dtype=numpy.float32)
    """
    m = image.shape[0]
    n = image.shape[1]
    result = numpy.zeros(shape=(out_height,out_width),
                         dtype=numpy.float32)
    fftlib.resize2f_r2r.argtypes = [ctypes.c_int,
                                 ctypes.c_int,
                                 ctypes.c_void_p,
                                 ctypes.c_int,
                                 ctypes.c_int,
                                 ctypes.c_void_p]
    fftlib.resize2f_r2r(ctypes.c_int(m),
                     ctypes.c_int(n),
                     image.ctypes.data,
                     ctypes.c_int(out_height),
                     ctypes.c_int(out_width),
                     result.ctypes.data)
    return result


def resize2f_c2c(image, out_height, out_width):
    """
    Resize two dimesional "image"
    :param image: numpy.ndarray(shape=(m,n), dtype=numpy.complex64)
    """
    m = image.shape[0]
    n = image.shape[1]
    result = numpy.zeros(shape=(out_height,out_width),
                         dtype=numpy.complex64)
    fftlib.resize2f_c2c.argtypes = [ctypes.c_int,
                                 ctypes.c_int,
                                 ctypes.c_void_p,
                                 ctypes.c_int,
                                 ctypes.c_int,
                                 ctypes.c_void_p]
    fftlib.resize2f_c2c(ctypes.c_int(m),
                     ctypes.c_int(n),
                     image.ctypes.data,
                     ctypes.c_int(out_height),
                     ctypes.c_int(out_width),
                     result.ctypes.data)
    return result


def resize2_r2r(image, out_height, out_width):
    """
    Resize two dimesional "image"
    :param image: numpy.ndarray(shape=(m,n), dtype=numpy.float64)
    """
    m = image.shape[0]
    n = image.shape[1]
    result = numpy.zeros(shape=(out_height,out_width),
                         dtype=numpy.float64)
    fftlib.resize2_r2r.argtypes = [ctypes.c_int,
                                ctypes.c_int,
                                ctypes.c_void_p,
                                ctypes.c_int,
                                ctypes.c_int,
                                ctypes.c_void_p]
    fftlib.resize2_r2r(ctypes.c_int(m),
                    ctypes.c_int(n),
                    image.ctypes.data,
                    ctypes.c_int(out_height),
                    ctypes.c_int(out_width),
                    result.ctypes.data)
    return result


def resize2_c2c(image, out_height, out_width):
    """
    Resize two dimesional "image"
    :param image: numpy.ndarray(shape=(m,n), dtype=numpy.complex128)
    """
    m = image.shape[0]
    n = image.shape[1]
    result = numpy.zeros(shape=(out_height,out_width),
                         dtype=numpy.complex128)
    fftlib.resize2_c2c.argtypes = [ctypes.c_int,
                                ctypes.c_int,
                                ctypes.c_void_p,
                                ctypes.c_int,
                                ctypes.c_int,
                                ctypes.c_void_p]
    fftlib.resize2_c2c(ctypes.c_int(m),
                    ctypes.c_int(n),
                    image.ctypes.data,
                    ctypes.c_int(out_height),
                    ctypes.c_int(out_width),
                    result.ctypes.data)
    return result
