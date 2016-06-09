import os
import ctypes
import numpy
import numpy.ctypeslib

# A bit ugly, but ctypes.util's find_library does not look in
# the LD_LIBRARY_PATH, but only PATH. Howver, we can also provide
# the full path of the shared object file
path = os.path.dirname(os.path.realpath(__file__))
path, junk = os.path.split(path)
path, junk = os.path.split(path)
libpath = os.path.join(path, 'libidg-fft.so')
lib = ctypes.cdll.LoadLibrary(libpath)


def fft2f(matrix):
    """
    Perform in-place 2d FFT
    :param matrix: numpy.ndarray(shape=(m,n), dtype=numpy.complex64)
    """
    m = matrix.shape[0]
    n = matrix.shape[1]
    lib.fft2f.argtypes = [ctypes.c_int,
                          ctypes.c_int,
                          ctypes.c_void_p]
    lib.fft2f(ctypes.c_int(m),
              ctypes.c_int(n),
              matrix.ctypes.data_as(ctypes.c_void_p))


def ifft2f(matrix):
    """
    Perform in-place 2d inverse FFT
    :param matrix: numpy.ndarray(shape=(m,n), dtype=numpy.complex64)
    """
    m = matrix.shape[0]
    n = matrix.shape[1]
    lib.ifft2f.argtypes = [ctypes.c_int,
                           ctypes.c_int,
                           ctypes.c_void_p]
    lib.ifft2f(ctypes.c_int(m),
               ctypes.c_int(n),
               matrix.ctypes.data_as(ctypes.c_void_p))


def fft2f_r2c(matrix):
    """
    Perform real to complex 2d FFT
    :param matrix: numpy.ndarray(shape=(m,n), dtype=numpy.float32)
    :returns result: numpy.ndarray(shape=(m,n), dtype=numpy.complex64)
    """
    m = matrix.shape[0]
    n = matrix.shape[1]
    result = numpy.zeros(shape=(m,n), dtype=numpy.complex64)
    lib.fft2f_r2c.argtypes = [ctypes.c_int,
                              ctypes.c_int,
                              ctypes.c_void_p,
                              ctypes.c_void_p]
    lib.fft2f_r2c(ctypes.c_int(m),
                  ctypes.c_int(n),
                  matrix.ctypes.data_as(ctypes.c_void_p),
                  result.ctypes.data_as(ctypes.c_void_p))
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
    lib.ifft2f_c2r.argtypes = [ctypes.c_int,
                               ctypes.c_int,
                               ctypes.c_void_p,
                               ctypes.c_void_p]
    lib.ifft2f_c2r(ctypes.c_int(m),
                   ctypes.c_int(n),
                   matrix.ctypes.data_as(ctypes.c_void_p),
                   result.ctypes.data_as(ctypes.c_void_p))
    return result


def fftshift2f(matrix):
    """
    Perform in-place 2d FFT shift
    :param matrix: numpy.ndarray(shape=(m,n), dtype=numpy.complex64)
    """
    m = matrix.shape[0]
    n = matrix.shape[1]
    lib.fftshift2f.argtypes = [ctypes.c_int,
                               ctypes.c_int,
                               ctypes.c_void_p]
    lib.fftshift2f(ctypes.c_int(m),
                   ctypes.c_int(n),
                   matrix.ctypes.data_as(ctypes.c_void_p))


def ifftshift2f(matrix):
    """
    Perform in-place 2d FFT shift
    :param matrix: numpy.ndarray(shape=(m,n), dtype=numpy.complex64)
    """
    m = matrix.shape[0]
    n = matrix.shape[1]
    lib.ifftshift2f.argtypes = [ctypes.c_int,
                                ctypes.c_int,
                                ctypes.c_void_p]
    lib.ifftshift2f(ctypes.c_int(m),
                    ctypes.c_int(n),
                    matrix.ctypes.data_as(ctypes.c_void_p))


def resize2f_r2r(image, out_height, out_width):
    """
    Resize two dimesional "image"
    :param image: numpy.ndarray(shape=(m,n), dtype=numpy.float32)
    """
    m = image.shape[0]
    n = image.shape[1]
    result = numpy.zeros(shape=(out_height,out_width),
                         dtype=numpy.float32)
    lib.resize2f_r2r.argtypes = [ctypes.c_int,
                                 ctypes.c_int,
                                 ctypes.c_void_p,
                                 ctypes.c_int,
                                 ctypes.c_int,
                                 ctypes.c_void_p]
    lib.resize2f_r2r(ctypes.c_int(m),
                     ctypes.c_int(n),
                     image.ctypes.data_as(ctypes.c_void_p),
                     ctypes.c_int(out_height),
                     ctypes.c_int(out_width),
                     result.ctypes.data_as(ctypes.c_void_p))
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
    lib.resize2f_c2c.argtypes = [ctypes.c_int,
                                 ctypes.c_int,
                                 ctypes.c_void_p,
                                 ctypes.c_int,
                                 ctypes.c_int,
                                 ctypes.c_void_p]
    lib.resize2f_c2c(ctypes.c_int(m),
                     ctypes.c_int(n),
                     image.ctypes.data_as(ctypes.c_void_p),
                     ctypes.c_int(out_height),
                     ctypes.c_int(out_width),
                     result.ctypes.data_as(ctypes.c_void_p))
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
    lib.resize2_r2r.argtypes = [ctypes.c_int,
                                ctypes.c_int,
                                ctypes.c_void_p,
                                ctypes.c_int,
                                ctypes.c_int,
                                ctypes.c_void_p]
    lib.resize2_r2r(ctypes.c_int(m),
                    ctypes.c_int(n),
                    image.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(out_height),
                    ctypes.c_int(out_width),
                    result.ctypes.data_as(ctypes.c_void_p))
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
    lib.resize2_c2c.argtypes = [ctypes.c_int,
                                ctypes.c_int,
                                ctypes.c_void_p,
                                ctypes.c_int,
                                ctypes.c_int,
                                ctypes.c_void_p]
    lib.resize2_c2c(ctypes.c_int(m),
                    ctypes.c_int(n),
                    image.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(out_height),
                    ctypes.c_int(out_width),
                    result.ctypes.data_as(ctypes.c_void_p))
    return result
