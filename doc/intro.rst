=====================
Image Domain Gridding
=====================

Image Domain Gridding (IDG) is a fast method for convolutional resampling (gridding/degridding) of radio astronomical data (visibilities). Direction dependent effects (DDEs) or A-tems can be applied in the gridding process.

Working in the image domain avoids the computation of oversampled convolution functions.
This is especially advantageous when the DDEs vary on short time scales.

This library has implementation for both CPU and GPU.
The algorithm uses sin/cos evaluations and multiply-add operations on many small on grids.
This makes it somewhat costly on a CPU, but it makes a very good match
with GPUs with hardware support for sin/cos evaluations.
Gridding speeds of several GB/s on a single GPU device have been achieved (cite veenboer)

The algorithm is described in `Van der Tol 2018 <https://www.aanda.org/articles/aa/pdf/2018/08/aa32858-18.pdf>`_. The implementation is described in
`veenboer 2020 <https://www.sciencedirect.com/science/article/abs/pii/S2213133720300408>`_
`(pdf) <https://www.astron.nl/~romein/papers/ASCOM-20/paper.pdf>`_

This library is written in C++. To make it available for other languages a C interface
is available. And there are Python bindings.
