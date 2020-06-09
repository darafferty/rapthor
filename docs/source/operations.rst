.. _operations:

Operations
==========

Most of the processing performed by rapthor is done in "operations," which are sets of steps that are grouped together. The available operations and the primary data products of each are described in detail below.


.. _calibrate:

calibrate
---------

This operation calibrates the data using the current sky model.

Primary products (in ``solutions/calibrate_X``, where ``X`` is the cycle number):
    * ``field-solutions.h5`` - the calibration solutions table


.. _predict:

predict
-------

This operation predicts visibilities for subtraction.

Primary products (in ``scratch/``):
    * Temporary measurement sets used for the image operation.


.. _image:

image (+ mosaic)
----------------

This operation images the data. If multiple imaging sectors are used, a mosaic operation is also run to mosaic the sector images together.

Primary products (in ``images/image_X``, where ``X`` is the cycle number):
    * ``field-MFS-I-image.fits`` - the Stokes I image
