.. _operations:

Operations
==========

Most of the processing performed by Rapthor is done in "operations," which are sets of steps that are grouped together into pipelines. The available operations and the primary data products of each are described in detail below.


.. _calibrate:

Calibrate
---------

This operation calibrates the data using the current sky model. The exact steps done during calibration depend on the strategy, but essentially there are three main parts: a phase-only solve on short timescales (the "fast-phase" solve), a solve on long time scales (the "slow-gain" solve, which consists of two actual solves with different constraints), and processing of the resulting solutions, including smoothing and the generation of a-term images.

Primary products (in ``solutions/calibrate_X``, where ``X`` is the cycle number):
    * ``field-solutions.h5`` - the calibration solution table containing both fast- and slow-solve solutions.


.. _predict:

Predict
-------

This operation predicts visibilities for subtraction. Sources that lie outside of imaged regions are subtracted, as are bright sources inside imaged regions (if desired).

Primary products (in ``scratch/``):
    * Temporary measurement sets used for the subsequent image operation.


.. _image:

Image (+ mosaic)
----------------

This operation images the data. If multiple imaging sectors are used, a mosaic operation is also run to mosaic the sector images together into a single image.

Primary products (in ``images/image_X``, where ``X`` is the cycle number):
    * ``field-MFS-I-image.fits`` - the Stokes I image
