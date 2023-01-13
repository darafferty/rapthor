.. _operations:

Operations
==========

Most of the processing performed by Rapthor is done in "operations," which are sets of steps that are grouped together into pipelines. The available operations and the primary data products of each are described in detail below.


.. _calibrate:

Calibrate
---------

This operation calibrates the data using the current sky model. The exact steps done during calibration depend on the strategy, but essentially there are three main parts: a phase-only solve (diagonal Jones matrix) on short timescales (the "fast phase solve"), a slow phase-only (diagonal) solve, an amplitude-only (diagonal) solve on long time scales (the "slow-gain" solve), and processing of the resulting solutions, including smoothing and the generation of a-term images. This calibration strategy is based on the LBA strategy of the LiLF pipeline (https://github.com/revoltek/LiLF), with the idea that the same strategy can be used for both HBA and LBA (similar to the way the calibrator pipeline works in LINC). However, the strategy may evolve in the future depending on the results of the commissioning.

The fast and slow diagonal phase-only solve steps use different calibration parameters, such as different antenna constraints.

For calibration, Rapthor designates calibrators (bright, compact sources or groups of sources) and then tessellates the full sky model, using those calibrators as the facet/patch centers. This ensures that each facet/patch has a bright source in it, as that seems to be fairly important to get good solutions. Despite the designation of calibrators for the tesselation, all sources are still used in the calibration (not just the bright sources).

To model the sources, the clean components are grouped by PyBDSF using the sources that it found in the image. Then the mean shift algorithm is used to identify compact groups of sources (e.g., two nearby bright sources are better used together as a single “calibrator,” rather than as two separate ones) that are then used as the basis for tessellation.

When multiple nodes are available, this task is distributed.

Primary products (in ``solutions/calibrate_X``, where ``X`` is the cycle number):
    * ``field-solutions.h5`` - the calibration solution table containing both fast- and slow-solve solutions.


.. _predict:

Predict
-------

This operation predicts visibilities for subtraction. Sources that lie outside of imaged regions are subtracted, as are bright sources inside imaged regions (if desired).

When multiple nodes are available, this task is distributed.

Primary products (in ``scratch/``):
    * Temporary measurement sets used for the subsequent image operation.


.. _image:

Image (+ mosaic)
----------------

This operation images the data. If multiple imaging sectors are used, a mosaic operation is also run to mosaic the sector images together into a single image.

When multiple nodes are available, it is possible to distribute the imaging over multiple nodes. This is done by running wsclean-mp instead of wsclean. Currently, Toil does not fully support openmpi. To remedy this, a wrapping script around wsclean-mp is used on the 'master' node. Because of this, imaging can only use the worker nodes, and the master node is idle.

Subtracted sources are restored in the image pipeline (near the end, in a step named wsclean_restore).

Primary products (in ``images/image_X``, where ``X`` is the cycle number):
    * ``field-MFS-I-image.fits`` - the Stokes I image
