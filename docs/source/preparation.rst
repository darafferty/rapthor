.. _data_preparation:

Data preparation
================

Rapthor requires that the input data be prepared using the Linc pipelines. These pipelines perform the calibration of the calibrator data, the removal of instrumental effects (e.g., station clock offsets), the setting of the overall amplitude scale, the calibration of the target data, and the subtraction of sources outside of the target field. The pipelines are available at https://git.astron.nl/RD/LINC and must be run before Rapthor can be used.

Input measurement sets
----------------------

The input data must have the direction-independent solutions applied to the DATA column (this is provided already by Linc) and be concatenated in frequency into a single MS file per observation [1]_. A script to perform this concatenation is included with Rapthor (``bin/concat_linc_files``).

Sky model
---------

Rapthor requires an initial model that is used during the first iteration. After the first iteration, Rapthor will use the model that is the result of the previous iteration. There are several ways to create a model, but if no model is available, we recommend creating one from TGSS [1]_. To make a TGSS model, go to https://tgssadr.strw.leidenuniv.nl/, and under "LOFAR Sky Model Creator" fill in the field's coordinates. For radius, 5 degrees can be used. The other default settings are fine (cut off at 0.3 Jy, and Deconvolve Beam: yes). 

.. [1] In the future, we will try to automate this.
       
