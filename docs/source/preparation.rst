.. _data_preparation:

Data preparation
================

Rapthor requires that the input data be prepared using the LOFAR initial calibration (LINC) pipelines. These pipelines perform the calibration of the calibrator data, the removal of instrumental effects (e.g., station clock offsets), the setting of the overall amplitude scale, and the direction-independent calibration of the target data. The pipelines are available at https://git.astron.nl/RD/LINC and must be run before Rapthor can be used.

Input measurement sets
----------------------

The input data must have the direction-independent solutions applied to the DATA column (this is provided already by LINC) and be concatenated in frequency into a single MS file per observation [1]_. A script to perform this concatenation is included with Rapthor (``bin/concat_linc_files``). More that one input MS file can be supplied, but in this case each file must cover a different time range (e.g., interleaved observations or observations from multiple nights).

Sky model
---------

Manual skymodel input
~~~~~~~~~~~~~~~~~~~~~
Rapthor requires an initial model that is used during the first iteration. After the first iteration, Rapthor will use the model that is the result of the previous iteration. There are several ways to create a model, but if no model is available, we recommend creating one from TGSS. To make a TGSS model, go to https://tgssadr.strw.leidenuniv.nl/, and under "LOFAR Sky Model Creator" fill in the field's coordinates. For radius, 5 degrees can be used. The other default settings are fine (cut off at 0.3 Jy, and Deconvolve Beam: yes).

Automatic skymodel download
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Rapthor can also download a suitable skymodel automatically. If ``download_initial_skymodel`` is set to ``True`` in the parset, rapthor will use the additional ``download_initial_skymodel_radius`` and ``download_initial_skymodel_server`` to download a skymodel out to the given radius from the given source. See :ref:`rapthor_parset` for more explanation about these parameters.

.. [1] In the future, we will try to automate this.

