.. _data_preparation:

Data preparation
================

Rapthor requires that the input data be prepared using initial calibration pipelines (LINC for LOFAR, INST and BPP for SKA-low). These pipelines perform the calibration of the calibrator data, the removal of instrumental effects (e.g., station clock offsets), the setting of the overall amplitude scale, and the direction-independent calibration of the target data. LOFAR pipelines are available at https://git.astron.nl/RD/LINC and must be run before Rapthor can be used.

Input measurement sets
----------------------

The input data must have the direction-independent solutions applied to the DATA column (this is provided already by LINC for LOFAR data). The multiple frequency bands output by LINC can be input directly to Rapthor (no concatenation is needed). Data from multiple epochs, such as interleaved observations or observations from multiple nights, are supported.

Sky model
---------

Manual sky model input
~~~~~~~~~~~~~~~~~~~~~~
If self calibration is to be done (see :ref:`rapthor_strategy` for information about defining a processing strategy), Rapthor requires an initial model to start the calibration of the first cycle (after the first cycle, Rapthor will use the model that is the result of the previous cycle). If such a model is available, it can be specified using the :term:`input_skymodel` option in the parset. Typically, however, an initial model is not available, in which case Rapthor can either generate the model (recommended) or download one. See below for more information on these options.

The input sky model consists of a list of sources stored in a plain text file, their coordinates, flux density, and other relevant parameters. For the curious reader, an overview of the data format is available in the `WSClean manual <https://wsclean.readthedocs.io/en/latest/component_list.html>`_.

.. _auto_sky_generation:

Automatic sky model generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If an initial sky model is unavailable, Rapthor can generate the sky model automatically from the input data. The generation can be activated by setting :term:`generate_initial_skymodel` to ``True`` in the parset. Rapthor will then image the full field and generate the initial model from the resulting clean components.

This method will usually result in a higher-quality sky model than can be obtained through downloading a model from a catalog. Additionally, although the generation of the model typically requires much more time than the download, the higher quality of the model means that many of the early self calibration cycles can be skipped, resulting in overall less time being required for a full reduction.

.. note::

    The default "selfcal" strategy will skip the phase-only self calibration cycles if automatic sky model generation is used. See :ref:`rapthor_strategy` for details.


.. _auto_sky_download:

Automatic sky model download
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Rapthor can also download a suitable sky model automatically. If :term:`download_initial_skymodel` is set to ``True`` in the parset, rapthor will use the additional :term:`download_initial_skymodel_radius` and :term:`download_initial_skymodel_server` to download a sky model out to the given radius from the given source. See :ref:`rapthor_parset` for more explanation about these parameters.
