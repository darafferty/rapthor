.. _products:

Output
======

Rapthor produces the following output inside the working directory:

``images/``
    Directory containing the FITS images. See :ref:`image` for a detailed description of the images.

``logs/``
    Directory containing the log files. The main log file for the run is called ``rapthor.log``. The detailed logs of each step of the processing can be found in the subdirectories, one subdirectory per operation. The directory also contains:

    * ``commands.jsonl`` - a list of every DP3, WSClean, or other command that was run, with its command line, run time, and resource use.
    * ``tasks.jsonl`` - a list of every step that was run, with its run time.
    * ``diagnostics.txt`` - a summary of the selfcal, calibration, and image diagnostics of the run, written when the run finishes.

    See :ref:`monitoring_rapthor` for details.

``pipelines/``
    Directory containing intermediate files of each operation. Once a run has finished successfully, this directory can be removed.

    Each operation has its own subdirectory, which also holds the files that Rapthor uses to resume a run: ``pipeline_inputs.json`` and ``pipeline_outputs.json`` list the inputs and outputs of the operation, and ``.done`` and ``.outputs.json`` are written once the operation has finished. An operation with a ``.done`` file is skipped when the run is resumed (see :ref:`resuming_rapthor`).

``plots/``
    Directory containing the plots of the calibration solutions and images. See :ref:`calibrate` and :ref:`image` for a detailed description of the plots. Overview plots of the field, showing the coverage of the calibration model and the images for each cycle, are also placed here (``field_overview_X.png``, where ``X`` is the cycle number).

``regions/``
    Directory containing ds9 region files. These regions define the imaged areas and the facet layout (if used).

``skymodels/``
    Directory containing sky model files. See :ref:`calibrate` and :ref:`image` for a detailed description of the sky models.

``solutions/``
    Directory containing the calibration solution h5parm files. See :ref:`calibrate` for a detailed description of the solution files.

``visibilities/``
    Directory containing the MS files with the visibilities used in imaging (only if the :term:`save_visibilities` or :term:`save_residual_visibilities` parameter is set to ``True``). See :ref:`image` for a detailed description of the MS files.
