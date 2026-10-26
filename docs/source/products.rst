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

    Each operation has its own subdirectory, which also holds the files that Rapthor uses to resume a run: ``pipeline_inputs.json`` and ``pipeline_outputs.json`` list the inputs and outputs of the operation, and ``.done`` and ``.outputs.json`` are written once the operation has finished. An operation with a ``.done`` file is skipped when the run is resumed (see :ref:`resuming_rapthor`). A product named ``.<name>.partial`` was still being made, or removed, when the run stopped.

    A calibration operation writes the plots of each solve to its own subdirectory, for example ``pipelines/calibrate_1/plots/solve1_phase/``. Only the plots listed in the operation's outputs are copied to ``plots/``; a resumed run replaces the plots of an interrupted attempt.

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


.. _model_mosaics:

Model images of several sectors
-------------------------------

When several sectors are imaged, their images are combined into mosaics (see
:ref:`image`). Most images are regridded onto the mosaic grid, with blank
pixels outside all sectors. The model images ``field-MFS-model-pb.fits`` and
``field-MFS-filtered-model.fits`` (see :term:`save_filtered_model_image`) are
made as set by :term:`model_mosaic_method`:

``wsclean`` (default)
    WSClean draws the images from the sky models of the sectors, without a
    restoring beam: ``field-MFS-model-pb.fits`` from the true-sky sky models
    used for calibration in the next cycle, and
    ``field-MFS-filtered-model.fits`` from the apparent-sky ones. If
    :term:`filter_skymodel` is ``True``, these sky models are filtered, so the
    images contain only the clean components that were kept. A component that
    lies between pixels is spread over the nearby pixels, with small positive
    and negative ripples, but its total flux is kept. A sector whose sky model
    has no sources adds nothing; if no sector has any sources, the image
    contains only zeros, but it is still made so that later cycles and resumed
    runs find it as usual.

    The images cover the same area, with the same pixel size, as the other
    mosaics, but on the WSClean pixel grid centred on the mosaic: they have no
    blank pixels, are one pixel larger along an axis with an odd number of
    pixels, and are offset by half a pixel along an axis with an even number.
    Compare them with the other images by sky position, not pixel by pixel.

    WSClean cannot draw a sky model with one thread. If an image of the mosaic
    takes more than a quarter of the memory of the node, WSClean would use
    only one thread, so the mosaic operation stops with an error; use
    ``sparse_fits`` for such a mosaic.

``sparse_fits``
    The model images of the sectors are mosaicked: the unfiltered true-sky
    models made by WSClean, and the filtered apparent-sky sky models restored
    with a Gaussian beam one pixel wide (so each pixel holds about the flux of
    the components in it). Each non-zero pixel is moved to the nearest mosaic
    pixel, without interpolation, so the flux is kept, except in the strip a
    few pixels wide where neighbouring sectors overlap: there the images are
    averaged, which halves the flux. Pixels outside all sectors are blank.

With a single sector, the model images are those of the sector, as described
for ``sparse_fits``. In a cycle that also images Stokes Q, U and V (the final
cycle when :term:`make_quv_images` is ``True``), WSClean makes no sky models,
so all model images are made as with ``sparse_fits`` and no filtered model
image is made.
