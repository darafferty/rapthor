.. _operations:

Operations
==========

Most of the processing performed by Rapthor is done in "operations," which are sets of steps that are grouped together into CWL workflows. The available operations and the primary data products of each are described in detail below.


.. _calibrate:

Calibrate
---------

This operation calibrates the data using the current sky model. The exact steps done during calibration depend on the strategy, but essentially there are two main parts: a phase-only (scalar) solve on short timescales (the "fast-phase" solve, which corrects for ionospheric errors) and a phase and amplitude (diagonal) solve on long time scales (the "slow-gain" solve, which corrects for beam errors). The slow-gain solve is divided into two parts: an optional amplitude-only solve that constrains all stations to have the same solutions and a phase plus amplitude solve (without the station constraint and usually with a longer solution interval). This calibration strategy is based on the LBA strategy of the `LiLF pipeline <https://github.com/revoltek/LiLF>`_, with the idea that the same strategy can be used for both HBA and LBA. This is similar to the way the calibrator pipeline works in `LINC <https://linc.readthedocs.io/>`_, which has been inspired by LiLF. Lastly, processing of the resulting solutions is done, including smoothing and renormalization.

For calibration, Rapthor searches for bright, compact sources (or groups of sources) throughout the field to use as calibrator sources. A target (apparent) flux density is used to ensure that the calibrators are sufficiently bright (set by :term:`target_flux` in the processing strategy). Rapthor then tessellates the full sky model, using the calibrators as the facet centers. This method ensures that each calibration patch (or facet) has a bright calibrator source in it. Despite this designation of calibrators for the tesselation, all sources are used in the calibration (not just the bright sources).

When multiple nodes are available, this task is distributed.

Primary products:
    * In ``skymodels/calibrate_X``, where ``X`` is the cycle number:
        * ``calibration_skymodel.txt`` - the sky model used for calibration, grouped into calibration patches (one per facet/direction). If a sky model was supplied by the user, this model will be identical (but potentially with a different grouping of the sources). If the sky model results from the previous cycle of self calibration, this model will be the sum of models from the imaging sectors (see :ref:`image` for details).
    * In ``solutions/calibrate_X``, where ``X`` is the cycle number:
        * ``field-solutions.h5`` - the calibration solution table containing both fast-phase and slow-gain solutions.
    * In ``plots/calibrate_X``, where ``X`` is the cycle number:
        * ``*.png`` files - plots of the calibration solutions. Plots are typically made with one file per direction (calibration patch), per solution type (amplitude, phase, or scalar phase). For example, the file ``scalarphase_dir[Patch_127].png`` contains the scalar phase solutions (from the "fast-phase" solve) for patch 127. If the "slow-gain" solve was done, additional files should be present with the names ``phase_dir[Patch_127]_polXX.png`` and ``amplitude_dir[Patch_127]_polXX.png`` (and similarly for the YY polarization). If the optional amplitude-only slow-gain solve was done, the solutions are combined with the phase plus amplitude solve before plotting.

If a full-Jones solve was done for a given cycle, then a number of further products are created:
    * In ``solutions/calibrate_di_X``, where ``X`` is the cycle number:
        * ``fulljones-solutions.h5`` - the calibration solution table containing full-Jones gain solutions.
    * In ``plots/calibrate_di_X``, where ``X`` is the cycle number:
        * ``*.png`` files - plots of the full-Jones calibration solutions. Since the full-Jones solve is a direction-independent one, there will be two sets of four plots: the four amplitude plots (for the XX, XY, YX, and YY polarizations) and the four phase plots (again for each polarization).

.. _predict:

Predict
-------

This operation predicts visibilities for subtraction. Sources that lie outside of imaged regions are subtracted, as are bright sources inside imaged regions (if desired). This operation will not be run if no prediction or subtraction needs to be done.

When multiple nodes are available, this task is distributed.

Primary products:
    * In ``skymodels/predict_X``, where ``X`` is the cycle number:
        * ``outlier_*_predict_skymodel.txt`` - sky models used for outlier subtraction
        * ``bright_source_*_predict_skymodel.txt`` - sky models used for bright-source subtraction
        * ``sector_*_predict_skymodel.txt`` - sky models used when multiple imaging sectors are used
    * In ``pipelines/predict_X``, where ``X`` is the cycle number:
        * Temporary measurement sets used for subsequent operations.

If LBA data are processed, then a predict operation is done before the calibration operation to subtract off non-calibrator sources, with the following output:
    * In ``pipelines/predict_nc_X``, where ``X`` is the cycle number:
        * Temporary measurement sets used for the calibration.

If a full-Jones solve was done for a given cycle, then a number of further products are created:
    * In ``skymodels/predict_X``, where ``X`` is the cycle number:
        * ``predict_*_predict_skymodel.txt`` - sky models used for the prediction needed for the full-Jones solve
    * In ``pipelines/predict_di_X``, where ``X`` is the cycle number:
        * Temporary measurement sets used for the full-Jones solve.


.. _image:

Image (+ mosaic)
----------------

This operation images the data. If multiple imaging sectors are used, a mosaic operation is also run to mosaic the sector images together into a single image. If bright sources were subtracted in the preceding :ref:`predict` operation, they are restored during this operation once imaging has finished.

Diagnostics for each image are written to the main log (``dir_working/logs/rapthor.log``). The diagnostics can be useful for judging how self calibration is proceeding. They include the following:

    * The minimum and theoretical RMS noise. The minimum noise is derived from 2-D RMS maps generated by PyBDSF using the non-primary beam corrected image. The theoretical noise is calculated following `SKA Memo 113 <https://arxiv.org/abs/1308.4267>`_ and the `LOFAR Image Noise Calculator <https://support.astron.nl/ImageNoiseCalculator/sens.php>`_. The calculation takes into account the amount of flagged data but does not include the effects of elevation.
    * The median RMS noise. The median noise is derived from 2-D RMS maps generated by PyBDSF using the non-primary beam corrected image. This median noise, along with the dynamic range (see below) is used to determine whether selfcal has converged (using the :term:`convergence_ratio` and :term:`divergence_ratio` defined by the processing strategy).
    * The dynamic range, calculated as the maximum value in the image divided by the minimum RMS noise, using the non-primary beam corrected image. This quantity gives an estimate of how well focused the brightest source in the image is and is used, along with the median noise (see above) and the number of sources found in the image (see below) to determine whether selfcal has converged.
    * The number of sources found by PyBDSF. As with the noise and dynamic range estimates, the number of sources is used to determine whether selfcal has converged.
    * The reference (central) frequency of the image.
    * The restoring beam size and position angle.
    * The fraction of unflagged data.
    * Estimates of the LOFAR-to-TGSS and LOFAR-to-LoTSS flux ratios (calculated as the mean of the measured LOFAR flux densities divided by the TGSS/LoTSS flux densities, after sigma clipping). This ratio gives an indication of the accuracy of the overall flux scale of the image. When the reference frequency of the LOFAR image differs from that of the reference catalogs, the ratio is corrected assuming a mean source spectral index of -0.7.

        .. note::

            This ratio should be considered as a rough estimate only. A careful analysis of the overall flux calibration of the field should be done outside of Rapthor.

        .. note::

            If the flux ratios from both the TGSS and LoTSS surveys are unavailable (due to, e.g., lack of coverage or too few source matches), an attempt is made to estimate the ratio using the NVSS survey (at 1.4 GHz). Note, however, that this ratio is especially uncertain due to the large extrapolation required to adjust the LOFAR and NVSS flux densities to a common frequency.

    * Estimates of the LOFAR-to-Pan-STARRS RA and Dec offsets (calculated as the mean of the LOFAR values minus the Pan-STARRS values, after sigma clipping). These offsets give an indication of the accuracy of the astrometry.

Primary products:
    * In ``images/image_X``, where ``X`` is the cycle number:
        * ``field-MFS-image.fits`` - the Stokes I image, uncorrected for the primary beam attenuation (i.e., the apparent-sky, "flat-noise" image)
        * ``field-MFS-image-pb.fits`` - the Stokes I image, corrected for the primary beam attenuation (i.e., the true-sky image)
        * ``field-MFS-residual.fits`` - the Stokes I residual image
        * ``field-MFS-model.fits`` - the Stokes I model image

        .. note::

            If Stokes QUV images are also made (see :term:`make_quv_images`), then there will be a set of output images for each Stokes parameter, The image names will include the Stokes parameter. E.g., the apparent-sky, "flat-noise" images will be named ``field-MFS-I-image.fits``, ``field-MFS-Q-image.fits``, etc.

        .. note::

            If an initial sky model was generated from the input data (see :term:`generate_initial_skymodel`), then there will be a set of output images in ``images/initial_image``. These images are generated directly from the input data (with no additional calibration) and are used to derive the initial sky model.

    * In ``plots/image_X``, where ``X`` is the cycle number:

        .. note::

            In the following, the "flux ratio" is calculated (per source) as the Rapthor-derived LOFAR flux density divided by the reference catalog flux density, where the reference catalog is one of TGSS, NVSS, or LoTSS. The "positional offsets" are calculated (per source) as the Rapthor-derived RA or Dec value minus the Pan-STARRS value.

        * ``sector_Y.flux_ratio_vs_distance_TGSS/NVSS/LoTSS.pdf``, where ``Y`` is the image sector number - plots of the flux ratio vs. distance from the phase center.
        * ``sector_Y.flux_ratio_vs_flux_TGSS/NVSS/LoTSS.pdf``, where ``Y`` is the image sector number - plots of the flux ratio vs. Rapthor-derived LOFAR flux density.
        * ``sector_Y.positional_offsets_sky.pdf``, where ``Y`` is the image sector number -  scatter plot of the RA and Dec positional offsets.
        * ``sector_Y.astrometry_offsets.pdf``, where ``Y`` is the image sector number -  plot showing the mean RA and Dec positional offsets for each facet covered by the sector. Arrows indicate the magnitude and direction of the mean offsets.

        .. note::

            If an initial sky model was generated from the input data (see :term:`generate_initial_skymodel`), then there will be a set of plots in ``plots/initial_image`` from the astrometry and photometry analysis of the initial image used to derive the initial sky model.

    * In ``skymodels/image_X``, where ``X`` is the cycle number:
        * ``bright_source_skymodel.txt`` - sky model used to restore bright sources after imaging (present only if bright sources were subtracted in the preceding predict operation).
        * ``sector_Y.source_catalog.fits``, where ``Y`` is the image sector number - the source catalog (generated by PyBDSF) for the sector.
        * ``sector_Y.true_sky.txt``, where ``Y`` is the image sector number - the sky model (generated by WSClean) for the sector, with true-sky flux densities.
        * ``sector_Y.apparent_sky.txt``, where ``Y`` is the image sector number - the sky model for the sector, with apparent-sky flux densities, generated from the true-sky one by attenuating it with the LOFAR primary beam.

        .. note::

            If Stokes QUV images are also made (see :term:`make_quv_images`), then WSClean does not generate the output sector sky models.

        .. note::

            If an initial sky model was generated from the input data (see :term:`generate_initial_skymodel`), then there will be two sky model files in ``skymodels/initial_image`` (an apparent-sky model and a true-sky model). These models are used as input for the first cycle of calibration.

    * In ``visibilities/image_X/sector_Y``, where ``X`` is the cycle number and ``Y`` is the image sector number (only if the :term:`save_visibilities` parameter is set to ``True``):
        * ``*.ms`` - measurement sets used as input to WSClean for imaging. Depending on
          the value of :term:`dde_method`, some or all of the calibration solutions may be
          preapplied: a value of "single" will preapply all solutions, whereas a value of
          "full" will preapply only the full-Jones solutions (if
          available), since the direction-dependent solutions in those cases are applied
          by WSClean itself. These MS files can be useful for further imaging or self
          calibration outside of Rapthor.
