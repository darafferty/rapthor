.. _changelog:

Changelog
=========

Version 2.1 (2025/12/xx)
------------------------

This minor release includes the following improvements:

    - Image quality has improved significantly. This is mainly due to the use
      of a more sophisticated calibration strategy.
    - Processing speed has improved further, with typical processing times
      reduced by a factor ~2 compared to v2.0.
    - Time and frequency smearing effects can now be corrected for during the
      prediction part of calibration and during imaging. It is disabled by
      default, as the smearing corrections are still experimental in WSClean,
      and need more testing.
    - The calibration operation can now use image-based prediction.
      Image-based prediction can be faster than the normal prediction,
      especially for large sky models. It is disabled by default, but can be
      useful in certain situations (e.g., when filtering of the calibration
      sky model is disabled).
    - Rapthor can now produce spectral image cubes for Stokes-I with a
      user-specified channel width.
    - IDGCal can now be used for calibration during the final cycle (note: this
      mode should be considered experimental).
    - Improvements in components used by Rapthor, like: AOFlagger, DP3,
      EveryBeam, WSClean, etc. Please refer to their respective changelogs for
      details.
    - Many more improvements and bug fixes. See the git log for details.


Version 2.0 (2025/04/11)
------------------------

This release includes improvements to:

    - Speed across all elements of the processing, with large gains to
      calibration and imaging. The speed improvements are partly due to changes
      to Rapthor and partly to changes to the underlying tools that Rapthor uses
      (e.g., DP3 and WSClean). Users can expect overall processing times to be
      ~ 5 times shorter than for v1.1.
    - Imaging quality and self-calibration stability (e.g., through new
      and tweaked calibration and imaging parameters).
    - Speed of solver convergence by propagating solutions from the
      previous cycle.
    - Determination of astrometry errors in the images (note:
      correction of these errors is planned for a future update).
    - Handling of scratch directories (both local and global) and
      cleanup of temporary files.
    - Self-calibration convergence checks.
    - Support for using LINC's output data products directly as input
      to Rapthor.
    - Chunking of the data (done when the specified data fraction is
      less than one).
    - Smoothing of calibration solutions.
    - Support for multi-epoch observations.
    - Diagnostics (e.g., supplementary images and job statistics).

 and the addition of:

    - Option to generate an initial calibration model directly from the
      input data.
    - Option to use direction-dependent solution intervals and
      direction-dependent smoothness constraints during calibration.
    - Option to do a direction-independent full-Jones solve (with separate
      corrections for all four polarizations) in each cycle.
    - Option to do full-polarization (IQUV) imaging in the final cycle.
    - Generation of overview plots for the field, showing the coverage of
      the calibration model and images.
    - Automatic correction of offsets to the global flux scale.
    - Option to generate calibrated visibilities for one or more
      directions (useful for further processing outside of Rapthor).
    - Option to specify the name of the input data column (previously only
      'DATA' was allowed).
    - Option to download a LoTSS sky model for the initial calibration.
    - Option to specify the facet layout used in calibration and imaging.
    - Support for baseline-dependent averaging during calibration.


Version 1.1 (2023/07/27)
------------------------

This minor release includes the following improvements:

    - Speed up in imaging for data fractions < 1, by first concatenating in time the multiple MS files. This avoids the large penalty incurred when each measurement set is gridded individually by WSClean.
    - SageCal can be used for speeding up the DP3 predict step in the calibration workflow. Note that the use of SageCal prediction is still considered experimental!
    - Improvements in the determination of facet regions for large images.
    - Several improvements in the documentation.
    - Several bug fixes.


Version 1.0 (2023/06/08)
------------------------

This release provides the following functionality:

    - Automated self calibration of HBA observations of "average" fields (i.e., those without very bright or extended sources).
    - Parallelization over multiple nodes of compute clusters (using Slurm).
    - Containerization via Docker or Singularity.

Known limitations, to be addressed in future releases, include the following:

    - Automated self calibration of low declination fields does not yet work well.
    - The use of screens should be considered experimental.
    - The use of GPUs is not yet supported except in imaging when using screens. Work is ongoing to add support for GPUs for prediction.
    - Processing times can be very long for large datasets. Considerable effort is being devoted to speeding up the slowest parts of calibration and imaging.
    - Only Stokes I imaging is currently done.

