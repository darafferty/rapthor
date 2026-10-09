.. _changelog:

Changelog
=========

Version 2.2rc1 (2026/10/09)
------------------------------

This release candidate for version 2.2 includes the following improvements:

    - Calibration strategies are now more configurable, including support for
      user-defined calibration strategy options.
    - Spectral cubes can now be generated for selected Stokes parameters.
      Deconvolution can optionally be disabled for full-Stokes imaging.
    - Clean masks can be reused between imaging runs, and images of filtered
      calibration sky models can be saved.
    - Imaging now supports frequency-dependent baseline-dependent averaging
      (BDA), with limits to prevent excessive averaging, and BDA can be used
      together with smearing corrections.
    - Astrometric corrections are now applied automatically when appropriate.
    - Rapthor can generate residual visibilities and save model data during
      imaging. Visibility averaging for imaging can optionally be disabled.
    - The final processing cycle can optionally be repeated, and the allowed
      pointing separation for input observations is now configurable.
    - Sky-model downloads and flux-scale normalization can be prepared before
      workflow execution. Reference sky models can be supplied for flux
      normalization, and internet access can be explicitly controlled.
      Local comparison sky models can also be supplied for photometry and
      astrometry diagnostics, and existing flux-normalization solutions can
      be supplied for imaging.
    - Pre-flight and pre-cycle memory checks and logging help identify
      out-of-memory risks before calibration jobs are started.
    - Workflow error handling, logging, restart behavior, and handling of
      temporary and output files have been improved.
    - Prediction and calibration workflows have received performance
      improvements, including better reuse of prediction ordering and parallel
      gridding support.
      The cluster option ``parallel_gridding_threads`` has been replaced by
      ``parallel_gridding_tasks``; custom parsets should use the new name.
      Experimental shared facet reads and writes can also be enabled.
    - WSClean-based prediction can now be used to generate model data for
      calibration.
    - Image workflow input generation now correctly provides parallel-gridding
      task counts for every imaging sector.
    - Observation chunking now balances full-data chunks across compute nodes
      while respecting minimum chunk durations, avoiding single-time-slot
      chunks that caused final-cycle imaging failures.
    - Image workflow outputs now correctly allow missing optional sky-model
      images, fixing CWL validation with newer cwltool versions.
    - Calibration workflows now use consistent names for DP3's BDA averaging
      step and its parameters.
    - The default container images and dependencies have been updated,
      including support for Ubuntu 24.04 and NumPy 2. The C++ dependencies are
      now pinned to fixed commit revisions. The most relevant changes in the
      external components used by Rapthor are:

        - WSClean now supports frequency-direction BDA, custom polarization
          selection in faceting, improved IDG facet imaging, and more efficient
          MPI facet I/O. It also includes improvements to smearing corrections,
          model rendering, and the EveryBeam interface.
        - DP3 adds further DDECal solver options, support for BDA model data,
          improved fast prediction, and fixes for BDA processing and calibration
          output. It also supports newer EveryBeam and NumPy versions.
        - IDG-Cal includes new solver and lower-memory options, performance
          improvements, and improved GPU reliability and resource handling.
        - Casacore and python-casacore include newer measures-data handling,
          storage-manager and metadata-compression improvements, and fixes for
          newer compilers and Python/NumPy environments.
        - AOFlagger improves handling of large data sets and spectral
          concatenation, and adds support for additional filterbank formats.
        - SageCal/libdirac is pinned to a revision compatible with Rapthor's
          GLib-free build.
        - The Python dependency stack now supports NumPy 2 and uses LSMTool
          1.9.0 or newer. The temporary cwltool version pin has been removed
          following the CWL output fixes.
        - Development and test dependencies are now defined in dedicated
          dependency groups, including formatting, parallel testing, coverage,
          and socket-isolation tooling.
    - Documentation and example strategy files have been updated to match
      current pipeline behavior and configuration options.
    - Many more improvements and bug fixes. See the git log for details.


Version 2.1 (2025/12/04)
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
      EveryBeam, WSClean, etc. For more details, please refer to their
      respective changelogs.
    - For WSClean, the most relevant changes are:
        - use of multi-frequency interface;
        - sub-pixel rendering;
        - smearing corrections;
        - better I/O in MPI mode thanks to shared facet reads/writes;
        - new options to tweak iteration strategy that lower the number of
          required major iterations;
        - support for time-BDA in facetting mode;
        - and lots of generic code improvements.
    - For DP3, the most relevant changes are:
        - meta data compression;
        - specify initial solutions in DDECal;
        - allow per-direction smoothness values as well as per-antenna
          smoothness and time-integration settings in DDECal;
        - better support of BDA data.
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
