.. _changelog:

Changelog
=========

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

