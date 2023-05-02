.. _tips:

Tips for running Rapthor
========================

Processing a subset of the data
    To speed up processing, it is recommended that only a small fraction of the full dataset be used for self calibration. Rapthor will internally perform self calibration on 20% of the full data by default, but this number can be set with the :term:`selfcal_data_fraction` parameter. Once self calibration converges, Rapthor will by default perform a final cycle using 100% of the input data (the final fraction can be set using the :term:`final_data_fraction` parameter).

Problematic fields
    Fields with very extended or very bright sources might pose problems for self calibration. For example, it is recommended that fields with very bright sources (> 20 Jy) use a processing strategy that starts with at least three rounds of phase-only calibration before moving to amplitude calibration (cf. the default strategy, which uses two rounds of phase-only calibration). The information here will be updated as further testing on a variety of field is done and our understanding of the sub-optimal cases improves.
