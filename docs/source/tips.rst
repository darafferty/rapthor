.. _tips:

Tips for running Rapthor
========================

Processing a subset of the data
    To speed up processing, it is recommended that only a small fraction of the full dataset be used for self calibration. Rapthor will internally perform self calibration on 20% of the full data by default, but this number can be set with the :term:`selfcal_data_fraction` parameter. Once self calibration converges, Rapthor will by default perform a final cycle using 100% of the input data (the final fraction can be set using the :term:`final_data_fraction` parameter).

Number of directions / calibration patches
    Increasing the number of directions (also referred to as calibration patches and imaging facets, if faceting is used) will generally result in better direction-dependent corrections. However, more directions implies fainter calibration sources as well as longer runtimes, especially during calibration. Therefore, the default strategy slowly increases the number of directions with each self calibration cycle, as the model of the field improves and fainter sources can be used for calibration. Most fields work well with a maximum of 50 directions, but fields with many bright sources may require more and those with a lack of bright sources may require fewer.

Problematic fields
    Fields that lie at low declinations or that have very extended or very bright sources might pose problems for self calibration. For example, it is recommended that fields with very bright sources (> 20 Jy) use a processing strategy that starts with at least three rounds of phase-only calibration before moving to amplitude calibration (cf. the default strategy, which uses two rounds of phase-only calibration). The information here will be updated as further testing on a variety of field is done and our understanding of the sub-optimal cases improves.

Bright outlier sources
    The presence of very bright outlier sources (sources that lie outside of imaged regions) can cause strong artifacts across the field that cannot be corrected during self calibration. Possible solutions to this problem are to increase the image regions to include the outliers (e.g., with the image grid parameters :term:`grid_width_ra_deg` and :term:`grid_width_dec_deg`) or to place small imaging sectors on each outlier (by specifying the sectors using the sector list parameters such as :term:`sector_center_ra_list`). With either of these options, the outliers are imaged along with the main field and hence their models are updated each self calibration cycle.
