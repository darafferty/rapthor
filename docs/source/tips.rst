.. _tips:

Tips for running Rapthor
========================

Processing a subset of the data
    To speed up processing, it is recommended that only a small fraction of the full
    dataset be used for self calibration. Rapthor will internally perform self calibration
    on 20% of the full data by default, but this number can be set with the
    :term:`selfcal_data_fraction` parameter. Once self calibration converges, Rapthor will
    by default perform a final cycle using 100% of the input data (the final fraction can
    be set using the :term:`final_data_fraction` parameter).

Number of directions / calibration patches
    Increasing the number of directions (also referred to as calibration patches and
    imaging facets, if faceting is used) will generally result in better
    direction-dependent corrections. However, more directions implies fainter calibration
    sources as well as longer runtimes, especially during calibration. Therefore, the
    default strategy slowly increases the number of directions with each self calibration
    cycle, as the model of the field improves and fainter sources can be used for
    calibration. Most fields work well with a maximum of 50 directions, but fields with
    many bright sources may require more and those with a lack of bright sources may
    require fewer.

Problematic fields
    Fields that lie at low declinations or that have very extended or very bright sources
    might pose problems for self calibration. For example, it is recommended that fields
    with very bright sources (> 20 Jy) use a processing strategy that starts with at least
    three rounds of phase-only calibration before moving to amplitude calibration (cf. the
    default strategy, which uses two rounds of phase-only calibration). The information
    here will be updated as further testing on a variety of field is done and our
    understanding of the sub-optimal cases improves.

Bright outlier sources
    The presence of very bright outlier sources (sources that lie outside of imaged
    regions) can cause strong artifacts across the field that cannot be corrected during
    self calibration. Possible solutions to this problem are to increase the image regions
    to include the outliers (e.g., with the image grid parameters
    :term:`grid_width_ra_deg` and :term:`grid_width_dec_deg`) or to place small imaging
    sectors on each outlier (by specifying the sectors using the sector list parameters
    such as :term:`sector_center_ra_list`). With either of these options, the outliers are
    imaged along with the main field and hence their models are updated each self
    calibration cycle.

Creating a dataset for further self calibration
    Rapthor can be used to create datasets that can be used for further self calibration
    outside of Rapthor, for example for targets of interest. These datasets can have
    non-target sources peeled and calibration solutions preapplied. The optimal way to
    generate such datasets with Rapthor is to run a standard reduction to generate a
    solution table (H5parm file) and sky model for the full field and to input these to a
    new Rapthor run using the "image" strategy, in which only peeling and imaging are
    done. The following parameters should be set:

        * Set the :term:`input_skymodel` and :term:`input_h5parm` parameters in the parset
          to the output of the full-field reduction. If a full-Jones solve was done,
          :term:`input_fulljones_h5parm` can also be set. The time and frequency coverage
          of the solution tables must be large enough to cover the duration and bandwidth
          of the input dataset. The easiest way to ensure this requirement is met is to
          use the solutions from a solve over the full dataset (i.e., those from the final
          cycle of a run with :term:`final_data_fraction` = 1.0)

          .. note::

              The sky model from a full-field reduction will contain only those sources
              that lie in the regions imaged during that reduction. If it is important to
              peel sources outside of these regions (e.g., there is a very bright source
              that lies outside the field, but near enough to cause problems if not
              peeled), then you need to add these sources to the input sky model before
              running Rapthor in this mode.

        * Set the :term:`regroup_input_skymodel` parameter to ``False`` to preserve the
          calibration patches that match the directions in the solutions file.

        * Set the :term:`strategy` parameter to "image" (or provide an equivalent custom
          strategy file).

        * Set the :term:`save_visibilities` parameter to ``True`` to save the MS files
          used for the imaging. The output MS files will be located in
          ``dir_working/visibilities``. Furthermore, :term:`dde_method` should be set to
          "single" to apply the solutions in single direction (the direction closest to
          the sector center). Other values of :term:`dde_method` will not apply the
          solutions to the visibilities, since the solutions in those cases are applied by
          WSClean during imaging.

        * Define the imaging sectors to cover the targets of interest. Multiple sectors can
          be used, and a set of calibrated visibilities will be generated for each sector.
