.. _rapthor_parset:

The Rapthor parset
==================

Before Rapthor can be run, a parset describing the reduction must be made. The
parset is a simple text file defining the parameters of a run in a number of
sections. For example, a typical parset for a basic reduction on a single
machine could look like the following (see :ref:`tips` for tips on setting up an
optimal parset):

.. code-block:: none

    [global]
    dir_working = /path/to/rapthor/working/dir
    input_ms = /path/to/input/dir/input.ms
    input_skymodel = /path/to/input/dir/input.sky


All the available options are described below under their respective sections.


.. _parset_global_options:

``[global]``
------------

.. glossary::

    dir_working
        Full path to working dir where rapthor will run (required). All output
        will be placed in this directory. E.g., ``dir_working = /data/rapthor``.

    input_ms
        Full path to directory containing the input MS files (required).
        Wildcards can be used (e.g., ``input_ms = /path/to/data/*.ms``). Note
        that Rapthor works on a copy of these files and does not modify the
        originals in any way.

    input_skymodel
        Full path to the input sky model file, with true-sky fluxes (required).
        If you also have a sky model with apparent flux densities, specify it
        with the :term:`apparent_skymodel` option.

    apparent_skymodel
        Full path to the input sky model file, with apparent-sky fluxes
        (optional). Note that the source names must be identical to those in
        :term:`input_skymodel`.

    regroup_input_skymodel
        Regroup input skymodel as needed to meet target flux (default =
        ``True``). If False, the existing patches are used for the calibration.

    strategy
        Name of processing strategy to use (default = ``selfcal``). A custom
        strategy can be used by giving instead the full path to the strategy
        file. See :ref:`rapthor_strategy` for details on making a custom
        strategy file.

    data_fraction
        Fraction of data to use (default = 1.0). If less than one, the input
        data are divided by time into chunks that sum to the requested fraction,
        spaced out evenly over the full time range.

    flag_abstime
        Range of times to flag (default = no flagging). The syntax is that of
        the preflagger ``abstime`` parameter (see the DPPP documentation on the
        LOFAR wiki for details of the syntax). E.g.,
        ``[12-Mar-2010/11:31:00.0..12-Mar-2010/11:50:00.0]``.

    flag_baseline
        Range of baselines to flag (default = no flagging). The syntax is that
        of the preflagger ``baseline`` parameter (see the DPPP documentation for
        details of the syntax). E.g., ``flag_baseline = [CS013HBA*]``.

    flag_freqrange
        Range of frequencies to flag (default = no flagging). The syntax is that
        of the preflagger ``freqrange`` parameter (see the DPPP documentation for
        details of the syntax). E.g., ``flag_freqrange = [125.2..126.4MHz]``.

    flag_expr
        Expression that defines how the above flagging ranges are combined to
        produce the final flags (default = all ranges are AND-ed). The syntax is
        that of the preflagger ``expr`` parameter (see the DPPP documentation on
        the LOFAR wiki for details of the syntax). E.g., ``flag_freqrange or
        flag_baseline``.


.. _parset_calibration_options:

``[calibration]``
-----------------

.. glossary::

    maxiter
        Maximum number of iterations to perform during calibration (default = 50).

    propagatesolutions
        Propagate solutions to next time slot as initial guess (default = ``True``)?

    stepsize
        Size of steps used during calibration (default = 0.02).

    tolerance
        Tolerance used to check convergence during calibration (default = 1e-3).

    solve_min_uv_lambda
        Minimum uv distance in lambda used during calibration (default = 80).

    fast_timestep_sec
        Time step used during fast phase calibration, in seconds (default = 8).

    fast_freqstep_hz
        Frequency step used during fast phase calibration, in MHz (default = 1e6).

    fast_smoothnessconstraint
        Smoothness constraint bandwidth used during fast phase calibration, in
        MHz (default = 6e6).

    slow_timestep_sec
        Time step used during slow amplitude calibration, in seconds (default = 600).

    slow_freqstep_hz
        Frequency step used during slow amplitude calibration, in MHz (default = 1e6).

    slow_smoothnessconstraint
        Smoothness constraint bandwidth used during slow amplitude calibration,
        in MHz (default = 3e6).

    use_idg_predict
       Use IDG for predict during calibration (default = ``False``)?


.. _parset_imaging_options:

``[imaging]``
-----------------

.. glossary::

    cellsize_arcsec
        Pixel size in arcsec (default = 1.5).

    robust
        Briggs robust parameter (default = -0.5).

    min_uv_lambda
        Minimum uv distance in lambda to use in imaging (default = 80).

    max_uv_lambda
        Maximum uv distance in lambda to use in imaging (default = 80).

    taper_arcsec
        Taper to apply when imaging, in arcsec (default = 0)

    multiscale_scales_pixel
        Scale sizes in pixels to use during multiscale clean (default = ``[0, 5, 10, 15]``)

    do_multiscale
        Use multiscale cleaning (default = auto)?

    use_screens
        Use screens during imaging (default = ``True``)? If ``False``, the
        solutions closest to the image centers will be used

    idg_mode
        IDG (image domain gridder) mode to use in WSClean (default = ``hybrid``).
        The mode can be cpu or hybrid

    use_mpi
        Use MPI to distribute WSClean jobs over multiple nodes (default =
        ``False``)? If ``True`` and more than one node can be allocated to each
        WSClean job (i.e., max_nodes / num_images >= 2), then distributed
        imaging will be used (only available if batch_system = slurm)

    reweight
        Reweight the visibility data before imaging (default = ``True``)

    grid_width_ra_deg
        Size of area to image when using a grid (default = mean FWHM of the
        primary beam)

    grid_width_dec_deg
        Size of area to image when using a grid (default = mean FWHM of the
        primary beam)

    grid_center_ra
        Center of area to image when using a grid (default = phase center)

    grid_center_dec
        Center of area to image when using a grid (default = phase center)

    grid_nsectors_ra
        Number of sectors along the RA axis (default = 0). The number of sectors
        in Dec will be determined automatically to ensure the whole area
        specified with :term:`grid_center_ra`, :term:`grid_center_dec`,
        :term:`grid_width_ra_deg`, and :term:`grid_width_dec_deg` is imaged. Set
        ``grid_nsectors_ra = 0`` to force a single sector for the full area.
        Multiple sectors are useful for parallelizing the imaging over multiple
        nodes of a cluster or for computers with limited memory

    sector_center_ra_list
        List of image centers (default = ``[]``). Instead of a grid, imaging sectors
        can be defined individually by specifying their centers and widths.

    sector_center_dec_list
        List of image centers (default = ``[]``).

    sector_width_ra_deg_list
        List of image widths, in degrees (default = ``[]``).

    sector_width_dec_deg_list
        List of image  widths, in degrees (default = ``[]``).

    sector_do_multiscale_list
        List of multiscale flags, one per sector (default = ``[]``). ``None``
        indicates that multiscale clean should be activated automatically if a
        large source is detected in the sector

    max_peak_smearing
        Max desired peak flux density reduction at center of the image edges due
        to bandwidth smearing (at the mean frequency) and time smearing (default
        = 0.15 = 15% reduction in peak flux). Higher values result in shorter
        run times but more smearing away from the image centers.


.. _parset_cluster_options:

``[cluster]``
-----------------

.. glossary::

    batch_system
        Cluster batch system (default = ``singleMachine``). Use ``batch_system =
        slurm`` to use a SLURM-based cluster.

    max_nodes
        For ``batch_system = slurm``, the maximum number of nodes of the cluster to
        use at once (via the ``--nodes`` option in ``sbatch``; default = 12).

    cpus_per_task
        For ``batch_system = slurm``, the number of processors per task to request
        (via the ``--ntasks-per-node`` option in ``sbatch``; default = 6). By
        setting this value to the number of processors per node, one can ensure
        that each task gets the entire node to itself, which is the recommended
        way of running Rapthor.

    max_cores
        Maximum number of cores per task to use on each node (default = 0 =
        all).

    max_threads
        Maximum number of threads per task to use on each node (default = 0 =
        all).

    dir_local
        Full path to a local disk on the nodes for IO-intensive processing (no
        default). The path must exist on all nodes. This parameter is useful if
        you have a fast local disk (e.g., an SSD) that is not the one used for
        :term:`dir_working`. If this parameter is not set, IO-intensive
        processing (e.g., WSClean) will use a default path in :term:`dir_working`
        instead.

        .. note::

            This parameter should not be set when :term:`batch_system` = ``singleMachine``
            and multiple imaging sectors are used, as each sector will overwrite files
            from the other sectors. In this case, it is best to leave ``dir_local`` unset.
