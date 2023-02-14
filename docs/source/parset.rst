.. _rapthor_parset:

The Rapthor parset
==================

Before Rapthor can be run, a parset describing the reduction must be made. The
parset is a simple text file defining the parameters of a run in a number of
sections. For example, a minimal parset for a basic reduction on a single
machine could look like the following (see :ref:`tips` for tips on setting up an
optimal parset):

.. code-block:: none

    [global]
    dir_working = /path/to/rapthor/working/dir
    input_ms = /path/to/input/dir/input.ms


The available options are described below under their respective sections.


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
        originals in any way. If multiple measurement sets are provided, they
        should be split in time. This is (currently) inconsistent with how `LINC
        <https://linc.readthedocs.io/>`_ outputs the measurement sets, which are
        split in frequency. Processing LINC outputs requires concatenating the
        measurement sets, see :doc:`preparation`.

    download_initial_skymodel
        Download the initial sky model automatically instead of using a user-provided one
        (default is ``True``). This option is ignored if a file is specified with the
        :term:`input_skymodel` option.

    download_initial_skymodel_radius
        The radius in degrees out to which a sky model should be downloaded (default is 5.0).

    download_initial_skymodel_server
        Place to download the initial sky model from (default is ``TGSS``). This can either
        be ``TGSS`` to use the TFIR GMRT Sky Survey or ``GSM`` to use the Global Sky
        Model.

    download_overwrite_skymodel
        Overwrite any existing sky model with a downloaded one (default is ``False``).

    input_skymodel
        Full path to the input sky model file, with true-sky fluxes (required if automatic
        download is disabled). If you also have a sky model with apparent flux densities,
        specify it with the :term:`apparent_skymodel` option.

	See :doc:`preparation` for more info on preparing the sky model.

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

    selfcal_data_fraction
        Fraction of data to use (default = 0.2). If less than one, the input
        data are divided by time into chunks that sum to the requested fraction,
        spaced out evenly over the full time range. Using a low value (0.2 or so)
        is strongly recommended for typical 8-hour, full-bandwidth observations.

    final_data_fraction
        A final data fraction can be specified (default = ``selfcal_data_fraction``)
        such that a final processing pass (i.e., after selfcal finishes) is
        done with a different fraction.

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

    llssolver
        The linear least-squares solver to use (one of "qr", "svd", or "lsmr";
        default = ``qr``)

    maxiter
        Maximum number of iterations to perform during calibration (default = 150).

    propagatesolutions
        Propagate solutions to next time slot as initial guess (default = ``True``)?

    solveralgorithm
        The algorithm used for solving (one of "directionsolve", "directioniterative",
        "lbfgs", or "hybrid"; default = ``hybrid``)? When using "lbfgs", the :term:`stepsize`
        should be set to a small value like 0.001.

    onebeamperpatch
        Calculate the beam correction once per calibration patch (default =
        ``False``)? If ``False``, the beam correction is calculated separately
        for each source in the patch. Setting this to ``True`` can speed up
        calibration and prediction, but can also reduce the quality when the
        patches are large.

    parallelbaselines
        Parallelize model calculation over baselines, instead of parallelizing over directions (default = ``False``).

    stepsize
        Size of steps used during calibration (default = 0.02). When using
        ``solveralgorithm = lbfgs``, the stepsize should be set to a small value like 0.001.

    tolerance
        Tolerance used to check convergence during calibration (default = 1e-3).

    solve_min_uv_lambda
        Minimum uv distance in lambda used during calibration (default = 350).

    fast_timestep_sec
        Time step used during fast phase calibration, in seconds (default = 8).

    fast_freqstep_hz
        Frequency step used during fast phase calibration, in Hz (default = 1e6).

    fast_smoothnessconstraint
        Smoothness constraint bandwidth used during fast phase calibration, in
        Hz (default = 3e6).

    fast_smoothnessreffrequency
        Smoothness constraint reference frequency used during fast phase calibration, in
        Hz (default = midpoint of frequency coverage).

    fast_smoothnessrefdistance
        Smoothness constraint reference distance used during fast phase calibration, in
        m (default = 0).

    slow_timestep_joint_sec
        Time step used during the first slow gain calibration, where a joint
        solution is found for all stations, in seconds (default = 0). Set to 0
        to disable this part of the slow-gain calibration.

    slow_timestep_separate_sec
        Time step used during the second slow gain calibration, where separate
        solutions are found for each station, in seconds (default = 600).

    slow_freqstep_hz
        Frequency step used during slow amplitude calibration, in Hz (default = 1e6).

    slow_smoothnessconstraint_joint
        Smoothness constraint bandwidth used during the first slow gain calibration,
        where a joint solution is found for all stations, in Hz (default = 3e6).

    slow_smoothnessconstraint_separate
        Smoothness constraint bandwidth used during the second slow gain calibration,
        where separate solutions are found for each station, in Hz (default = 3e6).

    use_idg_predict
       Use IDG for predict during calibration (default = ``False``)?

    solverlbfgs_dof
       Degrees of freedom for LBFGS solver (only used when solveralgorithm = "lbfgs"; default 200.0).

    solverlbfgs_minibatches
       Number of minibatches for LBFGS solver (only used when solveralgorithm = "lbfgs"; default 1).

    solverlbfgs_iter
       Number of iterations per minibat in LBFGS solver (only used when solveralgorithm = "lbfgs"; default 4).

.. _parset_imaging_options:

``[imaging]``
-----------------

.. glossary::

    cellsize_arcsec
        Pixel size in arcsec (default = 1.25).

    robust
        Briggs robust parameter (default = -0.5).

    min_uv_lambda
        Minimum uv distance in lambda to use in imaging (default = 0).

    max_uv_lambda
        Maximum uv distance in lambda to use in imaging (default = 0).

    taper_arcsec
        Taper to apply when imaging, in arcsec (default = 0).

    do_multiscale_clean
        Use multiscale cleaning (default = ``True``)?

    dde_method
        Method to use to correct for direction-dependent effects during imaging: "none",
        "facets", or "screens" (default = ``facets``). If "none", the solutions closest to the image centers
        will be used. If "facets", Voronoi faceting is used. If "screens", smooth 2-D
        screens are used.

    screen_type
        Type of screen to use (default = ``tessellated``), if ``dde_method = screens``:
        "tessellated" (simple, smoothed Voronoi tessellated screens) or
        "kl" (Karhunen-Lo`eve screens).

    idg_mode
        IDG (image domain gridder) mode to use in WSClean (default = "hybrid").
        The mode can be "cpu" or "hybrid".

    mem_fraction
        Fraction of the total memory (per node) to use for WSClean jobs (default = 0.9).

    use_mpi
        Use MPI to distribute WSClean jobs over multiple nodes (default =
        ``False``)? If ``True`` and more than one node can be allocated to each
        WSClean job (i.e., max_nodes / num_images >= 2), then distributed
        imaging will be used (only available if ``batch_system = slurm`` and
        ``dde_method = screens``).

        .. note::

            If MPI is activated, :term:`dir_local` (under the
            :ref:`parset_cluster_options` section below) must not be set unless
            it is on a shared filesystem.

    reweight
        Reweight the visibility data before imaging (default = ``False``). If
        ``True``, data with high residuals (compared to the predicted model
        visibilities) are down-weighted. This feature is experimental and
        should be used with caution.

    grid_width_ra_deg
        Size of area to image when using a grid (default = mean FWHM of the
        primary beam).

    grid_width_dec_deg
        Size of area to image when using a grid (default = mean FWHM of the
        primary beam).

    grid_center_ra
        Center of area to image when using a grid (default = phase center).

    grid_center_dec
        Center of area to image when using a grid (default = phase center).

    grid_nsectors_ra
        Number of sectors along the RA axis (default = 0). The number of sectors
        in Dec will be determined automatically to ensure the whole area
        specified with :term:`grid_center_ra`, :term:`grid_center_dec`,
        :term:`grid_width_ra_deg`, and :term:`grid_width_dec_deg` is imaged. Set
        ``grid_nsectors_ra = 0`` to force a single sector for the full area.
        Multiple sectors are useful for parallelizing the imaging over multiple
        nodes of a cluster or for computers with limited memory.

    sector_center_ra_list
        List of image centers (default = ``[]``). Instead of a grid, imaging sectors
        can be defined individually by specifying their centers and widths.

    sector_center_dec_list
        List of image centers (default = ``[]``).

    sector_width_ra_deg_list
        List of image widths, in degrees (default = ``[]``).

    sector_width_dec_deg_list
        List of image  widths, in degrees (default = ``[]``).

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
        Cluster batch system (default = "single_machine"). Use "single_machine" when
        running on a single machine and "slurm" to use multiple nodes of a SLURM-based
        cluster.

    max_nodes
        When batch_system = "slurm", the maximum number of nodes of the cluster to
        use at once (default = 12).

    cpus_per_task
        When batch_system = "slurm", the number of processors per task to
        request (default = 0 = all). By setting this value to the number of processors
        per node, one can ensure that each task gets the entire node to itself,
        which is the recommended way of running Rapthor.

    mem_per_node_gb
        When batch_system = "slurm", the amount of memory per node in GB to request
        (default = 0 = all).

    max_cores
        Maximum number of cores per task to use on each node (default = 0 =
        all).

    max_threads
        Maximum number of threads per task to use on each node (default = 0 =
        all).

    deconvolution_threads
        Number of threads to use by WSClean during deconvolution (default = 0 = 2/5
        of ``max_threads``).

    parallel_gridding_threads
        Number of threads to use by WSClean during parallel gridding (default = 0 = 2/5
        of ``max_threads``).

    dir_local
        Full path to a local disk on the nodes for IO-intensive processing (default =
        not used). The path must exist on all nodes (but does not have to be on a
        shared filesystem). This parameter is useful if you have a fast local disk
        (e.g., an SSD) that is not the one used for :term:`dir_working`. If this parameter is
        not set, IO-intensive processing (e.g., WSClean) will use a default path in
        :term:`dir_working` instead.

        .. note::

            This parameter should not be set in the following situations:

            - when :term:`batch_system` = ``single_machine`` and multiple imaging sectors are
              used (as each sector will overwrite files from the other sectors)

            - when :term:`use_mpi` = ``True`` under the :ref:`parset_imaging_options`
              section and ``dir_local`` is not on a shared filesystem.

    cwl_runner
        CWL runner to use. Currently supported runners are: cwltool and toil (default).
        Toil is the recommended runner, since it provides much more fine-grained control
        over the execution of a workflow. For example, Toil can use Slurm to automatically
        distribute workflow steps over different compute nodes, whereas CWLTool can only
        execute workflows on a single node. With CWLTool you also run the risk of
        overloading your machine when too many jobs are run in parallel. For debugging
        purposes CWLTool outshines Toil, because its logs are easier to understand.

    debug_workflow
        Debug workflow related issues. Enabling this will require significantly more
        disk space. The working directory will never be cleaned up, stdout and stderr
        will not be redirectied, and log level of the CWL runner will be set to DEBUG.
        Use this option with care!
