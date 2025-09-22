.. _rapthor_parset:

The Rapthor parset
==================

Before Rapthor can be run, a parset describing the reduction must be made. The parset is a
simple text file defining the parameters of a run in a number of sections. For example, a
minimal parset for a basic reduction on a single machine could look like the following
(see :ref:`tips` for tips on setting up an optimal parset):

.. code-block:: none

    [global]
    dir_working = /path/to/rapthor/working/dir
    input_ms = /path/to/input/dir/input.ms


The available options are described below under their respective sections.

.. note::

    An example parset is available in the `settings directory
    <https://git.astron.nl/RD/rapthor/-/blob/master/rapthor/settings/defaults.parset>`_.
    In addition, example parsets for running on the SKAO cluster using either a single node,
    or multiple nodes, are availble in the `examples directory
    <https://git.astron.nl/RD/rapthor/-/tree/master/examples>`_ and an example parset 
    for SKA-low is available in `the settings directory 
    <https://git.astron.nl/RD/rapthor/-/blob/master/rapthor/settings/defaults_skalow.parset>`_.

.. _parset_global_options:

``[global]``
------------

.. glossary::

    dir_working
        Full path to working dir where rapthor will run (required). All output will be
        placed in this directory. E.g., ``dir_working = /data/rapthor``.

    input_ms
        Full path to the input MS files (required). Wildcards can be used (e.g.,
        ``input_ms = /path/to/data/*.ms``). The paths can also be given as a list (e.g.,
        ``input_ms = [/path/to/data1/*.ms, /path/to/data2/*.ms, /path/to/data3/obs3.ms]``).
        Note that Rapthor works on a copy of these files and does not modify the originals
        in any way.

        .. note::

            The MS files output by the `LINC
            <https://linc.readthedocs.io/>`_ pipeline can be directly used with Rapthor.
            See :doc:`preparation` for details.

    data_colname
        Data column to be read from the input MS files (default = ``DATA``).

    generate_initial_skymodel
        Generate an initial target sky model from the input data (default = ``True``).
        This option is ignored if a file is specified with the :term:`input_skymodel`
        option. When this option is activated, an image of the full field is made from the
        input data (without doing any calibration). The initial sky model is generated
        from the clean components as part of this imaging and will be located in
        the ``dir_working/skymodels/initial_image`` directory.

    generate_initial_skymodel_radius
        The radius out to which the sky model will be generated (default = None, which
        results in coverage out to a width of 2 * FWHM of the primary beam at the mean
        frequency and mean elevation of the observations).

    generate_initial_skymodel_data_fraction
        Fraction of data to use during the generation of the initial sky model (default =
        0.2). If less than one, the input data are divided by time into chunks that sum to
        the requested fraction, spaced out evenly over the full time range.

    download_initial_skymodel
        Download the initial sky model automatically instead of using a user-provided one
        (default = ``False``). This option is ignored if a file is specified with the
        :term:`input_skymodel` option or if generation of the initial model is activated
        with the :term:`generate_initial_skymodel` option. The downloaded sky model will
        be named ``dir_working/skymodels/initial_skymodel_{catalog}.txt``, where
        ``{catalog}`` is the name of the catalog server specified by the
        :term:`download_initial_skymodel_server` option.

    download_initial_skymodel_radius
        The radius in degrees out to which a sky model should be downloaded (default =
        5.0).

    download_initial_skymodel_server
        Place to download the initial sky model from (default = ``TGSS``). This can
        either be ``TGSS`` to use the TFIR GMRT Sky Survey, ``LOTSS`` to use the LOFAR
        Two-metre Sky Survey, or ``GSM`` to use the Global Sky Model.

    download_overwrite_skymodel
        Overwrite any existing sky model with a downloaded one (default = ``False``).

    input_skymodel
        Full path to the input sky model file, with true-sky fluxes (required if automatic
        generation or download is disabled). If you also have a sky model with apparent
        flux densities, specify it with the :term:`apparent_skymodel` option.

	See :doc:`preparation` for more info on preparing the sky model.

    apparent_skymodel
        Full path to the input sky model file, with apparent-sky fluxes (optional). Note
        that the source names must be identical to those in the :term:`input_skymodel`.

    regroup_input_skymodel
        Regroup input skymodel as needed to meet target flux (default = ``True``). If
        False, the existing patches are used for the calibration.

    strategy
        Name of processing strategy to use (default = ``selfcal``). A custom strategy can
        be used by giving instead the full path to the strategy file. See
        :ref:`rapthor_strategy` for details on the available predefined strategies and on
        making a custom strategy file.

    selfcal_data_fraction
        Fraction of data to use (default = 0.2). If less than one, the input data are
        divided by time into chunks that sum to the requested fraction, spaced out evenly
        over the full time range. Using a low value (0.2 or so) is strongly recommended
        for typical 8-hour, full-bandwidth observations.

    final_data_fraction
        A final data fraction can be specified (default = 1.0) such that a final
        processing pass (i.e., after selfcal finishes) is done with a different fraction.

    input_h5parm
        Full path to an H5parm file with direction-dependent solutions (default = None).
        This file is used if no calibration is to be done.

        .. note::

            The directions in the H5parm file must match the patches in the input sky
            model, and the time and frequency coverage must be sufficient to cover the
            duration and bandwidth of the input dataset.

    input_fulljones_h5parm
        Full path to an H5parm file with full-Jones solutions (default = None). This
        file is used if no calibration is to be done.

    facet_layout
        Full path to a text file that defines the facet layout (default = None). This file
        must use the WSClean facet format, specified in the `WSClean documentation
        <https://wsclean.readthedocs.io/en/latest/ds9_facet_file.html>`_. Also note that
        the facet centroids (the `facet point of interest
        <https://wsclean.readthedocs.io/en/latest/ds9_facet_file.html#adding-a-facet-point
        -of-interest>`_) must be defined in the file as well. If a facet file is supplied,
        calibration patches and imaging facets will be set to those specified in the file,
        if possible, and the calibrator selection parameters specified in the strategy
        (e.g., :term:`target_flux`) will be ignored (and therefore the patch and facet
        layout will be held constant between cycles)

        .. note::

            In a given cycle, the calibration patches and imaging facets will match the
            input facet layout unless the layout would result in one or more empty
            calibration patches, in which case the empty patches are removed and the
            layout of the remaining patches is set using Voronoi tessellation.

    dde_mode
        Mode to use to derive and correct for direction-dependent effects: ``faceting`` or
        ``hybrid`` (default = ``faceting``). If ``faceting``, Voronoi faceting is used
        throughout the processing. If ``hybrid``, faceting is used only during the self
        calibration steps; in the final cycle (done after self calibration has been
        completed successfully), IDGCal is used during calibration to generate smooth 2-D
        screens that are then applied by WSClean in the final imaging step.

        .. note::

            The hybrid mode is not yet available; it will be enabled in a future
            update.

.. _parset_calibration_options:

``[calibration]``
-----------------

.. glossary::

    use_image_based_predict
        Use image-based prediction (default = ``False``)? Image-based prediction can be
        faster than the normal prediction, especially for large sky models.

    llssolver
        The linear least-squares solver to use (one of ``qr``, ``svd``, or ``lsmr``;
        default = ``qr``).

    maxiter
        Maximum number of iterations to perform during calibration (default = 150).

    propagatesolutions
        Propagate solutions to next time slot as initial guess (default = ``True``)?

    solveralgorithm
        The algorithm used for solving (one of ``directionsolve``, ``directioniterative``,
        ``lbfgs``, or ``hybrid``; default = ``directioniterative``). When using ``lbfgs``,
        the :term:`stepsize` should be set to a small value like 0.001.

    onebeamperpatch
        Calculate the beam correction once per calibration patch (default = ``False``)? If
        ``False``, the beam correction is calculated separately for each source in the
        patch. Setting this to ``True`` can speed up calibration and prediction, but can
        also reduce the quality when the patches are large.

    parallelbaselines
        Parallelize model calculation over baselines, instead of parallelizing over
        directions (default = ``False``).

    sagecalpredict
        Use SAGECal for model calculation, both in predict and calibration (default =
        ``False``).

    fast_datause
        This parameter sets the visibilities mode used during the fast-phase solves  (one
        of ``single``, ``dual``, or ``full``; default = ``single``). If set to ``single``,
        the XX and YY visibilities are averaged together to a single (Stokes I)
        visibility. If set to ``dual``, only the XX and YY visibilities are used (YX and
        XY are not used). If set to ``full``, all visibilities are used. Activating the
        ``single`` or ``dual`` mode improves the speed of the solves and lowers the memory
        usage during solving.

        .. note::

            Currently, only :term:`solveralgorithm` = ``directioniterative`` is supported
            when using ``single`` or ``dual`` modes. If one of these modes is activated
            and a different solver is specified, the solver will be automatically switched
            to the ``directioniterative`` one.

    slow_datause
        This parameter sets the the visibilities used during the slow-gain solves  (one
        of ``dual`` or ``full``; default = ``dual``). If set to ``dual``, only the XX and
        YY visibilities are used (YX and XY are not used). If set to ``full``, all
        visibilities are used. Activating the ``dual`` mode improves the speed of the
        solves and lowers the memory usage during solving.

        .. note::

            Currently, only :term:`solveralgorithm` = ``directioniterative`` is supported
            when using the ``dual`` mode. If this modes is activated
            and a different solver is specified, the solver will be automatically switched
            to the ``directioniterative`` one.

    stepsize
        Size of steps used during calibration (default = 0.02). When using
        :term:`solveralgorithm` = ``lbfgs``, the stepsize should be set to a small value
        like 0.001.

    stepsigma
        In order to stop solving iterations when no further improvement is seen, the mean
        of the step reduction is compared to the standard deviation multiplied by
        :term:`stepsigma` factor (default = 2.0). If mean of the step reduction is lower
        than this value (noise dominated), solver iterations are stopped since no possible
        improvement can be gained.

    tolerance
        Tolerance used to check convergence during calibration (default = 5e-3).

    fast_freqstep_hz
        Frequency step used during fast phase calibration, in Hz (default = 1e6).

    fast_smoothnessconstraint
        Smoothness constraint bandwidth used during fast phase calibration, in
        Hz (default = 3e6).

    fast_smoothnessreffrequency
        Smoothness constraint reference frequency used during fast phase calibration, in
        Hz. If not specified this will automatically be set to 144 MHz for HBA or the
        midpoint of the frequency coverage for LBA.

    fast_smoothnessrefdistance
        Smoothness constraint reference distance used during fast phase calibration, in
        m (default = 0).

    slow_freqstep_hz
        Frequency step used during slow amplitude calibration, in Hz (default = 1e6).

    slow_smoothnessconstraint
        Smoothness constraint bandwidth used during the slow gain calibration, in Hz
        (default = 3e6).

    fulljones_timestep_sec
        Time step used during the full-Jones gain calibration, in seconds (default = 600).

    fulljones_freqstep_hz
        Frequency step used during full-Jones amplitude calibration, in Hz (default = 1e6).

    fulljones_smoothnessconstraint
        Smoothness constraint bandwidth used during the full-Jones gain calibration,
        in Hz (default = 0).

    dd_interval_factor
        Maximum factor by which the direction-dependent solution intervals can be
        increased, so that fainter calibrators get longer intervals (in the fast and slow
        solves only; default = 1). The value determines the maximum allowed adjustment
        factor by which the solution intervals are allowed to be increased for faint
        sources. For a given direction, the adjustment is calculated from the ratio of the
        apparent flux density of the calibrator to the target flux density of the cycle
        (set in the strategy) or, if a target flux density is not defined, to that of the
        faintest calibrator in the sky model. A value of 1 disables the use of
        direction-dependent solution intervals; a value greater than 1 enables
        direction-dependent solution intervals.

        .. note::

            Direction-dependent solution intervals are not currently supported;
            they will be re-enabled in a future update.

        .. note::

            Currently, only :term:`solveralgorithm` = ``directioniterative`` is supported
            when using direction-dependent solution intervals. If direction-dependent
            solution intervals are activated and a different solver is specified, the
            solver will be automatically switched to the ``directioniterative`` one.

    dd_smoothness_factor
        Maximum factor by which the smoothnessconstraint can be increased, so that
        fainter calibrators get more smoothing (default = 3). The factors are calculated
        in the same way as the direction-dependent interval factors, set by
        :term:`dd_interval_factor`. A value of 1 disables the use of direction-dependent
        smoothness factors; a value greater than 1 enables direction-dependent smoothness
        factors.

    solverlbfgs_dof
        Degrees of freedom for the LBFGS solver (only used when :term:`solveralgorithm` =
        ``lbfgs``; default = 200.0).

    solverlbfgs_minibatches
        Number of minibatches for the LBFGS solver (only used when :term:`solveralgorithm`
        = ``lbfgs``; default = 1).

    solverlbfgs_iter
        Number of iterations per minibatch in the LBFGS solver (only used when
        :term:`solveralgorithm` = ``lbfgs``; default = 4).

    bda_timebase
        Maximum baseline used in baseline-dependent time averaging (BDA) during the
        calibration, in m (default = 20000). A value of 0 will disable the averaging.
        Depending on the solution time step used during the calibration,
        activating this option may improve the speed of the solve and lower the memory
        usage during solving.

    bda_frequencybase
        Maximum baseline used in baseline-dependent frequency averaging (BDA) during the
        calibration, in m (default = 20000). A value of 0 will disable the averaging.
        Depending on the solution time step used during the calibration,
        activating this option may improve the speed of the solve and lower the memory
        usage during solving.

.. _parset_imaging_options:

``[imaging]``
-------------

.. glossary::

    cellsize_arcsec
        Pixel size in arcsec (default = 1.5).

    robust
        Briggs robust parameter (default = -0.65).

    min_uv_lambda
        Minimum uv distance in lambda to use in imaging (default = 80).

    max_uv_lambda
        Maximum uv distance in lambda to use in imaging (default = 1e6).

    mgain
        Cleaning gain for major iterations, passed to the imager (default = 0.8). This
        setting does not affect the first "initial_image" round.

    taper_arcsec
        Taper to apply when imaging, in arcsec (default = 0).

    local_rms_strength
        Strength to use for the local RMS thresholding (default = 0.8). The
        strength is applied by WSClean to the local RMS map using ``local_rms ^
        strength``.

    local_rms_window
        Size of the window (in number of PSFs) to use for the local RMS thresholding
        (default = 50).

    local_rms_method
        Method to use for the local RMS thresholding: ``rms`` or ``rms-with-min``
        (default = ``rms``).

    do_multiscale_clean
        Use multiscale cleaning (default = ``True``)?

    bda_timebase
        Maximum baseline used in baseline-dependent averaging (BDA) during imaging, in m
        (default = 20000). A value of 0 will disable the averaging. Activating this option
        may improve the speed of imaging.

    dde_method
        Method to use to correct for direction-dependent effects during imaging:
        ``single`` or ``full`` (default = ``full``). If ``single``, a single,
        direction-independent solution (i.e., constant across the image sector) will be
        applied for each sector. In this case, the solution applied is the one in the
        direction closest to each sector center. If ``full``, the full,
        direction-dependent solutions are applied (using either facets or screens).

    filter_skymodel
        Filter out sky model components that lie outside of islands detected by PyBDSF
        (default = ``True``). If ``True``, only clean components from WSClean whose
        centers lie inside of detected islands are kept in the sky model used for
        calibration in the next cycle. If ``False``, all clean components generated by
        WSClean are kept in the sky model.

        .. note::

            It is not recommneded to turn off filtering of the sky model unless the
            :term:`use_image_based_predict` parameter is set, as the sky model without
            filtering can be very large (resulting in runtimes becoming very long unless
            image-based predict is used).

    save_visibilities
        Save visibilities used for imaging (default = ``False``). If ``True``, the imaging
        MS files will be saved, with the the direction-independent full-Jones solutions,
        if available, applied. Note, however, that the direction-dependent solutions will
        not be applied unless :term:`dde_method` = ``single``, in which case the solutions
        closest to the image centers are used.

    save_supplementary_images
        Save dirty images and the clean masks made during each imaging cycle (default =
        ``False``).

    compress_selfcal_images
        Compress intermediate selfcal images to reduce storage space (default = ``True``). Uses default
        ``fpack`` compression parameters, see `fpack documentation <https://heasarc.gsfc.nasa.gov/fitsio/fpack/>`_ 
        for details on precision. Some tools may be unable to read compressed fits files and will
        require decompression to be run first. This can be done with the ``funpack`` tool .

    compress_final_images
        Compress the final images to reduce storage space (default = ``False``).
        See :term:`compress_selfcal_images` option for compression details.

    idg_mode
        IDG (image domain gridder) mode to use in WSClean (default = ``cpu``). The mode
        can be ``cpu`` or ``hybrid``.

    mem_gb
        Maximum memory in GB (per node) to use for WSClean jobs (default = 0 = all
        available memory).

        .. note::

            If the :term:`mem_per_node_gb` parameter is set, then the maximum memory
            for WSClean jobs will be set to the smaller of ``mem_gb`` and
            ``mem_per_node_gb``.

    apply_diagonal_solutions
        Apply separate XX and YY corrections during facet-based imaging (default =
        ``True``). If ``False``, scalar solutions (the average of the XX and YY
        solutions) are applied instead. (Separate XX and YY corrections are always applied
        when using non-facet-based imaging methods.)

    make_quv_images
        Make Stokes QUV images in addition to the Stokes I image (default = ``False``).
        If ``True``, Stokes QUV images are made during the final imaging step, once self
        calibration has been completed.

    pol_combine_method
        The method used to combine the polarizations during deconvolution can also be
        specified. This method can be "link" to linked polarization cleaning or "join" to
        use joined polarization cleaning (default = link). When using linked cleaning,
        the Stokes I image is used for cleaning and its clean components are subtracted
        from all polarizations.

    dd_psf_grid
        The number of direction-dependent PSFs which should be fit horizontally and
        vertically in the image (default = ``[0, 0]`` = scale with the image size, with
        approximately one PSF per square degree of imaged area). Set to ``[1, 1]`` to use
        a direction-independent PSF.

    use_mpi
        Use MPI to distribute WSClean jobs over multiple nodes (default = ``False``)? If
        ``True`` and more than one node can be allocated to each WSClean job (i.e.,
        ``max_nodes`` / ``num_images`` >= 2), then distributed imaging will be used (only
        available if :term:`batch_system` = ``slurm``).

        .. note::

            If MPI is activated, :term:`dir_local` (under the
            :ref:`parset_cluster_options` section below) must not be set unless it is on a
            shared filesystem.

        .. note::

            Currently, Toil does not fully support ``openmpi``. Because of this, imaging
            can only use the worker nodes, and the master node will be idle.

        .. note::
            When running on SKAO cluster, be sure to export the ``SALLOC_PARTITION`` to 
            ensure Toil uses a specific partition (see example SLURM script `here
            <https://git.astron.nl/RD/rapthor/-/blob/master/examples/rapthor_skao_multinode.slurm>`_).

    reweight
        Reweight the visibility data before imaging (default = ``False``). If ``True``,
        data with high residuals (compared to the predicted model visibilities) are
        down-weighted. This feature is experimental and should be used with caution.

    grid_width_ra_deg
        Size of area to image when using a grid (default = 1.7 * mean FWHM of the primary
        beam).

    grid_width_dec_deg
        Size of area to image when using a grid (default = 1.7 * mean FWHM of the primary
        beam).

    grid_center_ra
        Center of area to image when using a grid (default = phase center).

    grid_center_dec
        Center of area to image when using a grid (default = phase center).

    grid_nsectors_ra
        Number of sectors along the RA axis (default = 0). The number of sectors in Dec
        will be determined automatically to ensure the whole area specified with
        :term:`grid_center_ra`, :term:`grid_center_dec`, :term:`grid_width_ra_deg`, and
        :term:`grid_width_dec_deg` is imaged. Set to 0 to force a single sector for the
        full area. A grid of sectors can be useful for computers with limited memory but
        generally will give inferior results compared to an equivalent single sector.

    sector_center_ra_list
        List of image centers (default = ``[]``). Instead of a grid, imaging sectors can
        be defined individually by specifying their centers and widths.

    sector_center_dec_list
        List of image centers (default = ``[]``).

    sector_width_ra_deg_list
        List of image widths, in degrees (default = ``[]``).

    sector_width_dec_deg_list
        List of image  widths, in degrees (default = ``[]``).

    max_peak_smearing
        Max desired peak flux density reduction at center of the image edges due to
        bandwidth smearing (at the mean frequency) and time smearing (default = 0.15 = 15%
        reduction in peak flux). Higher values result in shorter run times but more
        smearing away from the image centers.

    skip_final_major_iteration
        Skip the final WSClean major iteration for all but the last processing cycle
        (default = ``True``). If ``True``, the final iteration is skipped during
        imaging, which speeds up imaging but degrades the image slightly;
        however, the sky model is not affected by this setting. Therefore, it is
        safe to use this option for self calibration cycles.

        .. note::

            The final WSClean major iteration is never skipped in the final
            processing cycle regardless of this setting.

    skip_corner_sectors
        Skip corner sectors defined by the imaging grid (default = ``False``)? If ``True``
        and a grid is used (defined by the ``grid_*`` parameters above), the four corner
        sectors are not processed (if possible for the given grid).

.. _parset_cluster_options:

``[cluster]``
-------------

.. glossary::

    batch_system
        Cluster batch system (only used when either StreamFlow or Toil is the CWL runner;
        default = ``single_machine``). Use ``single_machine`` when running on a single
        machine and ``slurm`` to use multiple nodes of a Slurm-based cluster.

        .. note::

            When using the ``slurm`` batch system, additional Slurm arguments can be
            passed to Toil by setting the ``TOIL_SLURM_ARGS`` environment variable in
            your environment before running Rapthor. See the Toil
            `environment variables <https://toil.readthedocs.io/en/latest/appendices/environment_vars.html>`_
            page for details.

    max_nodes
        When :term:`batch_system` = ``slurm``, the maximum number of nodes of the cluster
        to use at once (default = 12).

    cpus_per_task
        When :term:`batch_system` = ``slurm``, the number of processors per task to
        request (default = 0 = all). By setting this value to the number of processors per
        node, one can ensure that each task gets the entire node to itself, which is the
        recommended way of running Rapthor.

    mem_per_node_gb
        When :term:`batch_system` = ``slurm``, the amount of memory per node in GB to
        request (default = 0 = all).

    max_cores
        Maximum number of cores per task to use on each node (default = 0 = all).

    max_threads
        Maximum number of threads per task to use on each node (default = 0 = all).

    deconvolution_threads
        Number of threads to use by WSClean during deconvolution (default = 0 = 2/5 of
        ``max_threads``, but not more than 14).

    parallel_gridding_threads
        Number of threads to use by WSClean for parallel gridding (default = 0 = 2/5 of
        ``max_threads``, but not more than 6).

    dir_local
        Full path to a local disk on the nodes for IO-intensive processing (default = not
        used). The path must exist on all nodes (but does not have to be on a shared
        filesystem). This parameter is useful if you have a fast local disk (e.g., an SSD)
        that is not the one used for :term:`dir_working`. If this parameter is not set,
        IO-intensive processing (e.g., WSClean) will use a default path in
        :term:`dir_working` instead.

        .. note::

            This parameter should not be set in the following situations:

            - when :term:`batch_system` = ``single_machine`` and multiple imaging sectors
              are used (as each sector will overwrite files from the other sectors).

            - when :term:`use_mpi` = ``True`` under the :ref:`parset_imaging_options`
              section and ``dir_local`` is not on a shared filesystem.

        .. attention::

            This parameter is deprecated. Use :term:`local_scratch_dir` instead.

    local_scratch_dir
        Full path to a local disk on the nodes for IO-intensive processing (default =
        ``/tmp``). When :term:`batch_system` = ``slurm``, the path must exist on all the
        compute nodes, but not necessarily on the head node.
        This parameter is useful if you have a fast local disk (e.g., an SSD)
        that is not the one used for :term:`dir_working`. If this parameter is not set,
        IO-intensive processing (e.g., WSClean) will use a default path in
        :term:`dir_working` instead.

        When :term:`cwl_runner` = ``toil`` and :term:`batch_system` = ``single_machine``,
        it is recommended to set this parameter, so that Rapthor can clean up any
        temporary files and directories that Toil left behind.

        .. warning::

            If you want to run multiple instances of Rapthor concurrently using Toil,
            make sure that you specify different directories as
            :term:`local_scratch_dir`. Otherwise, one Rapthor instance will
            potentially clobber files/directories created by another instance.

    global_scratch_dir
        Full path to a directory on a shared disk that is readable and writable by all
        the compute nodes and the head node. This directory will be used to store the
        intermediate outputs that need to be shared between the different steps in the
        workflow. If this parameter is not set and :term:`batch_system` = ``slurm``,
        then Rapthor will create a temporary directory in :term:`dir_working`.

        When :term:`cwl_runner` = ``toil``, it is recommended to set this parameter, so
        that Rapthor can clean up any temporary files and directories that Toil left
        behind.

        .. warning::

            If you want to run multiple instances of Rapthor concurrently using Toil,
            make sure that you specify different directories as
            :term:`global_scratch_dir`. Otherwise, one Rapthor instance will
            potentially clobber files/directories created by another instance.

    use_container
        Run the workflows inside a container (default = ``False``)? If ``True``, the CWL
        workflow for each operation (such as calibrate or image) will be run inside a
        container. The type of container can be specified with the :term:`container_type`
        parameter.

        .. note::

            This option should not be used when Rapthor itself is being run inside a
            container. See :ref:`using_containers` for details.

    container_type
        The type of container to use when :term:`use_container` = ``True``. The supported
        types are: ``docker`` (the default), ``udocker``, or ``singularity``.

    cwl_runner
        CWL runner to use. Currently supported runners are: ``cwltool``, ``streamflow``,
        and ``toil`` (default). Toil is the recommended runner, since it provides much
        more fine-grained control over the execution of a workflow. For example, Toil and
        StreamFlow can use Slurm to automatically distribute workflow steps over different
        compute nodes, whereas CWLTool can only execute workflows on a single node. With
        CWLTool you also run the risk of overloading your machine when too many jobs are
        run in parallel. For debugging purposes CWLTool outshines Toil, because its logs
        are easier to understand.

    debug_workflow
        Debug workflow related issues (default = ``False``). Enabling this option
        implies that temporary files, produced during the workflow run, will be kept
        (i.e. the option ``keep_temporary_files`` is implicitly set to ``True``). This
        will require significantly more disk space.  The working directory will never be
        cleaned up, ``stdout`` and ``stderr`` will not be redirectied, and log level of
        the CWL runner will be set to ``DEBUG``.  Additionally, when using Toil as the
        CWL runner, some tasks will run using only a single thread (to make debugging
        easier). Use this option with care!

        .. note::

            If Toil is the CWL runner, this option will only work when
            :term:`batch_system` = ``single_machine`` (the default).

    keep_temporary_files
        Keep temporary files created during the workflow execution (default =
        ``False``). If ``True``, temporary files and directories created during the
        workflow execution will not be deleted at the end of the run. This will require
        significantly more disk space. This option is useful for debugging purposes.

        .. note::

            This option will be set to ``True`` automatically when
            :term:`debug_workflow` = ``True``.
