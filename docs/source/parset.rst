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

    separation_tolerance_arcsec
        Sets the pointing separation tolerance (the maximum allowed separation between
        the pointings; default = 0.05 arcsec). This tolerance is used when more than
        one observation is included in :term:`input_ms` to ensure that the observation
        pointings are consistent with one another.

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
        Very small selections retain at least two time samples when available, as
        required for Dysco compression, even if this exceeds the requested fraction.

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

    ntimes_to_repeat_final_cycle
        Number of times to repeat the final cycle, if done (default = 0). Setting this
        parameter to a value of one or more may improve the final results but will increase
        the processing time

    input_h5parm
        Full path to an H5parm file with scalar phase or diagonal slow-gain
        calibration solutions (default = None). This file is used if no
        calibration is to be done. With an image-only DI strategy, Rapthor
        pre-applies this file during imaging preparation. With an image-only DD
        strategy, Rapthor applies this file during imaging and therefore needs
        matching sky-model patches or a facet layout.

        .. note::

            The time and frequency coverage must be sufficient to cover the duration
            and bandwidth of the input dataset.

        .. note::

            For DD use, the directions in the H5parm file must match the
            patches in the input sky model, or, if not, a facet layout file with
            matching directions must be supplied and
            :term:`regroup_input_skymodel` activated. DI use does not require
            an input sky model.

    input_fulljones_h5parm
        Full path to an H5parm file with DI full-Jones solutions (default =
        None). This file is used if no calibration is to be done. Rapthor
        pre-applies full-Jones solutions during imaging preparation, separately
        from the scalar phase or diagonal slow-gain products supplied through
        :term:`input_h5parm`.

    input_normalization_h5parm
        Full path to an H5parm file with flux-scale normalization solutions (default =
        None). This file is used if no calibration is to be done.

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

            The ``hybrid`` mode should be considered experimental.


.. _parset_calibration_options:

``[calibration]``
-----------------

.. glossary::

    use_included_skymodels
        Include a packaged sky model in calibration when it is within twice the primary
        beam FWHM of the field center (default = ``False``).

    use_image_based_predict
        Use image-based prediction (default = ``False``)? Image-based prediction can be
        faster than the normal prediction, especially for large sky models.

        .. note::

            Currently, correction for time and frequency smearing cannot be done
            when image-based prediction is used. If the averaging of the input data
            is such that time or frequency smearing is significant within the field
            of view of interest, then the use of image-based prediction is not
            recommended.

    use_wsclean_predict
        Use WSClean to predict calibration model columns before DD calibration solves
        (default = ``False``)? This is an alternative image-based prediction path for
        large sky models. It is mutually exclusive with :term:`use_image_based_predict`;
        if both are enabled, Rapthor keeps ``use_wsclean_predict`` and disables
        ``use_image_based_predict``.

    wsclean_predict_bw
        Bandwidth in Hz for each model image drawn during WSClean-based prediction
        (default = ``2e6``). Wide-band data are split into channel groups no wider
        than this value before model images are drawn and predicted into temporary
        model-data columns.

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
        the :term:`stepsize` should be set to a small value like 0.001. For IDGCal solves,
        the ``lbfgs`` algorithm is always used.

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

    medium_datause
        This parameter sets the visibilities mode used during the medium-phase solves  (one
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
        Frequency step used during the fast calibration, in Hz (default = 1e6).

    fast_smoothnessconstraint
        Smoothness constraint bandwidth used during the fast calibration, in
        Hz (default = 3e6).

    fast_smoothnessreffrequency
        Smoothness constraint reference frequency used during the fast calibration, in
        Hz. If not specified this will automatically be set to 144 MHz for HBA or the
        midpoint of the frequency coverage for LBA.

    fast_smoothnessrefdistance
        Smoothness constraint reference distance used during the fast calibration, in
        m (default = 0).

    medium_freqstep_hz
        Frequency step used during the medium-fast calibration, in Hz (default = 1e6).

    medium_smoothnessconstraint
        Smoothness constraint bandwidth used during the medium-fast calibration, in
        Hz (default = 6e6).

    medium_smoothnessreffrequency
        Smoothness constraint reference frequency used during the medium-fast calibration, in
        Hz. If not specified this will automatically be set to 144 MHz for HBA or the
        midpoint of the frequency coverage for LBA.

    medium_smoothnessrefdistance
        Smoothness constraint reference distance used during the medium-fast calibration, in
        m (default = 0).

    slow_freqstep_hz
        Frequency step used during the slow calibration, in Hz (default = 1e6).

    slow_smoothnessconstraint
        Smoothness constraint bandwidth used during the slow calibration, in Hz
        (default = 3e6).

    fulljones_freqstep_hz
        Frequency step used during the full-Jones calibration, in Hz (default = 1e6).

    fulljones_smoothnessconstraint
        Smoothness constraint bandwidth used during the full-Jones calibration,
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

    bda_timebase (calibration)
        Maximum baseline used in baseline-dependent time averaging (BDA) during the
        calibration, in m (default = 20000). A value of 0 will disable the averaging.
        Depending on the solution time step used during the calibration,
        activating this option may improve the speed of the solve and lower the memory
        usage during solving.

    bda_frequencybase (calibration)
        Maximum baseline used in baseline-dependent frequency averaging (BDA) during the
        calibration, in m (default = 20000). A value of 0 will disable the averaging.
        Depending on the solution time step used during the calibration,
        activating this option may improve the speed of the solve and lower the memory
        usage during solving.

    correct_time_frequency_smearing (calibration)
        Correct for time and frequency smearing during the prediction part of
        calibration (default = ``False``). Generally, if enabled and imaging is
        to be done, the identical parameter in the ``[imaging]`` section should
        also be enabled.

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

    bda_timebase (imaging)
        Maximum baseline used in baseline-dependent time averaging (BDA) during
        imaging, in m (default = 20000). A value of 0 will disable time BDA.
        Activating this option may improve the speed of imaging.

    bda_frequencybase (imaging)
        Maximum baseline used in baseline-dependent frequency averaging (BDA) during
        imaging, in m (default = 20000). A value of 0 will disable frequency BDA.
        Activating this option may improve the speed of imaging.

        Frequency BDA produces a Measurement Set with multiple spectral windows.
        Primary-beam correction for this layout requires EveryBeam 0.8.3 or later.

        .. note::

            Currently, correction for time and frequency smearing cannot be done
            when BDA is used during imaging. If the averaging of the input data
            is such that time or frequency smearing is significant within the field
            of view of interest, then the use of BDA is not recommended.

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
            :term:`use_image_based_predict` or :term:`use_wsclean_predict` parameter is
            set, as the sky model without filtering can be very large (resulting in
            runtimes becoming very long unless image-based predict is used).

    source_finder
        Source finder used when filtering the sky model (default = ``bdsf``).

    save_visibilities
        Save visibilities used for imaging (default = ``False``). If ``True``, the imaging
        MS files will be saved, with the the direction-independent full-Jones solutions,
        if available, applied. Note, however, that the direction-dependent solutions will
        not be applied unless :term:`dde_method` = ``single``, in which case the solutions
        closest to the image centers are used.

    save_residual_visibilities
        Save residual visibilities for the final image (default = ``False``).
        If ``True``, WSClean keeps the model data required for the final image, and Rapthor
        writes residual Measurement Sets as DATA minus MODEL_DATA in
        ``visibilities/image_X/sector_Y``.

    average_visibilities
        Perform averaging of the input visibilities for imaging (default = ``True``),
        determined by the maximum allowed before smearing effects become important (see
        :term:`max_peak_smearing`).

        .. note::

            This works in conjunction with :term:`save_visibilities`, so that
            visibilities used in each imaging cycle can be saved without averaging
            (unless other averaging such as BDA is requested).

    save_image_cube
        Save frequency cube(s) for the given Stokes parameters (default = ``False``).
        If ``True``, a cube is constructed from the channel images made during the
        final imaging step, once self calibration has been completed. The width of the
        frequency channels in the cube is set by the :term:`channel_width_hz` parameter
        in the strategy file. The Stokes parameters for which cubes will be made should
        be specified with the :term:`image_cube_stokes_list` parameter.

    image_cube_stokes_list
        The list of Stokes parameters for which frequency cubes should be saved,
        specified as a comma-separated list (e.g., ``[Q, U]``; default = ``[I]``).

    save_supplementary_images
        Save the supplementary images during each imaging cycle (default = ``False``). For now,
        this is just the PyBDSF-generated masks used for filtering of the sky model (if
        :term:`filter_skymodel` = ``True``).

    save_filtered_model_image
        Save images of the filtered sky model made during each imaging cycle
        (default = ``False``).

    model_mosaic_method
        Method used to generate model mosaics from multiple imaging sectors
        (default = ``wsclean``). With ``wsclean``, the model mosaic is drawn by
        WSClean from the sky models of the sectors. With ``sparse_fits``, the
        model images of the sectors are regridded instead; this method is
        intended mainly for testing.

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

    disable_iquv_clean
        Disable clean for the full-Stokes imaging (default = ``False``). If ``True``,
        no cleaning is done during full-Stokes imaging.

        .. note::

            Clean can be disabled only for full-Stokes imaging (i.e., only when
            :term:`make_quv_images` = ``True``).

        .. warning::

            Disabling clean can speed up processing, but the resulting images
            (including Stokes I) may be of poor quality and/or have sources with
            incorrect flux densities.

    dd_psf_grid
        The number of direction-dependent PSFs which should be fit horizontally and
        vertically in the image (default = ``[0, 0]`` = scale with the image size, with
        approximately one PSF per square degree of imaged area). Set to ``[1, 1]`` to use
        a direction-independent PSF.

    use_mpi
        Use MPI to distribute WSClean jobs over multiple nodes (default = ``False``)? If
        ``True`` and more than one node can be allocated to each WSClean job (i.e.,
        ``max_nodes`` / ``num_images`` >= 2), then distributed imaging will be used (only
        available if :term:`batch_system` = ``slurm`` or ``slurm_static``). WSClean is
        started with ``mpirun``, with one process on each node.

        .. note::

            When MPI is used, WSClean's temporary files are placed in
            :term:`global_scratch_dir` if it is set, and otherwise in
            :term:`dir_working`, because all the nodes must be able to see them.

        .. note::

            Whether MPI works depends on how MPI and Slurm are set up on the
            cluster. Check this mode with a short run before using it for a
            full reduction.

    shared_facet_rw
        Permit WSClean's ``-shared-facet-reads`` and ``-shared-facet-writes``
        during facet imaging (default = ``False``). When requested, Rapthor
        chooses both flags together for each sector:

        * One facet: sharing is disabled.
        * Two or more facets: sharing is disabled only when channel concurrency
          per node exceeds facet concurrency. Otherwise sharing is enabled.

        Both candidate concurrency values respect the requested
        :term:`parallel_gridding_tasks`, :term:`max_cores`, and WSClean's actual
        thread budget. They use the output channels per node and the actual
        calibration-facet count, respectively. DD-PSF regions do not count as
        calibration facets. Different sectors may therefore use different
        sharing modes. Setting this option to ``False`` disables sharing at
        every facet count and node count.

        There is no fixed facet-count cutoff. This resource-based comparison
        is an experimental heuristic, not a measured performance crossover.
        The decision depends on the number of nodes, output channels, threads
        and configured task limits. See the
        `WSClean facet documentation
        <https://wsclean.readthedocs.io/en/latest/facet_based_imaging.html#enabling-shared-reads>`_.

        .. warning::
            This option and its selection policy are experimental. Shared I/O
            can save repeated reads and writes, but gain/beam corrections and
            facet geometry can outweigh those savings.

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
        smearing away from the image centers. Note this option is not considered if 
        :term:`average_visibilities` = ``False``.

    correct_time_frequency_smearing (imaging)
        Correct for time and frequency smearing during imaging (default =
        ``False``). Generally, if enabled and calibration is to be done, the
        identical parameter in the ``[calibration]`` section should also be
        enabled.

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

    use_clean_mask
        Use a clean mask during cleaning (default = ``False``)? If ``True``, the clean mask
        generated by WSClean is used during cleaning. If ``False``, no clean mask is used
        in wsclean.

    photometry_skymodel
        Full path to the sky model file for photometry comparison when generating 
        image diagnostics. If this is not set, a sky model will be downloaded from 
        the TGSS and LOTSS catalogs. If this is not possible a backup sky model will be
        downloaded from the NVSS catalog. Default = ``None`` (sky model will be 
        downloaded by default).

    astrometry_skymodel
        Full path to the sky model file for astrometry comparison when generating 
        image diagnostics. If this is not set, a sky model will be downloaded from 
        Pan-STARRS. Default = ``None`` (sky model will be downloaded by default).

    normalization_skymodels
        List of at least two sky model paths used for flux normalization when enabled
        by the strategy (see :term:`do_normalize`). The default is ``None``, which allows
        Rapthor to download suitable sky models when internet access is available.

    normalization_reference_frequencies
        Reference frequencies, in Hz, corresponding to ``normalization_skymodels``
        (default = ``None``).

.. _parset_cluster_options:

``[cluster]``
-------------

The options in this section set where and how the processing is run. For a run
on a single machine, the defaults are usually sufficient. See :ref:`running`
for how these options are used.

.. glossary::

    batch_system
        Cluster batch system (default = ``single_machine``). Use
        ``single_machine`` when running on a single machine, and ``slurm`` or
        ``slurm_static`` to use multiple nodes of a Slurm-based cluster.

        With ``single_machine``, Rapthor starts the Dask scheduler and workers
        that it needs itself. With ``slurm`` or ``slurm_static``, Rapthor uses a
        Dask scheduler and workers that you have started inside your Slurm job
        (see :ref:`running_on_cluster`), and the address of the scheduler must
        be given (see :term:`dask_scheduler`). Rapthor does not submit Slurm
        jobs itself.

        The ``slurm`` and ``slurm_static`` systems differ only when
        :term:`use_mpi` is set: with ``slurm_static``, all of the nodes set by
        :term:`max_nodes` are used for imaging with MPI, whereas with
        ``slurm`` one node fewer is used for each imaging sector.

    max_nodes
        When :term:`batch_system` is ``slurm`` or ``slurm_static``, the number
        of nodes of the cluster to use. The default is 0, which results in a
        value of 1 for ``single_machine`` and 12 for the Slurm batch systems.
        Set this to the number of nodes in your Slurm job. When all of the data
        are used (a data fraction of 1), each observation is split in time
        into up to this many balanced chunks, subject to the minimum calibration
        duration and two samples per chunk, so that the nodes can work on them
        at the same time.

    local_dask_workers
        Number of Dask workers to start when Rapthor is run on a single machine
        (default = 0 = one worker). Each worker runs one step of the
        processing at a time, so this is the number of DP3 or WSClean commands
        that can be run at the same time. See :ref:`local_dask_runtime` for
        when it is useful to set this.

    cpus_per_task
        The number of processors that each task may use (default = 0 = all the
        processors of the machine on which Rapthor is started). When
        :term:`batch_system` = ``slurm``, set this value to the number of
        processors per node, so that each task gets the entire node to itself,
        which is the recommended way of running Rapthor.

    mem_per_node_gb
        The amount of memory per node in GB that Rapthor may use (default = 0
        = all). When set, it limits the memory of each Dask worker that Rapthor
        starts on a single machine, the memory used by WSClean (see
        :term:`mem_gb`), and the amount of data that the predict operation
        holds in memory at once.

        Rapthor also uses this value for DP3 calibration memory checks. A preflight
        check uses each strategy step's maximum number of directions, and a second
        check before each calibration cycle uses the resolved facet count and DP3
        solve intervals. If ``mem_per_node_gb`` is zero, the checks compare against
        memory available on the machine running Rapthor. When running on multiple
        nodes, set ``mem_per_node_gb``, because the memory of the machine on which
        Rapthor is started may not be the same as that of the other nodes.

        The estimate uses decimal GB and a conservative current DP3 peak-memory model
        of 80 bytes per visibility sample. It includes the original data buffer,
        visibility copies, weights, and weighted data, and uses the unaveraged channel
        count because calibration BDA is baseline-dependent. By default, Rapthor logs
        likely out-of-memory configurations but continues processing.

        The estimated peak memory is calculated as follows::

            baselines = nstations * (nstations + 1) // 2
            time_steps = ceil(solution_interval_seconds / sampling_interval_seconds)
            samples = baselines * channels * time_steps * (directions + 1)

            visibility_copies_gb = samples * 4 * 8 / 1e9
            weights_gb = samples * 4 * 4 / 1e9
            weighted_data_gb = samples * 4 * 8 / 1e9
            peak_memory_gb = visibility_copies_gb + weights_gb + weighted_data_gb

        A DI solve always uses one direction. A DD preflight estimate uses the
        strategy step's ``max_directions`` value, while the resolved estimate uses the
        actual number of calibration facets. The additional direction in
        ``directions + 1`` accounts for DP3 retaining the original data buffer.

    fail_on_calibration_oom_risk
        Stop Rapthor when a DP3 calibration memory estimate is greater than the
        applicable memory limit (default = ``False``). This applies to both the
        preflight upper bound and the resolved estimate immediately before each
        calibration cycle. An estimate exactly equal to the limit is allowed.

        When enabled, a high-risk estimate raises an actionable error before the
        affected pipeline operation starts. If capacity cannot be determined or the
        estimate cannot be calculated, the check remains advisory.

    max_cores
        Maximum number of cores per task to use on each node (default = 0 = all).

    max_threads
        Maximum number of threads per task to use on each node (default = 0 = all).

    filter_skymodel_ncores
        Number of cores used by PyBDSF when filtering the sky model during
        imaging (default = 15). Set to 0 to use :term:`max_threads`. This
        value can be set separately from :term:`max_threads`, since the
        filtering may run faster with fewer cores than DP3 or WSClean use.

    deconvolution_threads
        Number of threads to use by WSClean during deconvolution (default = 0 = 2/5 of
        ``max_threads``, but not more than 14).

    parallel_gridding_tasks
        Number of task groups WSClean can use for parallel gridding. If this is
        set to 0 (default), Rapthor uses ``max_threads // 8`` with a minimum of
        1. During imaging, Rapthor caps the value by :term:`max_cores` and the
        actual calibration-facet count when shared facet I/O is active. Without
        sharing, it instead uses a conservative cap of output channels per
        node (rounded down, with a minimum of one); independent channel/facet
        tasks can then overlap.
        It chooses a divisor of WSClean's actual ``-j`` thread count:
        :term:`max_threads` locally, or :term:`cpus_per_task` with MPI.

        For example, with 20 output channels, 192 threads per rank, three nodes,
        ``max_cores = 192`` and a requested limit of 24 tasks, two to five facets
        use six parallel gridders with sharing disabled. Six facets use six
        gridders and twelve facets use twelve gridders when
        :term:`shared_facet_rw` is enabled.
        Under the same CPU and task limits, sharing is selected as follows
        when :term:`shared_facet_rw` is enabled:

        .. list-table:: Shared I/O and parallel gridding for 20 output channels
            :header-rows: 1

            * - Nodes
              - P with sharing disabled
              - Minimum facets for sharing
            * - 1
              - 16
              - 16
            * - 3
              - 6
              - 6
            * - 5
              - 4
              - 4
            * - 10
              - 2
              - 2
            * - 20
              - 1
              - 2

        With sharing enabled, P follows the facet count and the CPU/thread
        and task limits above. For example, twelve facets use P=12 on three
        nodes, but use P=16 without sharing on one node. These examples
        describe the policy, not measured optimal node counts or runtimes.
        On a four-thread rank, for example, four facets can already provide
        P=4, so sharing is enabled even when many output channels are available.
        The effective facet count, sharing mode, channels, nodes, threads and
        gridding concurrency are recorded in the Rapthor log for each sector.

    local_scratch_dir
        Full path to a local disk on the nodes for the temporary files of each
        command, including those written by WSClean (default = ``None``). This
        parameter is useful if you have a fast local disk (e.g., an SSD). The
        path does not have to be on a shared filesystem, and does not have to
        exist on the machine on which Rapthor is started: Rapthor creates a
        directory for each command on the node that runs it and removes it
        when the command has finished. The path may contain environment
        variables (e.g., ``$TMPDIR``) and ``~``; these are expanded on the node
        that runs the command.

        If this parameter is not set, WSClean writes its temporary files to
        :term:`dir_working`, and the other commands use the default temporary
        directory of the node.

        When Rapthor is run on a single machine, the Dask workers that it
        starts also use this directory for their own temporary data. If you
        start the Dask workers yourself, set this with the
        ``--local-directory`` option of ``dask worker``.

        .. note::

            The temporary files are kept if :term:`keep_temporary_files` is
            set. If a directory cannot be removed, Rapthor logs a warning that
            names it and continues.

        .. note::

            PyBDSF always writes its temporary files to ``/tmp``. WSClean jobs
            that use MPI (see :term:`use_mpi`) use :term:`global_scratch_dir`
            instead of this directory, since all the nodes must be able to see
            their temporary files.

    global_scratch_dir
        Full path to a directory on a shared disk that is readable and writable
        by all the compute nodes and the head node (default = ``None``). If
        set, the intermediate files of each operation (the files that are
        passed from one step to the next) are written to this directory
        instead of to :term:`dir_working`.

        While an operation runs, ``dir_working/pipelines/<operation>`` is a
        link to a directory that Rapthor creates inside the global scratch
        directory. When the operation ends, whether it succeeded or failed,
        its files are moved back to ``dir_working/pipelines/<operation>``, so
        that the run can be resumed whether or not the scratch directory still
        exists. If Rapthor is interrupted before the files have been moved
        back, the move is completed the next time the run is resumed. The
        final products and the logs are always written to :term:`dir_working`.

    use_container
        Not supported, and must be left at its default of ``False``: a run in
        which this option is ``True`` stops with an error. To use a container,
        run Rapthor itself inside it. See :ref:`using_containers` for details.

    container_type
        Not used (see :term:`use_container`).

    dask_scheduler
        Address of a Dask scheduler that you have started yourself, for
        example ``tcp://127.0.0.1:8786`` (default = ``None``). If this is not
        set, the ``DASK_SCHEDULER`` environment variable is used when present.
        If neither is set, Rapthor starts its own scheduler and workers on the
        machine on which it is run. See :ref:`external_dask_runtime` for
        details.

    dask_dashboard_address
        Address at which the dashboard of the Dask scheduler started by Rapthor
        is made available, for example ``:8787`` (default = ``None``, which
        lets Dask choose).

    prefect_task_runner
        How the steps of each operation are run (default = ``None``). The
        default results in ``external_dask`` when a Dask scheduler is given
        with :term:`dask_scheduler` and ``local_dask`` otherwise, and does not
        normally need to be changed. With ``local_dask``, Rapthor starts its
        own Dask scheduler and workers. With ``external_dask``, Rapthor uses
        the scheduler given by :term:`dask_scheduler`. With ``sync``, the
        steps are run one at a time without Dask; this is intended for testing
        only.

    prefect_api_url
        URL of a Prefect server that you have started yourself, for example
        ``http://127.0.0.1:4200/api`` (default = ``None``). If this is not set,
        the ``PREFECT_API_URL`` environment variable is used when present. See
        :ref:`persistent_prefect_dashboard` for details.

    prefect_api_mode
        Sets which Prefect server is used: ``auto``, ``external``, or
        ``ephemeral`` (default = ``auto``). If ``auto``, the server given by
        :term:`prefect_api_url` is used if there is one, and otherwise a
        temporary server is used that exists only for the duration of the run.
        If ``external``, a server must be given with :term:`prefect_api_url`.
        If ``ephemeral``, a temporary server is always used, even if a server
        is given. In the ``auto`` and ``external`` modes, the run stops with an
        error if the server that was given cannot be reached.

        .. note::

            Any server set in your own Prefect configuration (a Prefect
            profile) is ignored. Each run that uses a temporary server keeps
            its Prefect data in its own temporary directory, so several runs
            can be done at the same time without affecting one another.

    prefect_run_tags
        Optional comma-separated list of labels to attach to the run in the
        Prefect dashboard, for example ``prefect_run_tags = ngc891,
        test-robust`` (default = ``None``). The labels can be used in the
        dashboard to find the run and everything that belongs to it.

    prefect_retries
        Number of times that a step that failed is tried again before the
        operation is stopped (default = ``0``).

    prefect_log_commands
        Record every command that is run in ``dir_working/logs/commands.jsonl``
        and write its output to ``dir_working/logs/<operation>/<task>.log``
        (default = ``True``). These files are written whether or not a Prefect
        dashboard is in use, and include the output of commands that failed.

    prefect_stream_output
        Show the output of DP3, WSClean, and the other commands in the logs of
        the Prefect dashboard as well (default = ``True``). This does not
        affect the log files written when :term:`prefect_log_commands` is
        set.

    prefect_command_profile
        Sets how the resource use of each command is measured: ``auto``,
        ``time``, or ``off`` (default = ``auto``). If ``auto`` or
        ``time``, the CPU time, memory use, and disk I/O of each command are
        recorded in ``dir_working/logs/commands.jsonl``, using
        ``/usr/bin/time`` when it is available. If ``off``, only the run time
        of each command is recorded.

    prefect_publish_fits_previews
        Make PNG previews of the images and show them in the Prefect dashboard
        (default = ``False``). The previews take extra time and disk space to
        make. The images themselves, the plots, and the image diagnostics are
        made whether or not this option is set.

    prefect_publish_postage_stamp_previews
        Make small PNG previews of the area around the brightest sources found
        by PyBDSF and show them in the Prefect dashboard (default =
        ``False``). The previews are made from the primary-beam-corrected
        image, are also saved in ``dir_working/.rapthor-artifacts/postage-stamps``,
        and are independent of the previews made with
        :term:`prefect_publish_fits_previews`.

    prefect_postage_stamp_preview_count
        Maximum number of sources for which a preview is made when
        :term:`prefect_publish_postage_stamp_previews` is set (default =
        ``5``).

    prefect_postage_stamp_preview_size_px
        Width and height, in image pixels, of each preview made when
        :term:`prefect_publish_postage_stamp_previews` is set (default =
        ``96``).

    prefect_fits_preview_clip_percentile
        Upper percentile used for FITS preview display limits (default =
        ``99.9``). Rapthor uses ``100 - value`` as the lower percentile, so the
        default displays whole-field previews and postage stamps with ``0.1``
        and ``99.9`` percentile clipping. Postage stamps use limits computed
        from the full FITS image plane, not from the cropped stamp, so their
        colour scale matches the corresponding full-field preview. Set this to
        ``100`` to use the finite minimum and maximum values.

    debug_workflow
        Has the same effect as :term:`keep_temporary_files` (default =
        ``False``).

    keep_temporary_files
        Keep the temporary and intermediate files of each operation (default =
        ``False``). If ``True``, these files will not be deleted when the
        operation has finished. This will require significantly more disk
        space. This option is useful for debugging purposes.

        .. note::

            This option will be set to ``True`` automatically when
            :term:`debug_workflow` = ``True``.

    allow_internet_access
        Allow internet access for downloading sky models when these are not provided.
        Default = ``True``. If ``False``, then the user must either provide the path
        to a sky model (see :term:`input_skymodel`) or set 
        :term:`generate_initial_skymodel` = ``True``. If photometry and/or astrometry
        skymodels are not provided, then these will not be downloaded and the image
        diagnostics will not be generated.
