.. _rapthor_strategy:

Setting the processing strategy
===============================

The steps used during a Rapthor run are defined though the choice of a processing
strategy. The processing strategy is set with the :term:`strategy` parameter of the
Rapthor parset. The options for this parameter are described below:

``strategy = selfcal`` (the default)
    This strategy performs self calibration of the field in which the sky model is
    iteratively improved though calibration and imaging. The processing generally involves
    up to two cycles of phase-only calibration followed by up to 6 cycles of phase and
    amplitude calibration until convergence is obtained.

    .. note::

        If an initial sky model is automatically generated from the data (see
        :ref:`auto_sky_generation`), the phase-only cycles are not generally
        needed and are therefore skipped.

``strategy = image``
    This strategy performs imaging only; no calibration is done. As such, a file
    containing the direction-dependent corrections must be supplied via the
    :term:`input_h5parm` option in the parset. This strategy should be selected
    only when a full self-calibration has already been done. It is useful, for
    example, for reimaging the data at a different resolution. It can also be
    used to generate calibrated visibilities for a given source or sources (see
    :term:`save_visibilities`).

    .. note::

        If peeling of outlier or bright sources is desired, an initial sky model
        with patches that corresponds to the input direction-dependent
        corrections must also be supplied via the :term:`input_skymodel` option
        in the parset.

``strategy = /path/to/custom_strategy.py``
    By giving the path to a strategy file, the user can define a custom processing
    strategy. See below for details of how to make this file.


.. _custom_strategy:

Defining a custom processing strategy
-------------------------------------

Rapthor includes two predefined processing strategies (one designed for self calibration
of a typical LOFAR dataset and one for imaging only, see above for details). However,
depending on the field and the aims of the reduction, these predefined strategies may not
be sufficient. A custom processing strategy can be supplied by specifying the full path to
a strategy file in the :term:`strategy` entry of the Rapthor parset. The strategy file is
a Python file with the following structure:

.. code-block::

    """
    Script that defines a custom user processing strategy
    """
    strategy_steps = []
    n_cycles = 8
    for i in range(n_cycles):
        strategy_steps.append({})

        strategy_steps[i][parameter1] = value1
        strategy_steps[i][parameter2] = value2

As can be seen from this example, the file basically defines the variable
``strategy_steps``, which is a list of dictionaries. There is one entry (dictionary) in
the list per processing cycle. Each dictionary stores the processing parameters for that
cycle.


.. note::

    Examples of custom strategy files are available `here
    <https://git.astron.nl/RD/rapthor/-/blob/master/examples/custom_calibration_strategy.py>`_
    (for self calibration), `here
    <https://git.astron.nl/RD/rapthor/-/blob/master/examples/custom_imaging_strategy.py>`_
    (for imaging only) and `here
    <https://git.astron.nl/RD/rapthor/-/blob/master/examples/custom_ska_low.py>`_
    (for SKA low). Files that duplicate the default strategies are available `here
    <https://git.astron.nl/RD/rapthor/-/blob/master/examples/default_calibration_strategy.py>`_
    (for self calibration) and `here
    <https://git.astron.nl/RD/rapthor/-/blob/master/examples/default_imaging_strategy.py>`_
    (for imaging only).

.. note::

    If the strategy performs self calibration, the last entry in ``strategy_steps`` can be
    used to specify parameters specific to the final cycle (which is performed after
    self calibration finishes). In the default self calibration strategy, the parameters
    for the final cycle are set to those of the last cycle of selfcal.

.. note::

    If no self calibration is to be done, only a single processing cycle will be done.
    Therefore, ``strategy_steps`` should have only a single entry.

The following processing parameters can be set for each cycle:

.. glossary::

    do_calibrate
        Boolean flag that determines whether the calibration step should be done for this cycle.

    solve_min_uv_lambda
        Float that sets the minimum uv distance in lambda used during calibration for this cycle (applies to both fast-phase and slow-gain solves).

    fast_timestep_sec
        Float that sets the solution interval in sec to use in the fast (scalarphase) solve. For this solve, all the core stations are constrained to have the same solutions.

    medium_timestep_sec
        Float that sets the solution interval in sec to use in the medium-fast (scalarphase) solves. For the first medium-fast solve, each station is solved for independently. For the second medium-fast solve (done only when ``do_slowgain_solve`` is activated), the core stations are constrained to have the same solutions.

    do_slowgain_solve
        Boolean flag that determines whether the slow (diagonal) solve should be done for this cycle. If enabled, a slow solve is done, followed by a second medium-fast solve.

    slow_timestep_sec
        Float that sets the solution interval in sec to use in the slow-gain solve. For this solve, each station is solved for independently.

    do_fulljones_solve
        Boolean flag that determines whether the direction-independent full-Jones part of calibration should be done for this cycle.

    peel_outliers
        Boolean flag that determines whether the outlier sources (sources that lie outside of any imaging sector region) should be peeled for this cycle. Outliers can only be peeled once (unlike bright sources, see below), as they are not added back for subsequent selfcal cycles. Note that, because they are not imaged, outlier source models do not change during self calibration: however, the solutions they receive may change. To include one or more outlier sources in self calibration, a small imaging sector can be placed on each outlier of interest. The outliers will than be imaging and its model updated with the rest of the field.

    peel_bright_sources
        Boolean flag that determines whether the bright sources should be peeled for this cycle (for imaging only). The peeled bright sources are added back before subsequent selfcal cycles are performed (so they are included in the calibration, etc.). Currently, peeling is not supported when screens are used.

    max_normalization_delta
        Float that sets the maximum allowed fractional delta from unity for the per-station normalization.

    scale_normalization_delta
        Boolean flag that determines whether the maximum allowed fractional normalization delta (set by the ``max_normalization_delta`` parameter) is constrained to vary linearly with distance from the phase center. If True, the maximum delta is zero at the phase center and reaches the value set by ``max_normalization_delta`` for the most distant calibration patch. If False, the maximum delta is the same for all calibration patches.

    do_normalize
        Boolean flag that determines whether the normalization of the flux scale is done. This normalization determines and applies the corrections (as a function of frequency) needed to achieve obs_flux / true_flux = 1. The "true" flux is determined by cross matching with the VLSSr and WENSS catalogs.

    do_image
        Boolean flag that determines whether the imaging step should be done for this cycle.

    auto_mask
        Float that sets WSClean's automask value for this cycle.

    auto_mask_nmiter
        Integer that sets the maximum number of WSClean's major iterations done once the automasking threshold is reached for this cycle.

    threshisl
        Float that sets PyBDSF's threshisl value for this cycle.

    threshpix
        Float that sets PyBDSF's threshpix value for this cycle.

    max_nmiter
        Integer that sets the maximum number of major iterations done during imaging for this cycle.

    channel_width_hz
        Float that sets the target bandwidth in Hz of each output image channel.

    target_flux
        Float (or ``None``) that sets the target flux density in Jy for DDE calibrators for this cycle. If ``None``, a value must be specified for ``max_directions``.

    max_directions
        Integer (or ``None``) that sets the maximum number of directions (DDE calibrators) used during calibration for this cycle. If ``None``, a value must be specified for ``target_flux``. If both ``max_directions`` and ``target_flux`` are specified, the specified target flux density is used unless it would result in more than the specified maximum number of directions, in which case the target flux density is increased to ensure that the maximum number of directions is not exceeded.

    max_distance
        Float (or ``None``) that sets the maximum distance in degrees from the phase center for DDE calibrators for this cycle. If ``None``, all sources in the sky model are considered to be potential calibrators. This cut is made before the cuts due to the target flux (``target_flux``) or maximum number of directions (``max_directions``).

    regroup_model
        Boolean flag that determines whether the sky model should be regrouped for this cycle.

    do_check
        Boolean flag that determines whether the check for self-calibration convergence should be done for this cycle.

    convergence_ratio
        Float that sets the minimum ratio of the current image noise to the previous image noise above which selfcal is considered to have converged (must be in the range 0.5 -- 2). A check is also done for the image dynamic range and number of sources, where the ratio of the current to previous value must be below 1 / ``convergence_ratio``. Selfcal is considered to have converged only if all of these conditions are met.

    divergence_ratio
        Float that sets the minimum ratio of the current image noise to the previous image noise above which selfcal is considered to have diverged (must be > 1).

    failure_ratio
        Float that sets the minimum ratio of the current image noise to the theoretical image noise above which selfcal is considered to have failed (must be > 1).

