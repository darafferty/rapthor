.. _rapthor_strategy:

Defining a custom processing strategy
=====================================

The default processing strategy is designed to perform full self calibration of
a typical LOFAR dataset. However, depending on the field and the aims of the
reduction, the default strategy may not be optimal. A custom processing strategy
can be supplied by specifying the full path to a strategy file in the
:term:`strategy` entry of the Rapthor parset. The strategy file is a Python file
with the following structure:

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

The file basically defines the variable ``strategy_steps``, which is a list of
dictionaries. There is one entry (dictionary) in the list per processing cycle. Each
dictionary stores the processing parameters for that cycle.

.. note::

    An example of a custom strategy file is available `here <https://git.astron.nl/RD/rapthor/-/blob/master/examples/custom_calibration_strategy.py>`_. An example of the default self calibration strategy is available `here <https://git.astron.nl/RD/rapthor/-/blob/master/examples/default_calibration_strategy.py>`_.

The following processing parameters must be set for each cycle:

.. glossary::

    do_calibrate
        Boolean flag that determines whether the calibration step should be done for this cycle.

    solve_min_uv_lambda
        Minimum uv distance in lambda used during calibration for this cycle (applies to both fast-phase and slow-gain solves).

    do_slowgain_solve
        Boolean flag that determines whether the slow-gain part of calibration should be done for this cycle.

    do_fulljones_solve
        Boolean flag that determines whether the direction-independent full-Jones part of calibration should be done for this cycle.

    peel_outliers
        Boolean flag that determines whether the outlier sources (sources that lie outside of any imaging sector region) should be peeled for this cycle. Outliers can only be peeled once (unlike bright sources, see below), as they are not added back for subsequent selfcal cycles. Note that, because they are not imaged, outlier source models do not change during self calibration: however, the solutions they receive may change. To include one or more outlier sources in self calibration, a small imaging sector can be placed on each outlier of interest. The outliers will than be imaging and its model updated with the rest of the field.

    peel_bright_sources
        Boolean flag that determines whether the bright sources should be peeled for this cycle (for imaging only). The peeled bright sources are added back before subsequent selfcal cycles are performed (so they are included in the calibration, etc.). Generally, peeling of bright sources is beneficial when using screens but not when using facets.

    max_normalization_delta
        Float that sets the maximum allowed fractional delta from unity for the per-station normalization.

    scale_normalization_delta
        Boolean flag that determines whether the maximum allowed fractional normalization delta (set by the ``max_normalization_delta`` parameter) is constrained to vary linearly with distance from the phase center. If True, the maximum delta is zero at the phase center and reaches the value set by ``max_normalization_delta`` for the most distant calibration patch. If False, the maximum delta is the same for all calibration patches.

    do_image
        Boolean flag that determines whether the imaging step should be done for this cycle.

    auto_mask
        Float that sets WSClean's automask value for this cycle.

    threshisl
        Float that sets PyBDSF's threshisl value for this cycle.

    threshpix
        Float that sets PyBDSF's threshpix value for this cycle.

    max_nmiter
        Integer that sets the maximum number of major iterations done during imaging for this cycle.

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

