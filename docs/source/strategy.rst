.. _rapthor_strategy:

Defining a custom processing strategy
=====================================

The default processing strategy is designed to perform full self calibration of a
typical LOFAR dataset (see the file ``rapthor/examples/default_calibration_strategy.py``
in the Rapthor source tree for details). However, depending on the field and the aims of
the reduction, the default strategy may not be optimal. A custom processing strategy can
be supplied by specifying the full path to a strategy file in the :term:`strategy` entry
of the Rapthor parset. The strategy file is a Python file with the following structure:

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
dictionary stores the processing parameters for that cycle. For an example of a
custom strategy, see the file ``rapthor/examples/custom_calibration_strategy.py``
in the Rapthor source tree.

The following processing parameters must be set for each cycle:

.. glossary::

    do_calibrate
        Boolean flag that determines whether the calibration step should be done for this cycle.

    do_slowgain_solve
        Boolean flag that determines whether the slow-gain part of calibration should be done for this cycle.

    peel_outliers
        Boolean flag that determines whether the outlier sources (sources that lie outside of any imaging sector) should be peeled for this cycle. Outliers can only be peeled once (unlike bright sources, see below), as they are not added back for subsequent selfcal cycles.

    peel_bright_sources
        Boolean flag that determines whether the bright sources should be peeled for this cycle (for imaging only). The peeled bright sources are added back before subsequent selfcal cycles are performed (so they are included in the calibration, etc.).

    do_image
        Boolean flag that determines whether the imaging step should be done for this cycle.

    do_multiscale_clean
        Boolean flag that determines whether multiscale clean should be used for this cycle.

    auto_mask
        Float that sets WSClean's automask value for this cycle.

    threshisl
        Float that sets PyBDFS's threshisl value for this cycle.

    threshpix
        Float that sets PyBDFS's threshpix value for this cycle.

    max_nmiter
        Integer that sets the maximum number of major iterations done during imaging for this cycle.

    target_flux
        Float (or ``None``) that sets the target flux density for DDE calibrators for this cycle. If ``None``, a value must be specified for ``max_directions``.

    max_directions
        Integer (or ``None``) that sets the maximum number of directions (DDE calibrators) used during calibration for this cycle. If ``None``, a value must be specified for ``target_flux``. If both ``max_directions`` and ``target_flux`` are specified, the specified target flux density is used unless it would result in more than the specified maximum number of directions, in which case the target flux density is increased to ensure that the maximum number of directions is not exceeded.

    regroup_model
        Boolean flag that determines whether the sky model should be regrouped for this cycle.

    do_check
        Boolean flag that determines whether the check for self-calibration convergence should be done for this cycle.

    convergence_ratio
        Float that sets the minimum ratio of the current image noise to the previous image noise above which selfcal is considered to have converged (must be in the range 0.5 -- 2). A check is also done for the image dynamic range, where the ratio of the current to previous value must be below 1 / ``convergence_ratio``. Selfcal is considered to have converged only if both of these conditions are met.

    divergence_ratio
        Float that sets the minimum ratio of the current image noise to the previous image noise above which selfcal is considered to have diverged (must be > 1).

