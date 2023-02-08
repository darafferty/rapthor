.. _rapthor_strategy:

Defining a custom processing strategy
=====================================

A custom processing strategy can be supplied to Rapthor by specifying the full path
to a strategy file in the :term:`strategy` entry of the Rapthor parset. The strategy file
is a Python file with the following structure:

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

The following processing parameters my be set:

.. glossary::

    do_calibrate
        Boolean flag that determines whether the calibration step should be done for this cycle.

    do_slowgain_solve
        Boolean flag that determines whether the slow-gain part of calibration should be done for this cycle.

    peel_outliers
        Boolean flag that determines whether the outlier sources (sources that lie outside of any imaging sector) should be peeled for this cycle.

    peel_bright_sources
        Boolean flag that determines whether the bright sources should be peeled for this cycle.

    do_image
        Boolean flag that determines whether the imaging step should be done for this cycle.

    auto_mask
        Float that sets WSClean's automask value for this cycle.

    threshisl
        Float that sets PyBDFS's threshisl value for this cycle.

    threshpix
        Float that sets PyBDFS's threshpix value for this cycle.

    max_nmiter
        Integer that sets the maximum number of major iterations done during imaging for this cycle

    target_flux
        Float (or ``None``) that sets the target flux density for DDE calibrators for this cycle. If ``None``, a value must be specified for ``max_directions``.

    max_directions
        Integer (or ``None``) that sets the maximum number of directions (DDE calibrators) used during calibration for this cycle. If ``None``, a value must be specified for ``target_flux``.

    regroup_model
        Boolean flag that determines whether the sky model should be regrouped for this cycle.

    do_check
        Boolean flag that determines whether the check for self-calibration convergence should be done for this cycle.

    convergence_ratio
        Float that sets the minimum ratio of the current image noise to the previous image noise above which selfcal is considered to have converged (must be in the range 0.5 -- 2).

    divergence_ratio
        Float that sets the minimum ratio of the current image noise to the previous image noise above which selfcal is considered to have diverged (must be > 1).

