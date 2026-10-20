.. _rapthor_strategy:

Setting the processing strategy
===============================

The steps used during a Rapthor run are defined through the choice of a processing
strategy. The processing strategy is set with the :term:`strategy` parameter of the
Rapthor parset. The options for this parameter are described below:

``strategy = selfcal`` (the default)
    This strategy performs self calibration of the field in which the sky model is
    iteratively improved through calibration and imaging. The processing generally involves
    up to two cycles of phase-only calibration followed by up to 6 cycles of phase and
    amplitude calibration until convergence is obtained.

    .. note::

        If an initial sky model is automatically generated from the data (see
        :ref:`auto_sky_generation`), the phase-only cycles are not generally
        needed and are therefore skipped. If a sky model is downloaded from a
        survey catalog or supplied directly by the user, the phase-only cycles
        are not skipped. In these cases, a custom strategy can disable them
        when they are not needed (see :ref:`custom_strategy`).

        The built-in strategy sets :term:`calibration_strategy` explicitly for
        every cycle. Its early phase-only sequence is
        ``["fast_phase", "medium_phase"]`` under ``"dd"``; later cycles add
        ``"slow_gains"`` and a second ``"medium_phase"``. It does not rely on the
        fallback used when a custom cycle omits the parameter.

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
    <https://git.astron.nl/RD/rapthor/-/blob/main/examples/custom_calibration_strategy.py>`_
    (for self calibration), `here
    <https://git.astron.nl/RD/rapthor/-/blob/main/examples/custom_imaging_strategy.py>`_
    (for imaging only) and `here
    <https://git.astron.nl/RD/rapthor/-/blob/main/examples/flexible_calibration_strategy.py>`_
    (for more control over the calibration strategy). Templates based on the
    built-in strategies are available `here
    <https://git.astron.nl/RD/rapthor/-/blob/main/examples/default_calibration_strategy.py>`_
    (for self calibration; unlike the built-in self calibration strategy, this
    file never runs phase-only cycles) and `here
    <https://git.astron.nl/RD/rapthor/-/blob/main/examples/default_imaging_strategy.py>`_
    (for imaging only).

.. note::

    If the strategy performs self calibration, the last entry in ``strategy_steps`` can be
    used to specify parameters specific to the final cycle (which is performed after
    self calibration finishes). In the default self calibration strategy, the parameters
    for the final cycle are set to those of the last cycle of selfcal, except for the
    output channel width, which is reduced from 8 MHz to 4 MHz for the final cycle.

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
        Float that sets the solution interval in sec to use in the fast
        (scalarphase) solve. For DD calibration, the core stations are constrained
        to have the same solutions. DI solves do not use this core constraint.

    medium_timestep_sec
        Float that sets the solution interval in sec to use in the medium-fast
        (scalarphase) solves. In DD calibration, each station is solved for
        independently before ``slow_gains``. A ``medium_phase`` solve requested
        after ``slow_gains`` constrains the core stations to have the same
        solutions, even if it is the first ``medium_phase`` in the sequence.
        DI solves do not use this core constraint.

    slow_timestep_sec
        Float that sets the solution interval in sec to use in the slow-gain solve. For this solve, each station is solved for independently.

    fulljones_timestep_sec
        Float that sets the solution interval in sec to use in the full-Jones solve. For this solve, each station is solved for independently.

    peel_outliers
        Boolean flag that determines whether the outlier sources (sources that lie outside of any imaging sector region) should be peeled for this cycle. Outliers can only be peeled once (unlike bright sources, see below), as they are not added back for subsequent selfcal cycles. Note that, because they are not imaged, outlier source models do not change during self calibration: however, the solutions they receive may change. To include one or more outlier sources in self calibration, a small imaging sector can be placed on each outlier of interest. The outliers will then be imaged and their models updated with the rest of the field.

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

    calibration_strategy
        Dictionary that sets the sequence of solves for this cycle. The keys of the dictionary are "di" and "dd", which set the direction-independent and direction-dependent calibration strategies, respectively. The value for each key is a list of solve names to run for that calibration mode. For example:

        .. code-block:: python

            strategy_steps[i]["calibration_strategy"] = {
                "dd": ["fast_phase", "medium_phase", "slow_gains", "medium_phase"],
                "di": [],
            }

        Allowed solve names are:

        - "fast_phase": run the fast (scalarphase) solve.
        - "medium_phase": run the medium-fast (scalarphase) solve.
        - "slow_gains": run the slow (diagonal) solve.
        - "full_jones": run the full-Jones solve.

        The order of the top-level keys (DI and DD) and the order of solve names
        within each list determine the order requested by the user. An empty
        list means that calibration mode is skipped, allowing DI-only or DD-only
        cycles.

        An omitted parameter, ``None`` and an empty dictionary all count as
        unspecified. If either legacy solve flag is present in that cycle,
        Rapthor first translates the flags into an explicit strategy; see
        :ref:`legacy_calibration_flags`. This also applies to flags set to
        ``False``.

        If neither a strategy nor legacy flags are supplied, a calibrating cycle
        logs a warning and uses the sequence above (the full DD sequence and no
        DI solves). Each calibrating cycle resolves this default independently;
        an earlier cycle's sequence does not carry over. Preflight checks use
        the same resolved sequence. Each calibrating cycle logs the sequence in
        its requested order and identifies whether it was defaulted.

        A cycle without calibration preserves the previous sequence when neither
        a strategy nor legacy flags are supplied, so that imaging can apply the
        previous solutions. An explicit strategy, including one translated from
        legacy flags, overrides that sequence. If there is no previous sequence,
        the full DD sequence above is used to select the supplied solutions to
        apply; no calibration is run.

        A dictionary with empty solve lists, such as ``{"di": [], "dd": []}``,
        explicitly disables those modes; it does not request the default. See
        :ref:`calibration_strategy_details` for supported solve combinations and
        how the resulting solutions are applied.


.. _calibration_strategy_details:

Setting the calibration strategy
--------------------------------

The :term:`calibration_strategy` parameter sets, for each cycle, which solves are done and in what order. Not every sequence of solves is possible. The sequences that can be used are:

For the direction-dependent ("dd") calibration:

- ``["fast_phase"]``
- ``["fast_phase", "medium_phase"]``
- ``["fast_phase", "medium_phase", "slow_gains"]``
- ``["fast_phase", "medium_phase", "slow_gains", "medium_phase"]``
- ``["slow_gains"]``
- ``["slow_gains", "medium_phase"]``

For the direction-independent ("di") calibration:

- ``["fast_phase"]``
- ``["fast_phase", "medium_phase"]``
- ``["fast_phase", "medium_phase", "slow_gains"]``
- ``["slow_gains"]``
- ``["full_jones"]``

An empty list is also allowed for either mode and disables its solves. Rapthor
stops with an error at the start of the run if the strategy contains any other
nonempty sequence. A full-Jones solve can only be done as a direction-independent
solve, on its own. To do a full-Jones solve in addition to other
direction-independent solves, do them in different cycles.

The order of the "dd" and "di" keys sets which of the two is done first:

- A direction-dependent calibration is done by the calibrate operation (named ``calibrate_X`` in the working directory, where ``X`` is the cycle number).
- A direction-independent calibration is done in two parts: the model visibilities are first predicted (``predict_di_X``) and the solve is then done against them (``calibrate_di_X``).
- If the direction-independent calibration is done first, its solutions are applied to the data before the direction-dependent solves are done.
- If the direction-dependent calibration is done first, its solutions are used when the model visibilities for the direction-independent solve are predicted.

The solutions are then used in the imaging of the same cycle as follows:

- Direction-independent solutions (including full-Jones solutions) are applied to the visibilities before imaging.
- Direction-dependent solutions are applied by WSClean during imaging, facet by facet. If :term:`dde_method` = ``single``, the solutions of the direction closest to the center of each sector are applied to the visibilities before imaging instead.

The solutions found in one cycle are not applied to the data in later cycles in which calibration is done again. They are used only as the starting values for the same solves in the next cycle, which speeds up the convergence of the solver.

.. _legacy_calibration_flags:

Legacy solve flags
~~~~~~~~~~~~~~~~~~

The ``do_slowgain_solve`` and ``do_fulljones_solve`` parameters are deprecated.
If at least one is present and :term:`calibration_strategy` is unspecified,
Rapthor translates the flags into an explicit strategy when loading the
processing strategy. It logs a deprecation warning naming the replacement.
Any missing companion flag is treated as ``False`` for that cycle.

The table below applies when ``calibration_strategy`` is absent, ``None`` or
``{}``. "Phase-only DD" means ``["fast_phase", "medium_phase"]``. "Full DD"
means ``["fast_phase", "medium_phase", "slow_gains", "medium_phase"]``.

.. list-table:: Solve selection for a calibrating cycle
    :header-rows: 1
    :widths: 25 25 25 25

    * - ``do_slowgain_solve``
      - ``do_fulljones_solve``
      - DD solves
      - DI solves
    * - Absent
      - Absent
      - Full DD (default)
      - None
    * - ``False``
      - ``False``
      - Phase-only DD
      - None
    * - ``True``
      - ``False``
      - Full DD
      - None
    * - ``False``
      - ``True``
      - Phase-only DD
      - ``["full_jones"]``
    * - ``True``
      - ``True``
      - Full DD
      - ``["full_jones"]``

Explicitly setting either flag to ``False`` therefore differs from omitting
both flags. For example, setting only ``do_fulljones_solve = False`` requests
phase-only DD calibration. Translated strategies always put DD before DI.

A nonempty ``calibration_strategy`` dictionary cannot be combined with either
legacy flag, even when the flag is ``False`` or the dictionary contains only
empty solve lists. Rapthor rejects that combination as ambiguous. Use the
explicit dictionary and remove the legacy flags.

Legacy flags are also translated in image-only cycles. They override the
previous strategy used to select solutions for imaging, but do not enable
calibration when :term:`do_calibrate` is ``False``.
