.. _upgrading:

Upgrading from version 2
========================

Version 3 does the same processing as version 2, but runs it in a different
way. In version 2, each operation was a CWL workflow that was run by Toil. In
version 3, each operation is run by Rapthor itself, using Prefect to keep
track of the work and Dask to run it (see :ref:`architecture`).

The following are unchanged:

* the command used to start, resume and reset a run (see :ref:`running`);
* the format of the parset and of the strategy file;
* the layout of the working directory and the names of the images, solutions
  and sky models (see :ref:`products`).

We recommend that you do not resume a run that was started with version 2:
either let it finish with version 2, or start a new run with version 3 in a
new working directory.

This page lists what to check in an existing parset or strategy file.


Changes to the parset
---------------------

The following options of the ``[cluster]`` section have been removed. If one
of them is present, Rapthor logs a warning and ignores it:

``cwl_runner``
    No longer needed. Remove it.

``dir_local``
    Replaced by :term:`local_scratch_dir`.

The following options have changed:

:term:`use_container`
    No longer supported. Rapthor cannot run its operations inside a container
    while running itself outside one, and a run in which this option is
    ``True`` stops with an error. Run Rapthor itself inside the container
    instead (see :ref:`using_containers`).

:term:`batch_system`
    With ``slurm`` or ``slurm_static``, Rapthor no longer submits Slurm jobs.
    Instead, it uses a Dask scheduler and workers that you start inside your
    own Slurm job (see :ref:`running_on_cluster`). The ``TOIL_SLURM_ARGS``
    environment variable has no effect.

:term:`local_scratch_dir` and :term:`global_scratch_dir`
    Used as before: the local scratch directory holds the temporary files of
    each command, and the global scratch directory holds intermediate files
    that are shared between the nodes.

:term:`debug_workflow`
    Now has the same effect as :term:`keep_temporary_files`.

A number of options have been added to the ``[cluster]`` section. None of them
needs to be set for a run on a single machine. The most useful are:

* :term:`local_dask_workers`, to run several tasks at the same time on a
  single machine;
* :term:`dask_scheduler`, to use a Dask cluster that you started yourself;
* :term:`prefect_api_url`, to keep the history of your runs and follow them in
  the Prefect dashboard;
* :term:`prefect_run_tags`, to label a run in the dashboard.


CPU and memory budgets
----------------------

CWL's ``runtime.cores`` no longer supplies command thread counts. Rapthor
resolves :term:`cpus_per_task` as a worker budget, then resolves
:term:`max_threads` and the DP3/WSClean overrides within it. Zero selects an
automatic value; explicit oversubscription fails early. Multiple local Dask
workers divide automatic CPU and memory budgets equally. :term:`max_cores`
only limits gridding groups. See :doc:`running` for the worker and MPI policy.


Changes to the strategy
-----------------------

The solves done during calibration are set with :term:`calibration_strategy`,
which lists the solves to do, in order, for the direction-dependent ("dd") and
direction-independent ("di") parts of the calibration:

.. code-block:: python

    strategy_steps[i]["calibration_strategy"] = {
        "dd": ["fast_phase", "medium_phase", "slow_gains", "medium_phase"],
        "di": ["full_jones"],
    }

The ``do_slowgain_solve`` and ``do_fulljones_solve`` parameters are deprecated.
A strategy file that uses them still works: Rapthor replaces them with the
equivalent ``calibration_strategy`` and logs a warning that names the
replacement. A cycle that sets both one of these parameters and
``calibration_strategy`` is an error, as it is then unclear which solves are
wanted.


.. _migration_behavior_differences:

Processing and output differences
---------------------------------

The following differences from the version 2 CWL workflows are intentional:

Calibration carried between cycles
    After a new calibration step, imaging uses the active cycle's solutions.
    In particular, a DD-only cycle does not silently apply a DI full-Jones
    solution from an earlier cycle. Explicit image-only cycles can reuse
    compatible previous solutions.

    Earlier solutions can still seed the same solve in the same calibration
    mode. DD seeds need not have the same direction names or count: DP3 uses
    the nearest h5parm direction for each new direction. Seeding the optimizer
    does not apply those solutions as corrections to the data.

    Related documentation: :ref:`calibration_strategy_details`.

Slow-gain amplitudes
    The combined ``field-solutions.h5`` retains the slow-gain amplitudes when
    those solves were requested. The old CWL combination path could report
    success while leaving only phase solutions in the combined product.

    Related documentation: :ref:`calibrate` lists the individual and combined
    solution products.

WSClean prediction channels
    With :term:`use_wsclean_predict`, prediction covers every requested channel,
    including the last channel of each frequency chunk. The ``-channel-range``
    end index is exclusive; the old path could omit that last channel.

    Related documentation: :term:`wsclean_predict_bw` controls prediction
    frequency grouping.

Output records
    File and directory entries in the operation's JSON output records contain
    ``class`` and ``path``. They omit the checksums and file-size metadata
    produced by CWL; scripts reading these records should use the path to
    inspect the product. This changes the metadata, not the scientific product
    itself.

    Related documentation: :ref:`products` lists record locations and restart
    files.

Task counts and timing
    Prefect exposes more processing steps as separate tasks and records their
    runtime metrics. Task counts therefore cannot be compared directly with
    CWL task counts. Compare elapsed time for the full run and its operations,
    using repeated runs to account for runtime variation.

    Related documentation: :ref:`architecture_tasks` lists the task boundaries;
    :ref:`monitoring_rapthor` describes logs and resource metrics.


Other changes
-------------

The ``plotrapthor`` command has been removed. Plots of the calibration
solutions are made during each cycle as before (see :ref:`calibrate`). To plot
a solution table yourself, use:

.. code-block:: console

    $ python -m rapthor.execution.calibrate.plotting_cli field-solutions.h5 phase


Following a run
---------------

The Toil job store and its log files are gone. In their place:

* the output of every command is written to
  ``dir_working/logs/<operation>/<task>.log``, and every command that was run
  is listed in ``dir_working/logs/commands.jsonl``;
* the progress of a run can be followed in the Prefect and Dask dashboards.

See :ref:`monitoring_rapthor` for details.
