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
