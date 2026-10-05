.. _migrating_from_cwl:

Migrating from CWL
==================

This page assumes you know how to run Rapthor v2.x.x and provides instructions 
for migrating from CWL-based Rapthor 2.x.x to Prefect/Dask-based Rapthor 3.0.0.


How To Run Rapthor
------------------

Launching a Rapthor 3.0.0 run remains unchanged from v2.x.x (see :ref:`running` for details)
but with v3.0.0 you can use the Prefect UI and Dask dashboard for monitoring and managing the run.
By default, Rapthor will use a temporary Prefect server and a local Dask cluster. You can 
also start your own Dask cluster and provide the address to Rapthor via the parset configuration option ``dask_scheduler``
and start your own persistent Prefect server and provide its API URL to Rapthor via the parset configuration option ``prefect_api_url``.

Parset Adaptations
------------------

There are some new features and breaking changes to the parset and strategy
configuration options in Rapthor 3.0.0. All parset options are documented in 
:ref:`rapthor_parset`.


Scratch directories
~~~~~~~~~~~~~~~~~~~

The ``local_scratch_dir`` and ``global_scratch_dir`` options retain their CWL
roles: local scratch holds command temporaries and WSClean reordering files;
shared scratch holds intermediates used by multiple workflow tasks.

.. code-block:: ini

    [cluster]
    local_scratch_dir = /local/scratch
    global_scratch_dir = /shared/scratch

Rapthor creates isolated child directories for workflows and commands.
``local_scratch_dir`` is created on the executing worker and need not exist
on the head node. The deprecated ``dir_local`` option has been removed; replace
it with ``local_scratch_dir`` in existing parsets. MPI commands always use shared
temporary storage: ``global_scratch_dir`` when configured, otherwise the shared
operation workspace. Local Dask workers also use local scratch for spill storage.
For external Dask workers, configure spill storage with the worker launcher's
``--local-directory`` option.

With global scratch, ``dir_working/pipelines/<operation>`` temporarily links to
the scratch workspace, preserving all recorded paths. Workflow products are
promoted back when the flow exits, including after command failure. A sibling
``.<operation>.scratch`` recovery record allows restart to repair interrupted
directory moves before operation setup, including when the operation link is
missing. Failed promotion preserves the scratch products and recovery record;
successful promotion removes that record. Internal links remain valid after
promotion. Restart records, final outputs and logs retain their paths under
``dir_working``.

When scratch options are unset, shared intermediates and WSClean imaging's
temporary directory use the operation workspace. Other commands inherit their
temporary environment. PyBDSF helper commands continue to use ``/tmp`` to keep
multiprocessing socket paths short, matching the CWL behavior. Enable
``keep_temporary_files`` or ``debug_workflow`` to retain command temporaries
for inspection.

Successful workflows also discard task-only products, matching CWL output
staging: channel FITS, image concatenation and prediction model-data Measurement
Sets, per-chunk solve h5parms, calibration prediction inputs, and mosaic assembly
files. Declared workflow products remain available for downstream operations
and restarts; operation finalizers still apply the configured save options.
Failed workflows preserve intermediate products for recovery. The keep/debug
settings preserve these workflow intermediates as well as command temporaries.
Command scratch removal failures produce a warning identifying the remaining
directory; they do not hide the original command error.


Use ``calibration_strategy``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Calibration solve type and order are controlled by ``calibration_strategy``
in the strategy file. Use explicit strategies such as:

.. code-block:: python

    strategy_steps[i]["calibration_strategy"] = {
        "dd": ["fast_phase", "medium_phase"],
        "di": ["full_jones"],
    }

Allowed solve names are ``fast_phase``, ``medium_phase``, ``slow_gains`` and 
``full_jones``.

The legacy ``do_slowgain_solve`` and ``do_fulljones_solve`` options are
deprecated but still work: Rapthor translates them into the equivalent
``calibration_strategy`` and logs a warning naming the replacement, so an
unmodified strategy file runs the same solves here as it does under CWL.
Setting a legacy option and ``calibration_strategy`` in the same cycle is an
error, since the requested solves would be ambiguous.


Run identity and diagnostics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use tags to make test runs easy to identify in the Prefect UI:

.. code-block:: ini

    [cluster]
    prefect_run_tags = tag1, tag2, tag3

Image previews and profiling options are useful during manual testing but may add
runtime and disk usage:

.. code-block:: ini

    [cluster]
    prefect_command_profile = time
    prefect_publish_fits_previews = True
    prefect_publish_postage_stamp_previews = True


Remote Dashboards
-----------------

When Rapthor runs on a remote compute node, note its hostname and
forward the Prefect and Dask dashboard ports through the cluster login node:

.. code-block:: console

    $ hostname

.. code-block:: console

    $ ssh -N \
        -L 127.0.0.1:4200:compute-node:4200 \
        -L 127.0.0.1:8787:compute-node:8787 \
        user@login.cluster.example

Open ``http://127.0.0.1:4200`` for Prefect and
``http://127.0.0.1:8787/status`` for Dask. Replace the host names and ports
with the values used by the run. If Rapthor runs directly on the login or
interactive host, use ``127.0.0.1`` as both tunnel destinations.
