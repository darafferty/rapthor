.. _manual_testing_prefect_dask:

Manual Testing The Prefect/Dask Branch
======================================

This page is for developers and scientists testing the refactored Prefect/Dask
branch with real parsets. It assumes you know what your Rapthor run is meant to
do scientifically, but not the details of the refactor.

The goal is to make testing quick, reproducible, and useful for the decision on
whether this branch should replace ``master``.

Quick Start
-----------

Start Prefect in one terminal:

.. code-block:: console

    $ scripts/dev/start-prefect-server.sh

The script only starts Prefect. It creates a fresh server database in system
temporary storage, disables analytics, configures a tunnel-friendly UI API
path, waits for the API health check, and prints the connection details. It
does not start or otherwise wrap Rapthor. Execute the script as shown; do not
``source`` it, because it manages a foreground server process and its cleanup
traps.

Keep the server terminal open. In a second terminal on the same host, point
Rapthor at the server and use the normal interface:

.. code-block:: console

    $ export PREFECT_API_URL=http://127.0.0.1:4200/api
    $ rapthor rapthor.parset

Open ``http://127.0.0.1:4200`` locally, or use the tunnel below when Rapthor is
running remotely. Stop the server with ``Ctrl+C`` in its terminal when testing
is complete. The setup script then stops the server and removes the temporary
state it created. This is the recommended tester workflow: Rapthor is started
exactly as it would be without a dashboard.

To use a different dashboard port, pass it to the setup script and use the
same port in ``PREFECT_API_URL``:

.. code-block:: console

    $ scripts/dev/start-prefect-server.sh --port 14200

.. code-block:: console

    $ export PREFECT_API_URL=http://127.0.0.1:14200/api
    $ rapthor rapthor.parset

Start each test with a copy of the parset and a fresh working directory:

.. code-block:: console

    $ cp my-master.parset my-prefect-dask.parset
    $ mkdir -p runs/manual-my-target

Edit ``my-prefect-dask.parset`` so ``global.dir_working`` points at the new
directory, then pass that parset to the normal ``rapthor`` command above.

When running interactively on a remote compute node, first note its hostname:

.. code-block:: console

    $ hostname

Then forward both dashboard ports through the cluster login node from your
laptop, following the pattern already proven by the Prefect prototype:

.. code-block:: console

    $ ssh -N \
        -L 127.0.0.1:4200:compute-node:4200 \
        -L 127.0.0.1:8787:compute-node:8787 \
        user@login.cluster.example

Open ``http://127.0.0.1:4200`` for Prefect and
``http://127.0.0.1:8787/status`` for Dask. Replace the host names with those
used by the site. If Rapthor runs directly on the login or interactive host,
use ``127.0.0.1`` as both tunnel destinations.

For a run without a persistent dashboard, leave ``PREFECT_API_URL`` unset and
run Rapthor directly:

.. code-block:: console

    $ rapthor my-prefect-dask.parset

This is the lowest-friction mode. If no Prefect API URL or Dask scheduler is
configured, Rapthor uses a temporary Prefect API/server and a local Dask
scheduler for the run.

Multiple Concurrent Jobs And Dashboards
---------------------------------------

Testers can run more than one Rapthor job at a time. There are two useful
dashboard arrangements:

* **One shared dashboard:** start one Prefect server, then export the same
  ``PREFECT_API_URL`` in every Rapthor terminal. Each job appears as a separate
  pipeline flow run with its own flow-run page, logs, subflows, tasks, and
  artifacts. There is one dashboard URL for all the jobs, rather than a
  separate dashboard server for each job. Open the individual flow-run pages
  in separate browser tabs when monitoring the jobs side by side.
* **One isolated dashboard per job:** start a Prefect server for each job on a
  different port. Each setup-script process creates independent temporary
  Prefect state, so every job has its own dashboard URL and SQLite database.

The shared arrangement is convenient for a small number of interactive test
runs. It is not the production recommendation because the local Prefect server
uses SQLite. Use isolated servers for stronger test isolation, or the no-server
mode described below for concurrent production jobs until a Postgres-backed
Prefect service is available.

For example, start two isolated dashboard servers in two terminals:

.. code-block:: console

    $ scripts/dev/start-prefect-server.sh --port 14200

.. code-block:: console

    $ scripts/dev/start-prefect-server.sh --port 14201

Then start each job from its own terminal with the matching API URL:

.. code-block:: console

    $ export PREFECT_API_URL=http://127.0.0.1:14200/api
    $ rapthor job-a.parset

.. code-block:: console

    $ export PREFECT_API_URL=http://127.0.0.1:14201/api
    $ rapthor job-b.parset

Open ``http://127.0.0.1:14200`` for job A and
``http://127.0.0.1:14201`` for job B. Keep both server terminals open until
the jobs and dashboard inspection are complete. Leave ``PREFECT_HOME`` unset
to let each script create unique temporary state. If you set ``PREFECT_HOME``
explicitly, use a different directory for every server.

Every concurrent job must have a unique working directory. When using local
Dask, it must also have a unique Dask dashboard port. Run tags make jobs easier
to identify, especially when they share one Prefect dashboard. For example,
job A can use:

.. code-block:: ini

    [global]
    dir_working = /path/to/runs/job-a

    [cluster]
    prefect_run_tags = manual-test, job-a
    dask_dashboard_address = :18787

and job B can use:

.. code-block:: ini

    [global]
    dir_working = /path/to/runs/job-b

    [cluster]
    prefect_run_tags = manual-test, job-b
    dask_dashboard_address = :18788

If the jobs run remotely, forward each selected Prefect and Dask port through
SSH. For example, add ``14200``, ``14201``, ``18787``, and ``18788`` as
separate local forwards. A port can be omitted when its dashboard is not
needed.

Install Or Environment Choices
------------------------------

Use whichever environment is natural on the system you are testing:

* **Dev container:** best for quick local checks in this repository. The dev
  container installs Rapthor in editable mode with development and documentation
  dependencies.
* **Existing site environment:** fine for interactive testing when DP3,
  WSClean, python-casacore, PyBDSF, LoSoTo, LSMTool, and the other astronomy
  dependencies are already available. Install this branch in editable mode
  inside that environment if needed.
* **Spack/module environment:** preferred for production-like staging,
  multi-node Slurm, and comparisons with the legacy deployment, but not
  required for every manual tester.

Before running a real parset, record the install method and the key tool
versions so failures can be reproduced.

Production Parallel Runs Without A Prefect Server
-------------------------------------------------

Until a shared Prefect service is deployed with a Postgres backend, production
parallelism should use independent Rapthor processes rather than many jobs
writing to one local SQLite-backed Prefect server.

For each job:

* leave ``PREFECT_API_URL`` unset, or set ``prefect_api_mode = ephemeral``
* use a unique ``global.dir_working``
* use unique run tags if you later collect logs and reports together
* choose local Dask or an existing external Dask scheduler according to the
  site resource policy

In no-server mode, Rapthor clears any inherited Prefect profile API URL and
uses an isolated temporary Prefect home for that process. That keeps concurrent
jobs from sharing Prefect state. The trade-off is that Prefect history and the
Prefect dashboard are temporary; use logs, command records, task timing files,
and saved reports for production provenance.

When a persistent dashboard and long-lived run history become production
requirements, use an explicitly configured Prefect API backed by Postgres
rather than the local development server.

Add run tags to make manual tests easy to find in the dashboard:

.. code-block:: ini

    [cluster]
    prefect_run_tags = manual-test, my-target, current-branch

Optional Bundled Demo Helper
----------------------------

``scripts/dev/run-rapthor-prefect-demo.py`` remains available for repeatable
bundled demos and benchmark automation. It starts and stops the supporting
services itself, but it is not the primary workflow for developers testing
their own parsets. To exercise the small repository fixture with it:

.. code-block:: console

    $ scripts/dev/run-rapthor-prefect-demo.py examples/prefect_demo.parset

Dask Choices
------------

For a single machine, local Dask is the normal choice:

.. code-block:: ini

    [cluster]
    prefect_task_runner = local_dask
    local_dask_workers = 4
    cpus_per_task = 15
    max_threads = 15
    dask_dashboard_address = :8787

For an existing scheduler:

.. code-block:: ini

    [cluster]
    prefect_task_runner = external_dask
    dask_scheduler = tcp://127.0.0.1:8786

The Dask dashboard helps diagnose worker occupancy, task concurrency, and
whether external tools are oversubscribed. The Prefect dashboard is better for
operation/task status, logs, artifacts, and restart behavior.

Slurm, External Dask, And WSClean MPI
-------------------------------------

Multi-node production validation should run inside one Slurm allocation with:

* one Dask scheduler started on the first allocated node
* one Dask worker per allocated node
* ``DASK_SCHEDULER`` exported before ``rapthor`` starts
* ``prefect_task_runner = external_dask``
* ``batch_system = slurm``
* ``imaging.use_mpi = True`` for the MPI WSClean imaging path

The staging script should check that all expected workers are connected before
launching Rapthor, and should record the Dask scheduler/dashboard address,
worker count, Slurm node list, WSClean version, DP3 version, and the exact
parset used.

If a Prefect dashboard is used for a multi-node run, the configured
``PREFECT_API_URL`` must be reachable from the Dask worker nodes, not only from
the login node or the process that launched Rapthor. Until a Postgres-backed
Prefect service is available, treat this as staging validation rather than the
default production mode.

For a demonstration, the Slurm launcher should write the dashboard endpoints
and SSH tunnel commands into the run output directory. The intended local view
is:

* Prefect dashboard in a local browser, for example
  ``http://127.0.0.1:14200``
* Dask dashboard in a local browser, for example
  ``http://127.0.0.1:18787/status``

The tunnel normally goes through the login/head node to the host running the
Prefect server and the first compute node running the Dask scheduler. The exact
host names and ports are site-specific, so record the generated tunnel command
with the manual-test outcome.

Spack Or Module-Based Staging
-----------------------------

For production-like manual tests, prefer a loadable Spack environment once the
Prefect/Dask branch recipe exists. This is not required for all interactive
testers. The intended package is a new recipe in ``../ska-sdp-spack/packages/``
rather than the existing ``py-rapthor`` recipe, so testers can compare the
legacy and refactored branches side by side.

The recipe should provide:

* the ``rapthor`` command from this branch
* Prefect, Prefect-Dask, Prefect-Shell, and Dask distributed
* Python astronomy dependencies such as python-casacore, PyBDSF, LoSoTo, and
  LSMTool
* external tools such as DP3, WSClean including ``wsclean-mp``, ``fpack``, and
  AOFlagger

Before using the module for a real run, record:

.. code-block:: console

    $ spack find --loaded
    $ rapthor --help
    $ python -c "import rapthor, prefect, prefect_dask, dask.distributed"
    $ DP3 --version
    $ wsclean --version
    $ wsclean-mp --version

Common Parset Adaptations
-------------------------

Use a new working directory
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Do not point a manual test at an old ``master`` work directory. The restart
markers and operation outputs are intentionally different enough that a clean
directory is safer and easier to inspect.

Use ``calibration_strategy``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Calibration solve type and order are now controlled by
``calibration_strategy`` in the strategy file. Prefer explicit strategies such
as:

.. code-block:: python

    strategy_steps[i]["calibration_strategy"] = {
        "dd": ["fast_phase", "medium_phase"],
        "di": ["full_jones"],
    }

Allowed solve names are:

* ``fast_phase``
* ``medium_phase``
* ``slow_gains`` for slow diagonal gain solves
* ``full_jones``

Avoid relying on legacy toggles such as ``do_fulljones_solve`` or
``do_slowgain_solve`` when preparing new tests. The current branch is intended
to make solve order explicit.

Image-only and existing solutions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For image-only strategies:

* DI scalar phase, DI diagonal slow-gain, and DI full-Jones products are
  pre-applied during imaging preparation.
* DD products are applied on the fly during imaging when matching directions
  are available.
* ``input_h5parm`` may be used for scalar phase or diagonal slow-gain
  solutions.
* ``input_fulljones_h5parm`` should be used for DI full-Jones solutions.
* DD use needs matching sky-model patches or a compatible facet layout.

Preview artifacts
~~~~~~~~~~~~~~~~~

FITS previews are useful for demos but can add runtime and disk usage:

.. code-block:: ini

    [cluster]
    prefect_publish_fits_previews = False
    prefect_publish_postage_stamp_previews = True

Postage-stamp previews are lighter and are saved under the working directory as
well as published as Prefect artifacts.

What To Inspect
---------------

For each manual run, check:

* the final command return code
* ``dir_working/logs/rapthor.log``
* ``dir_working/logs/commands.jsonl``
* ``dir_working/pipelines/<operation>/pipeline_inputs.json``
* ``dir_working/pipelines/<operation>/.outputs.json``
* FITS images under ``dir_working/images/``
* h5parm solutions under ``dir_working/solutions/``
* sky models, catalogs, diagnostics, and plots relevant to your workflow
* Prefect flow/task names, task logs, artifacts, and retries
* Dask task stream and worker occupancy for performance-sensitive runs

What To Record
--------------

Please record enough information for another developer to understand the
outcome:

.. code-block:: text

    Tester:
    Branch/commit:
    Date:
    Dataset/target:
    Parset:
    Strategy:
    Run tags:
    Runtime mode: ephemeral/local_dask, persistent Prefect, external Dask, Slurm
    Install/environment: dev container / editable install / site module / Spack
    Required parset changes:
    Return code:
    Working directory:
    Dashboard observations:
    Output sanity check:
    Performance observations:
    Problems or surprises:
    Switch recommendation: yes / yes with caveats / no

Switch-Decision Focus
---------------------

The most important manual tests are everyday single-sector self-calibration and
image-only/applycal workflows. Multi-sector mosaic is useful to smoke-test when
you already have such a workflow, but it is rare on ``master`` and is not a
primary switch blocker.

Treat a failure as important when it affects:

* default-like single-sector self-calibration
* phase-only or mixed DD/DI calibration
* image-only/applycal behavior
* custom ``calibration_strategy`` behavior
* restart/reset behavior
* output products that downstream users depend on
* the ability to understand failures from logs and dashboards

Where To Find The Evidence
--------------------------

The current evidence-driven switch summary is ``EQUIVALENCE_REPORT.md``.
Detailed archived reports live under:

* ``docs/source/development/science_equivalence_runs/``
* ``docs/source/development/performance_equivalence_runs/``
* ``docs/source/development/benchmark_baselines/``

Manual testing should update the switch decision by adding a short summary of
real-user outcomes and any accepted caveats.
