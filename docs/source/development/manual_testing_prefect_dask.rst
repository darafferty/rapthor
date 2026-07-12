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

Start with a copy of your parset and a fresh working directory:

.. code-block:: console

    $ cp my-master.parset my-prefect-dask.parset
    $ mkdir -p runs/manual-my-target

Edit ``my-prefect-dask.parset`` so ``global.dir_working`` points at a new
directory. Then run:

.. code-block:: console

    $ rapthor my-prefect-dask.parset

This is the lowest-friction mode. If no Prefect API URL or Dask scheduler is
configured, Rapthor uses a temporary Prefect API/server and a local Dask
scheduler for the run.

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

Persistent Dashboard Run
------------------------

Use this when you want to inspect flows, tasks, logs, and artifacts in the
Prefect dashboard:

.. code-block:: console

    $ PREFECT_SERVER_ANALYTICS_ENABLED=false \
      prefect server start --host 0.0.0.0 --port 4200

In another terminal:

.. code-block:: console

    $ export PREFECT_API_URL=http://127.0.0.1:4200/api
    $ rapthor my-prefect-dask.parset

The dashboard is available at ``http://127.0.0.1:4200``. Add run tags to make
manual tests easy to find:

.. code-block:: ini

    [cluster]
    prefect_run_tags = manual-test, my-target, current-branch

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
------------------------------------

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
