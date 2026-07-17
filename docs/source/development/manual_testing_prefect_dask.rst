.. _manual_testing_prefect_dask:

Manual Testing The Prefect/Dask Branch
======================================

This page is for developers and scientists testing the refactored Prefect/Dask
branch with real parsets. It assumes you know what your Rapthor run is meant to
do scientifically, but not the details of the refactor.

The goal is to make each test reproducible and useful for deciding whether this
branch should replace ``master``. Runtime setup and launch commands are kept in
one canonical place, :ref:`running`; this page covers only the additional work
needed to prepare, inspect, and report a manual branch test.

How To Run Rapthor
------------------

Choose the runtime mode that matches the test:

* :ref:`local_runtime` for the simplest ``rapthor input.parset`` run, with an
  isolated temporary Prefect API and local Dask.
* :ref:`persistent_prefect_dashboard` when the run should remain visible in a
  local Prefect dashboard.
* :ref:`concurrent_rapthor_jobs` for multiple jobs, including the SQLite
  limitations and the choice between a shared or isolated dashboard.
* :ref:`external_dask_runtime` when a Dask scheduler already exists.
* :ref:`slurm_external_dask` for production-like multi-node staging and
  WSClean MPI.

In every mode, the Rapthor interface remains:

.. code-block:: console

    $ rapthor my-prefect-dask.parset

Do not source ``scripts/dev/start-prefect-server.sh``. When using the optional
persistent dashboard, execute the script as documented in
:ref:`persistent_prefect_dashboard`; it owns a foreground server and its
cleanup traps.

Prepare A Test Run
------------------

Start with a copy of the original parset and a fresh working directory:

.. code-block:: console

    $ cp my-master.parset my-prefect-dask.parset
    $ mkdir -p runs/manual-my-target

Edit ``my-prefect-dask.parset`` so ``global.dir_working`` points at the new
directory. Do not reuse a ``master`` work directory: restart markers and
operation outputs differ, and a clean directory is easier to inspect.

Record the branch commit, input dataset, parset, strategy, install method, and
external-tool versions before starting. This makes environment-specific
failures reproducible.

Install Or Environment Choices
------------------------------

Use whichever supported environment is natural on the test system:

* **Dev container:** best for quick local checks in this repository. It
  installs Rapthor in editable mode with development and documentation
  dependencies.
* **Existing site environment:** suitable when DP3, WSClean,
  python-casacore, PyBDSF, LoSoTo, LSMTool, and the other astronomy
  dependencies are already available. Install this branch in editable mode if
  needed.
* **Spack/module environment:** preferred for reproducible production-like
  staging, multi-node Slurm, and comparisons with the legacy deployment, but
  not required for interactive testing.

Common Parset Adaptations
-------------------------

Use ``calibration_strategy``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Calibration solve type and order are controlled by ``calibration_strategy``
in the strategy file. Prefer explicit strategies such as:

.. code-block:: python

    strategy_steps[i]["calibration_strategy"] = {
        "dd": ["fast_phase", "medium_phase"],
        "di": ["full_jones"],
    }

Allowed solve names are ``fast_phase``, ``medium_phase``, ``slow_gains`` for
slow diagonal gain solves, and ``full_jones``. Do not rely on legacy toggles
such as ``do_fulljones_solve`` or ``do_slowgain_solve`` in new tests.

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

Run identity and diagnostics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use tags to make test runs easy to identify in Prefect:

.. code-block:: ini

    [cluster]
    prefect_run_tags = manual-test, my-target, current-branch

Preview and profiling options are useful during manual testing but may add
runtime and disk usage:

.. code-block:: ini

    [cluster]
    prefect_command_profile = time
    prefect_publish_fits_previews = False
    prefect_publish_postage_stamp_previews = True

The detailed behavior of command profiling and preview artifacts is documented
with the demo helper in :ref:`prefect_demo_helper`.

Remote Dashboards
-----------------

When Rapthor runs interactively on a remote compute node, note its hostname and
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

For Slurm, use the setup and network-reachability requirements in
:ref:`slurm_external_dask`. The Prefect API must be reachable from Dask worker
nodes; the SSH tunnel only exposes the dashboards to the local browser.
For the exact demo-helper launch and tunnel commands, see
:ref:`demo_compute_node_tunnel`.

Optional Bundled Demo
---------------------

The bundled helper is useful for learning the dashboards before testing real
data:

.. code-block:: console

    $ scripts/dev/run-rapthor-prefect-demo.py examples/prefect_demo.parset

See :ref:`prefect_demo_helper` for generated rich and multi-sector datasets,
resource overrides, Dask performance reports, and preview artifacts. The
helper is intended for demos and benchmark automation; developers testing
their own parsets should continue to launch Rapthor with
``rapthor input.parset``.

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
* sky models, catalogs, diagnostics, and plots relevant to the workflow
* Prefect flow/task names, task logs, artifacts, and retries
* Dask task stream and worker occupancy for performance-sensitive runs

What To Record
--------------

Use this copy/paste template so another developer can understand the outcome:

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

The highest-priority manual tests are everyday single-sector self-calibration
and image-only/applycal workflows. Multi-sector mosaic is useful to smoke-test
when a tester already has such a workflow, but it is rare on ``master`` and is
not a primary switch blocker.

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

Manual testing should update the switch decision with a short summary of
real-user outcomes and any accepted caveats.
