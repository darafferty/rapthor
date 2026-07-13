ADR: Replace CWL/Toil With Python/Prefect/Dask
================================================

Status
------

Proposed.

Date
----

2026-07-13.

Decision
--------

Rapthor should replace the generated CWL/Toil production execution path with
typed Python orchestration built on Prefect and Dask.

The user-facing command remains:

.. code-block:: bash

   rapthor input.parset

The change is an execution-architecture decision, not a change to Rapthor's
scientific purpose. The self-calibration loop, calibration products, sky-model
products, image products, and diagnostics must remain scientifically equivalent
for the supported contracts.

Context
-------

Historically, Rapthor selected operation objects in Python, rendered generated
CWL/Jinja workflow files, and delegated execution to CWLRunner/Toil. That model
worked, but over time it split important behaviour across several places:

* Python operation classes and field state
* generated CWL workflow graphs
* Jinja templates
* shell wrapper scripts
* Toil job-store state
* external commands such as DP3, WSClean, fpack, and helper scripts
* output-record and restart marker conventions

This made the system harder to inspect, test, refactor, and scale. It also made
it difficult to expose useful runtime information to users while a run was in
progress.

The current refactor has already moved the orchestration graph into typed Python
flows and migrated helper-script behaviour into importable execution modules.
This ADR records the proposed decision to make that architecture the production
path.

Decision Drivers
----------------

The main drivers are:

* **Scientific trust:** changes to orchestration must preserve the tested
  science product contract.
* **Observability:** users need clear task names, dashboards, logs, command
  metrics, artifacts, and failure locations.
* **Maintainability:** operation behaviour should be readable in Python rather
  than split between Python, generated CWL, Jinja, and wrappers.
* **Testability:** command builders, payload builders, output discovery, and
  task bodies should be unit-testable without generating workflow files.
* **Scalability:** execution should support local Dask, external Dask, Slurm,
  and MPI WSClean paths.
* **Runtime UX:** ``rapthor input.parset`` should work with low friction,
  including without a pre-existing Prefect server or Dask cluster.
* **Evidence-driven switching:** the branch should replace ``master`` only with
  documented science and performance evidence.

Options Considered
------------------

Keep CWL/Toil as the production path
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This would minimize short-term migration risk, but would keep the existing
maintenance and observability problems. It would also leave the orchestration
graph outside the typed Python code that now owns most operation state.

Keep CWL/Toil and add more wrappers or generated tests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This would improve local confidence in the old architecture, but would not
address the root problem: important behaviour would still be split across
Python, generated workflow files, templates, wrapper scripts, and runner state.

Replace CWL/Toil with Python/Prefect/Dask
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This moves orchestration into typed Python flows while keeping external radio
astronomy tools as leaf commands. It makes operation boundaries explicit,
serializable payloads testable, task state visible in dashboards, and Dask/Slurm
runtime choices configurable.

This is the proposed option.

Consequences
------------

Positive consequences
~~~~~~~~~~~~~~~~~~~~~

* The orchestration graph is visible in Python flow modules.
* Operation adapters can remain thin and focused on translating domain state
  into execution payloads and finalizing outputs.
* Payloads passed to workers are plain, serializable data structures.
* Command construction is deterministic and independently testable.
* Helper-script logic lives in execution owner packages instead of standalone
  scripts.
* Prefect gives run, flow, subflow, task, log, and artifact visibility.
* Dask gives a path to local and distributed execution.
* Task boundaries can be split where they improve observability or measured
  performance.
* Runtime metrics, command logs, FITS previews, and postage-stamp artifacts can
  be published consistently.
* Science and performance equivalence gates provide reviewer-facing evidence
  for the switch decision.

Negative consequences and risks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Prefect, Dask, ``prefect-dask``, and ``prefect-shell`` become production
  runtime dependencies.
* Users and operators need clear guidance for dashboards, local Dask, external
  Dask, and Slurm deployments.
* Parallel production jobs without a shared Prefect service need isolated
  runtime state and unique working directories.
* Multi-node Slurm runs and MPI WSClean need staging validation before broad
  operational rollout.
* Some historical CWL output metadata no longer exists in the same form; the
  science product contract must distinguish scientific products from legacy
  runner metadata.
* Long-lived compatibility with ``master`` parsets requires migration guidance,
  especially for calibration strategy options.

Architecture Rules
------------------

The following rules are part of the decision:

* Domain objects such as ``Field``, ``Observation``, ``Sector``, and strategy
  handling must not depend on Prefect or Dask.
* Operation classes should stay thin. They translate field/domain state into
  execution payloads, run the appropriate flow, and finalize outputs.
* Prefect tasks must receive serializable payloads. They must not receive live
  ``Field`` objects, operation instances, open tables, open file handles, or
  subprocess state.
* Command builders should be deterministic and runnable in unit tests.
* Execution owner packages should own migrated helper logic:
  ``rapthor.execution.image``, ``rapthor.execution.calibrate``,
  ``rapthor.execution.predict``, ``rapthor.execution.mosaic``, and
  ``rapthor.execution.concatenate``.
* External tools remain leaf commands unless there is a clear, tested reason to
  call an importable Python API directly.
* Science-facing state transitions must be explicit, especially calibration
  solution seeding, same-cycle solution use, DI pre-apply, and DD on-the-fly
  application.

Evidence Required Before Acceptance
-----------------------------------

This ADR should move from Proposed to Accepted when the switch-readiness
criteria in ``PLAN.md`` are met.

The evidence package should include:

* a current reviewer-facing summary in ``EQUIVALENCE_REPORT.md``
* science equivalence reports under
  ``docs/source/development/science_equivalence_runs/``
* performance equivalence reports under
  ``docs/source/development/performance_equivalence_runs/``
* representative integration-test results
* benchmark reports for the current optimisation phase
* manual tester feedback from developers who were not involved in the refactor
* documented caveats and accepted differences from ``master``

At the time this ADR was written, the branch already had accepted science
equivalence for the covered LOFAR HBA self-calibration contract and
repeatability-aware performance evidence showing the current branch faster than
``master`` for the tested phase-only and DD/full-Jones scenarios. Final
acceptance still depends on switch-readiness and operational validation.

Operational Requirements
------------------------

The Prefect/Dask production path should support:

* interactive single-machine runs without Slurm
* runs with no existing Prefect server
* optional use of a persistent Prefect server when the user sets a Prefect API
  URL
* local Dask workers for normal developer and small production runs
* external Dask schedulers for staged and production-like runs
* multiple independent Rapthor jobs running concurrently without a shared
  Prefect server, provided they use isolated runtime state and unique working
  directories
* Slurm allocation scripts for multi-node Dask workers
* WSClean MPI imaging across allocated nodes
* clear copy/paste dashboard tunnel instructions for Prefect and Dask in
  multi-node demonstrations

Implementation Notes
--------------------

The implementation should keep the following package shape:

* ``rapthor.operations``: thin operation adapters
* ``rapthor.execution.pipeline``: top-level pipeline flow and lifecycle hooks
* ``rapthor.execution.<operation>.flow``: operation-level Prefect flow
  orchestration
* ``rapthor.execution.<operation>.payloads``: typed serializable payload
  contracts
* ``rapthor.execution.<operation>.builders`` and ``validation``: mapping and
  validation between operation inputs and payloads
* ``rapthor.execution.<operation>.commands``: deterministic command token
  builders
* ``rapthor.execution.<operation>`` work-unit modules: scheduler-independent
  task bodies
* ``rapthor.execution.shell``, ``task_runner``, ``slurm``, ``resources``, and
  ``artifacts``: shared runtime infrastructure

Documentation Impact
--------------------

The following documentation should stay aligned with this ADR:

* ``docs/source/development/architecture.rst``
* ``docs/source/development/architecture_views.rst``
* ``docs/source/development/manual_testing_prefect_dask.rst``
* ``docs/source/development/science_equivalence_contract.rst``
* ``docs/source/development/performance_equivalence_contract.rst``
* ``EQUIVALENCE_REPORT.md``

Follow-Up Decisions
-------------------

Separate ADRs or decision notes may be useful for:

* production Prefect server deployment with a Postgres backend
* Slurm/external-Dask launcher strategy
* Spack/module packaging for the Prefect/Dask branch
* long-term benchmark/performance gate policy
* retirement date for any remaining compatibility shims
* whether and where Dask arrays should replace in-memory NumPy arrays for large
  FITS/mosaic operations
