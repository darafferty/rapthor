Architecture Views
==================

This page keeps concise architecture diagrams for Rapthor's current
Prefect/Dask implementation. 

The editable Mermaid sources live beside this page:

* ``rapthor_component_view.mmd``
* ``rapthor_prefect_dask_connector_view.mmd``
* ``rapthor_runtime_deployment_connector_view.mmd``

Component View
--------------

Purpose
~~~~~~~

Show the main implementation components and their intended dependency
direction. This view is most useful when deciding where new code should live.

Primary Presentation
~~~~~~~~~~~~~~~~~~~~

.. mermaid:: rapthor_component_view.mmd

Element Catalog
~~~~~~~~~~~~~~~

``rapthor.cli``
   User-facing command entry point. Reads a parset and starts the pipeline.

``rapthor.lib``
   Domain model: parset interpretation, field state, observations, sectors,
   strategy handling, and operation lifecycle primitives.

``rapthor.operations``
   Thin operation adapters. They translate domain state into execution
   payloads, call the appropriate flow, and finalize domain-visible outputs.

``rapthor.execution.<owner>``
   Owner packages for concatenate, calibrate, predict, image, mosaic, and the
   top-level pipeline. They contain payloads, builders, validation, command
   builders, work-unit modules, outputs, and Prefect flows.

``rapthor.execution`` runtime helpers
   Shared runtime infrastructure: shell execution, task runners, artifacts,
   resources, Slurm, command profiling, and runtime bootstrap.

External tools
   DP3, WSClean, LoSoTo, PyBDSF, fpack, and related astronomy tools.

Connector Catalog
~~~~~~~~~~~~~~~~~

``reads``
   CLI reads parset, strategy, and runtime configuration.

``plans``
   Domain and operation layers choose the operations and build payloads.

``submits``
   Operation adapters submit serializable payloads to operation flows.

``orchestrates``
   Prefect flows create task boundaries and schedule work through Dask.

``runs command`` / ``imports helper``
   Work-unit modules either call imported execution helpers or run isolated
   shell/module-adapter commands.

Rationale
~~~~~~~~~

The dependency direction keeps scientific state and parset semantics separate
from Prefect, Dask, shell execution, and deployment details. Payloads are plain
serializable data so task boundaries remain visible and testable.

Prefect/Dask Connector View
---------------------------

Purpose
~~~~~~~

Show how a normal ``rapthor input.parset`` run connects the top-level pipeline
flow, operation flows, task bodies, Dask workers, external tools, and output
records.

Primary Presentation
~~~~~~~~~~~~~~~~~~~~

.. mermaid:: rapthor_prefect_dask_connector_view.mmd

Element Catalog
~~~~~~~~~~~~~~~

``pipeline_flow``
   Top-level Prefect flow. It handles preflight, lifecycle hooks, strategy
   iteration, operation ordering, and final pipeline state.

Operation flows
   Operation-level Prefect flows for concatenate, calibrate, predict, image,
   and mosaic. They own task names, task dependencies, retries, task tags, and
   flow-local artifacts.

Work-unit modules
   Plain Python task bodies that build commands, call helper functions, run
   shell commands, collect outputs, and return serializable records.

Dask task runner
   Executes Prefect tasks on local or external Dask workers when configured.

External commands and module adapters
   Leaf work such as DP3, WSClean, fpack, PyBDSF-backed filtering, image-cube
   helpers, and LoSoTo plotting.

Products and observability outputs
   Images, mosaics, h5parm solutions, sky models, diagnostics, logs,
   command records, profiles, and Prefect artifacts.

Connector Catalog
~~~~~~~~~~~~~~~~~

``call flow``
   A parent flow invokes an operation flow.

``submit task``
   A flow submits a Prefect task with serializable payloads.

``execute``
   The task runner executes the task body on a worker or locally.

``spawn``
   A task body starts an external command or ``python -m`` module adapter.

``write`` / ``publish``
   Tasks write products and publish logs, profiles, metrics, and artifacts.

Rationale
~~~~~~~~~

The orchestration graph is explicit Python, while expensive astronomy tools
remain isolated leaf work. This gives the dashboard useful task granularity
without pushing domain objects or live resources across Dask worker
boundaries.

Runtime Deployment Connector View
---------------------------------

Purpose
~~~~~~~

Show the supported runtime deployment shapes for local testing, persistent
dashboards, and production-style Slurm or external-Dask execution.

Primary Presentation
~~~~~~~~~~~~~~~~~~~~

.. mermaid:: rapthor_runtime_deployment_connector_view.mmd

Element Catalog
~~~~~~~~~~~~~~~

Local no-server run
   A single Rapthor process runs with no configured ``PREFECT_API_URL`` and a
   local Dask task runner. This is the lowest-friction path for developers.

Persistent Prefect server
   Optional Prefect API and dashboard used when the user wants durable flow
   history and browser-based monitoring.

External Dask cluster
   Optional scheduler and workers used for larger local, staging, or
   production runs.

Slurm allocation
   Production-style path where scheduler, workers, MPI-capable WSClean, and
   external tools run inside allocated resources.

Connector Catalog
~~~~~~~~~~~~~~~~~

``configure``
   Parset and environment choose local or external runtime resources.

``report state``
   Flow and task states are reported to Prefect when a reachable API is
   configured.

``schedule tasks``
   Prefect/Dask submit Python task bodies to local or external workers.

``launch MPI``
   WSClean MPI runs under the allocated Slurm resources when imaging is
   configured for MPI.

Rationale
~~~~~~~~~

The same application path supports low-friction interactive testing and
production-style multi-node runs. The branch should remain usable without a
shared Prefect server, while still allowing dashboards when users explicitly
configure them.
