Development Architecture
========================

This page documents the intended architecture for the post-CWL/Toil
Prefect/Dask codebase. It is a working guide for refactors, not a claim that
every module has already reached the target shape.

Clean Architecture Rule
-----------------------

Rapthor should follow one-way dependencies:

* Domain code does not depend on execution frameworks.
* Application/use-case code may depend on the domain and plain serializable
  contracts, but not on Prefect task objects or live Dask runtime state.
* Interface adapters translate between use-case contracts and the outside world:
  command lines, scripts, output records, filesystems, and shell execution.
* Frameworks and drivers contain Prefect flows, Dask task runners, Slurm
  integration, artifacts, dashboards, and runtime resource checks.

The practical rule is that inner layers must not import outer layers. If a
dependency has to cross outward, introduce a small protocol, callback, or
adapter and inject it from the outer layer.

Layer Ownership
---------------

.. list-table::
   :header-rows: 1

   * - Area
     - Responsibility
     - Primary tests
   * - ``rapthor.lib``
     - Field, observation, sector, cluster, strategy, parset interpretation,
       operation lifecycle primitives, and finalizer-visible domain state.
     - ``tests/lib``
   * - ``rapthor.application`` or ``rapthor.use_cases``
     - Future home for scheduler-independent operation planning, typed payload
       contracts, parset/field-to-payload mapping, restart decisions, and
       workflow decisions.
     - Contract and operation tests as the layer is introduced.
   * - ``rapthor.operations``
     - Operation adapters that connect domain objects to payload builders, run
       flows, update field state, and handle finalizer side effects.
     - ``tests/operations``
   * - ``rapthor.execution.commands``
     - Deterministic external-command token construction and display helpers.
     - ``tests/execution/test_commands.py`` and command reference fixtures.
   * - ``rapthor.execution.payloads``
     - Transitional home for payload serialization checks and typed payload
       contracts until scheduler-independent use-case modules own them.
     - ``tests/execution/test_payloads.py``
   * - ``rapthor.execution.outputs``
     - Transitional or adapter-level output-record helpers until record handling
       is consolidated with the finalizer/domain record API.
     - ``tests/execution/test_outputs.py``
   * - ``rapthor.execution.flows``
     - Prefect orchestration: task boundaries, scheduling, retries, artifacts,
       flow-level validation, and runtime integration.
     - ``tests/execution/test_*_flow.py``
   * - ``rapthor.execution.runtime``, ``task_runner``, ``resources``,
       ``slurm``, ``workdirs``, ``artifacts``, and ``shell``
     - Runtime infrastructure and adapters for local, Dask, Slurm, shell, and
       artifact behaviour.
     - Focused tests under ``tests/execution``
   * - ``rapthor.scripts``
     - Standalone helper scripts. As scripts are touched, move core behaviour
       into importable functions and keep CLIs as thin wrappers.
     - ``tests/scripts``

Public Export Guidance
----------------------

``rapthor.execution`` and ``rapthor.execution.flows`` currently expose broad
facades for compatibility with the migration-era test and import surface. Treat
these as transitional convenience surfaces.

New code should import from the module that owns the behaviour, for example:

* command helpers from ``rapthor.execution.commands`` or operation-specific
  command modules once they exist
* payload helpers from the payload/use-case module that owns the contract
* Prefect flows from the concrete ``rapthor.execution.flows.<operation>`` module
* runtime helpers from their concrete runtime module

Do not add new facade exports unless there is a documented compatibility reason.

Current Boundary Exceptions
---------------------------

The first architecture fitness tests intentionally allow known migration-era
exceptions so they can prevent new leaks without forcing a large first patch.

Known exception:

* ``rapthor.lib.operation.Operation.run_prefect_flow`` imports
  ``rapthor.execution.config.ExecutionConfig``. This keeps current operation
  adapters small, but it points from the domain/lifecycle layer into the
  execution layer. Remove this exception when operation lifecycle and flow
  execution are split more cleanly.

Change Workflow
---------------

When changing a parset option:

* update defaults and documentation
* update the domain or use-case mapping that owns the meaning
* update the payload contract and command builder if execution changes
* add focused payload, command, operation, and process tests as needed

When changing an operation:

* keep lifecycle/finalizer effects in the operation adapter
* keep parset/field-to-payload mapping in pure helpers
* keep command construction deterministic and independently testable
* cover restart/reuse and field hand-off behaviour

When changing runtime behaviour:

* keep Prefect/Dask/Slurm details outside domain and command modules
* test configuration, resource validation, task-runner selection, and failure
  messages without real external tools where possible
* use integration or target-environment tests only for behaviour that needs the
  deployment stack

When converting a script to a module:

* extract an importable Python function first
* keep the existing CLI as a compatibility wrapper
* add Python function tests and CLI parity tests
* keep large data movement explicit and Dask-friendly
