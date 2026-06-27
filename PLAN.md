# Rapthor Architecture Refactor Plan

Status snapshot: 2026-06-27.

## Goal

Make the post-migration Python/Prefect/Dask codebase easier to understand,
extend, test, and operate without changing Rapthor's scientific behaviour or
public parset/strategy contract.

The target architecture should let a developer or scientist answer these
questions quickly:

- Where do I add or change a parset option?
- Where do I translate domain objects into an execution payload?
- Where do I build the external tool command?
- Where does Prefect/Dask scheduling happen?
- Where are operation outputs recorded and finalized?
- Which tests prove that a change preserved command lines, output records, and
  restart behaviour?

## Refactor Principles

- Keep every refactor slice behaviour-preserving unless it is explicitly scoped
  as a feature change.
- Preserve the existing public parset, strategy, CLI, output file, restart, and
  finalizer contracts.
- Prefer pure functions and typed data contracts at module boundaries.
- Keep Rapthor domain objects out of Dask worker payloads; pass plain,
  serializable values only.
- Keep external-tool command builders deterministic and independently testable.
- Keep Prefect flows thin: orchestration, task wiring, artifacts, and runtime
  concerns belong there; domain extraction and command construction should not.
- Move broad code only after tests pin the current behaviour.
- Update documentation and examples whenever the contribution path changes.

## Clean Architecture Boundaries

The refactor should follow the dependency rule: inner layers must not import
outer layers. Scientific domain code should stay independent of Prefect, Dask,
Slurm, shell execution, artifact publication, and external command runners.

Use these layers as the architectural guide:

- Domain: field, observation, sector, cluster, strategy, parset-derived scientific
  state, and finalizer-visible operation state.
- Application/use cases: operation planning, parset-to-payload mapping,
  restart/reuse decisions, process feature detection, and workflow decisions
  that are independent of a specific scheduler.
- Interface adapters: command builders, output-record conversion, script
  wrappers, filesystem record handling, and adapters between domain/use-case
  concepts and serializable execution payloads.
- Frameworks/drivers: Prefect flows and tasks, Dask task runners, Slurm
  integration, shell execution, artifact publishing, dashboards, logging
  integrations, and runtime resource checks.

Dependency direction should be one-way:

- `rapthor.lib` must not import `rapthor.execution`, Prefect, Dask, Slurm, or
  shell-command infrastructure.
- Application/use-case helpers may depend on domain objects and plain typed
  payload contracts, but not on Prefect task objects or Dask runtime state.
- Command builders should return deterministic token lists and should not run
  commands, publish artifacts, or inspect Prefect context.
- Prefect flows should depend inward on payload contracts, command builders, and
  pure helpers; pure helpers should not depend outward on Prefect flows.
- Runtime integrations should be replaceable adapters around stable ports such
  as command execution, artifact publishing, work-directory layout, task
  scheduling, and resource validation.

When a dependency needs to cross outward, define a small protocol or adapter
interface first. For example, use injectable collaborators for command
execution, artifact publication, runtime scheduling, and lifecycle hooks so
tests can exercise the use case without starting Prefect, Dask, Slurm, or real
external tools.

## Target Module Shape

The exact package names can evolve while implementing the plan, but the intended
responsibilities are:

- `rapthor.lib`: domain model, parset/strategy interpretation, field/sector/
  observation state, operation lifecycle primitives, and finalizer-compatible
  record utilities.
- `rapthor.operations`: operation adapters that connect domain objects to flow
  payload builders, run the selected flow, update field state, and handle
  operation finalization.
- `rapthor.execution.payloads`: typed payload contracts and serialization
  validation shared by operation adapters and Prefect tasks.
- `rapthor.execution.commands`: shared command token utilities plus small
  operation-specific command modules where useful.
- `rapthor.execution.outputs`: a single compatibility layer, or a deprecated
  shim, for finalizer-compatible output records.
- `rapthor.execution.flows`: Prefect orchestration only: task boundaries,
  scheduling, retries/failure handling, artifact publication, and task-runner
  integration.
- `rapthor.execution.runtime`, `task_runner`, `resources`, `slurm`, `workdirs`,
  `artifacts`, and `shell`: reusable runtime infrastructure.
- `rapthor.scripts`: standalone helper scripts used by external command
  builders, kept testable with small fixtures.

## Step-By-Step Plan

### 0. Establish The Refactor Safety Net

Outcome: contributors can refactor with confidence because the current behaviour
is pinned before code moves.

- Record the current architecture boundaries in this plan and in developer docs.
- Keep the existing command/output reference fixtures as regression anchors.
- Add missing focused tests before moving behaviour out of large modules.
- For each refactor slice, run the narrowest relevant tests first, then a broader
  non-integration lane before merging.
- Avoid large formatting-only rewrites during behaviour-preserving moves; they
  make scientific parity review harder.

Suggested first checks:

```bash
python3 -m pytest tests/execution/test_commands.py tests/execution/test_outputs.py -q --tb=short
python3 -m pytest tests/execution/test_payloads.py tests/operations -q --tb=short
```

### 1. Define Stable Internal Boundaries

Outcome: the codebase has a small number of documented internal contracts instead
of accidental imports from large implementation modules.

- Audit `rapthor.execution.__init__` and `rapthor.execution.flows.__init__`.
- Decide which imports are stable internal API and which are test convenience
  exports.
- Move tests toward direct imports from the module that owns the behaviour.
- Keep temporary compatibility exports only where existing users are likely to
  rely on them.
- Add a short `docs/source/development/architecture.rst` page describing the
  domain, operation, payload, command, flow, and runtime layers.
- Add an import-boundary note for new contributors: operation adapters build
  payloads, payload builders create serializable contracts, command builders
  create deterministic token lists, flows orchestrate execution.

Completion criteria:

- New contributors can identify the owning module for payload, command, output,
  flow, and finalizer changes.
- `__init__` exports are either intentionally documented or scheduled for
  deprecation.

### 2. Consolidate Output Record Handling

Outcome: Rapthor has one finalizer-compatible record API for files,
directories, optional values, nested lists, path extraction, validation, copying,
and cleanup.

- Compare `rapthor.execution.outputs` with `rapthor.lib.records`.
- Pick the long-term home for record creation and validation. Prefer the domain
  layer if finalizers and operations both need the API.
- Move shared helpers into that home:
  - `file_record` and `directory_record`
  - required and optional path extraction
  - basename validation where command builders depend on it
  - nested record validation for lists and optional records
  - copy, move, and cleanup helpers used by finalizers
- Leave `rapthor.execution.outputs` as a small compatibility shim if needed.
- Replace local `_file_record_path`, `_directory_record_path`, and
  `_optional_file_record_path` helpers in flow modules with the shared API.
- Add tests for malformed records, missing paths, optional records, nested lists,
  and path extraction error messages.

Completion criteria:

- Flow modules no longer contain duplicated record path extraction helpers.
- Operation finalizers and Prefect flows validate the same record contract.
- Existing output reference tests still pass.

### 3. Introduce Typed Payload Contracts

Outcome: payload builders, Prefect tasks, and tests stop depending on large
untyped dictionaries whose shape is hard to discover.

- Start with the highest-risk flows: image and calibrate.
- Add small `TypedDict` or dataclass contracts in `rapthor.execution.payloads`
  or operation-specific payload modules.
- Keep payload values plain and serializable: strings, numbers, booleans, lists,
  dictionaries, and `None`.
- Provide explicit conversion helpers such as `from_operation_inputs(...)`,
  `to_payload()`, or `validate_payload(...)`.
- Keep the existing `assert_serializable_payload()` check at task boundaries.
- Add tests that exercise the payload builders without starting Prefect.
- Keep payload keys stable until downstream code has migrated.

Recommended extraction order:

1. Shared record/path fields.
2. Concatenate, mosaic, and predict payloads as smaller proving grounds.
3. Image sector payloads and image flow payloads.
4. Calibrate solver/chunk/screen/collection payloads.
5. Process-flow lifecycle payloads if they become more complex.

Completion criteria:

- A developer can inspect one payload type to see required and optional fields.
- Payload tests fail at construction time for missing or invalid high-value
  fields.
- Dask workers continue to receive only serializable data.

### 4. Consolidate Command Builder Utilities

Outcome: external-tool command construction is deterministic, readable, and easy
to test without running DP3, WSClean, EveryBeam, IDG, or helper scripts.

- Expand `rapthor.execution.commands` with shared utilities already repeated in
  flow modules:
  - boolean tokens
  - optional flag/value appending
  - list token expansion
  - path-list joins
  - normalized command wrappers
  - shell-safe display strings
- Keep operation-specific command builders close to their domain until the best
  module split is obvious.
- Extract command builder groups from large flows into modules such as:
  - `rapthor.execution.commands.image`
  - `rapthor.execution.commands.calibrate`
  - `rapthor.execution.commands.predict`
  - `rapthor.execution.commands.mosaic`
  - `rapthor.execution.commands.concatenate`
- Preserve existing function names through compatibility imports while tests are
  migrated.
- Add focused tests for each command builder group using command token fixtures.

Completion criteria:

- Flow modules call command builders but do not assemble long command token lists
  inline.
- All command builders are importable and testable without Prefect.
- Command reference fixtures remain stable unless an intentional scientific or
  runtime change is documented.

### 5. Split The Image Flow By Responsibility

Outcome: `rapthor.execution.flows.image` becomes a readable orchestration layer
instead of one large module containing command construction, payload mapping,
sector execution, output discovery, and artifact logic.

- Extract image command builders first, using the command plan above.
- Extract image payload mapping into a pure payload module.
- Extract sector work into focused helpers:
  - prepare imaging data
  - concatenate time chunks
  - mask and region preparation
  - WSClean command selection
  - image cube and catalog creation
  - source filtering and diagnostics
  - output discovery and validation
- Extract artifact publication wrappers only if they are reusable or obscure the
  flow.
- Keep the Prefect flow responsible for task wiring, scheduling, result
  aggregation, and publishing flow-level artifacts.
- Split tests to mirror the new boundaries:
  - command builders
  - payload mapping
  - sector execution with shell mocked
  - output discovery contracts
  - flow-level orchestration

Completion criteria:

- The image flow module can be read top to bottom as orchestration.
- Most image logic can be tested without Prefect or external radio astronomy
  tools.
- Image operation tests still prove finalizer-visible field state and restart
  behaviour.

### 6. Split The Calibrate Flow By Responsibility

Outcome: `rapthor.execution.flows.calibrate` becomes a readable orchestration
layer with solver, screen, collection, plotting, and combine logic separated.

- Extract calibration command builders.
- Extract solver payload mapping and validation.
- Split pure helpers for:
  - DDECal solve command setup
  - IDGCal phase and phase/gain solve setup
  - draw-model and region setup
  - solve chunk execution
  - screen collection
  - H5Parm collection and combination
  - plot solution selection and artifact publication
  - source adjustment and gain processing
- Make solve-mode branching explicit and testable.
- Keep Prefect tasks small and named after the unit of work scientists recognise.
- Split tests by command, payload, chunk execution, collect/combine, and
  flow-level orchestration.

Completion criteria:

- Adding a new solve mode or solver command does not require editing unrelated
  plotting, collection, or artifact code.
- Calibration branch coverage improves without invoking external tools.
- Existing calibration output records and field finalization remain unchanged.

### 7. Thin Operation Adapters

Outcome: operation classes express lifecycle and finalizer effects, while pure
helpers do parset/field/input-to-payload mapping.

- For `Image` and `Calibrate`, identify methods that only read parset/field state
  and build flow inputs.
- Move those mappings into pure helper modules or functions with focused tests.
- Keep operation classes responsible for:
  - lifecycle setup
  - restart/done/output file handling through the base `Operation`
  - calling the selected Prefect flow
  - updating field attributes expected by later operations
  - copying/cleaning outputs
- Preserve monkeypatch-friendly operation constructors used by process tests.
- Repeat the same pattern for predict, mosaic, and concatenate only where it
  simplifies real behaviour.

Completion criteria:

- A scientist can review finalizer side effects without reading command builder
  code.
- A developer can test parset-to-payload mapping without constructing a full
  runtime environment.
- Operation adapters are smaller and follow the same shape across operations.

### 8. Clarify Process Orchestration

Outcome: top-level orchestration remains easy to reason about as more runtimes,
feature flags, and scientific modes are added.

- Keep `rapthor.process.run()` as the user-facing entry point.
- Keep `rapthor.execution.flows.process` responsible for Prefect/Dask process
  scheduling.
- Continue using injectable lifecycle hooks and operation factories for tests.
- Move feature detection helpers into a small, documented module if they grow.
- Keep preflight validation close to runtime capability checks.
- Document how process features map to strategy steps and parset options.
- Add tests for new feature flags before wiring them into execution.

Completion criteria:

- Process tests can cover operation ordering, skip conditions, error propagation,
  restart/reset behaviour, and compatibility helpers without real commands.
- Runtime preflight failures produce actionable messages.

### 9. Improve Test Structure And Coverage

Outcome: tests teach the architecture rather than only guarding historical
behaviour.

- Keep the default non-integration suite free of external-tool requirements.
- Keep integration tests representative and environment-aware.
- Split oversized tests along the same boundaries as the code.
- Add fixtures that make common payloads, records, sectors, fields, and shell
  command results easy to construct.
- Prefer small FITS, H5Parm, sky-model, region, and Measurement Set fixtures
  already in `tests/resources/`.
- Target the next practical milestone at 85% non-integration coverage.
- Do not add a hard coverage gate until CI is stable above the chosen threshold
  for several runs.

High-value coverage areas:

- malformed command-log and artifact contexts
- `perf` success, failure, and no-sample paths
- FITS preview edge cases
- Dask/task-runner fallback behaviour
- `rapthor.process` restart/reset and compatibility helpers
- `Field` regrouping, target selection, normalization scaling, and empty model
  branches
- pure script helpers such as `subtract_sector_models.py`,
  `collect_screen_h5parms.py`, `check_image_beam.py`, `blank_image.py`,
  `combine_h5parms.py`, `process_gains.py`, and `make_region_file.py`

Completion criteria:

- The most important branches in payload mapping, command construction, output
  validation, orchestration, and finalization are covered by focused tests.
- Integration tests remain smoke/regression coverage for real external tools
  rather than the only way to validate business logic.

### 10. Improve Contribution Documentation

Outcome: developers and scientists have a clear path for common changes.

- Add a development architecture guide under `docs/source/development/`.
- Add "How to add a new parset option" documentation:
  - defaults
  - docs/examples
  - domain object
  - payload mapping
  - command builder
  - flow/task usage
  - tests
- Add "How to add or modify an operation" documentation:
  - operation adapter
  - payload contract
  - command builder
  - flow task
  - output records
  - finalizer effects
  - focused and integration tests
- Add "How to add a new external command helper" documentation:
  - script location
  - command builder
  - unit fixtures
  - artifact/logging expectations
- Keep examples in sync with supported runtime options.

Completion criteria:

- A new contributor can make a small operation or parset change by following the
  guide without reverse-engineering image or calibration internals first.

### 11. Runtime And Scalability Validation

Outcome: the cleaner architecture still supports local development, external
Dask, Slurm, and future SKA-Low scaling work.

- Keep runtime concerns isolated in `execution.config`, `runtime`,
  `task_runner`, `resources`, `slurm`, and `workdirs`.
- Use Dask dashboard and performance reports to review task granularity after
  code boundaries are clearer.
- Validate external-Dask and Slurm in representative allocations.
- Validate MPI WSClean in the intended deployment stack.
- Validate `prefect_command_profile = perf` in development and CI container
  runtimes where host kernel permissions allow sampling.
- Use rich demo runs to review command summary charts, logs, artifacts, and
  flamegraphs.

Completion criteria:

- The architecture supports runtime changes without pushing Slurm/Dask details
  into operation adapters or command builders.
- Profiling artifacts remain useful for DP3/WSClean bottleneck analysis.

### 12. Final Polish And Maintenance

Outcome: the refactor lands as a sequence of small, reviewable improvements.

- Run Ruff/formatting after each cluster of code moves.
- Remove compatibility shims only after imports and docs are migrated.
- Keep deprecations explicit and documented.
- Update `README.md`, release notes, and examples when user-facing behaviour or
  contribution paths change.
- Keep generated demo data, large reference artifacts, and run outputs ignored by
  VCS.

## Suggested Implementation Order

1. Document internal boundaries and audit public exports.
2. Consolidate output records.
3. Add typed payload contracts for concatenate, mosaic, and predict.
4. Extract shared command utilities.
5. Move image command builders and payload mapping out of the image flow.
6. Move image sector execution and output discovery into focused helpers.
7. Move calibration command builders and payload mapping out of the calibration
   flow.
8. Move calibration chunk/screen/collect/combine helpers into focused modules.
9. Thin `Image` and `Calibrate` operation adapters.
10. Split tests to match the new modules.
11. Add contributor documentation for common change paths.
12. Validate broader non-integration tests, then representative integration/demo
    runs.

## Useful Commands

Serial Prefect-server lane:

```bash
python3 -m pytest -m "not integration and prefect" -k "not test_field.py" tests -q --tb=short
```

Parallel non-Prefect lane:

```bash
python3 -m pytest -m "not integration and not prefect" -n auto -k "not test_field.py" tests -q --tb=short
```

Focused process-flow checks:

```bash
python3 -m pytest tests/execution/test_process_flow.py tests/test_process.py -q --tb=short
```

Focused output/payload/command checks:

```bash
python3 -m pytest tests/execution/test_outputs.py tests/execution/test_payloads.py tests/execution/test_commands.py -q --tb=short
```

Focused operation adapter checks:

```bash
python3 -m pytest tests/operations -q --tb=short
```

Non-integration coverage check:

```bash
python3 -m pytest -m "not integration" --cov=rapthor --cov-report=term-missing tests
```

Rich local demo:

```bash
scripts/dev/generate-prefect-demo-data.py --force
scripts/dev/run-rapthor-prefect-demo.py \
  --task-runner local_dask \
  --dask-dashboard-address :8787 \
  --dask-performance-report \
  examples/generated/prefect_demo_rich/prefect_demo_rich.parset
```

## Merge Criteria For Refactor Slices

- Public parset, strategy, CLI, restart, output-record, and finalizer behaviour is
  unchanged unless an intentional behaviour change is documented.
- Command token fixtures and output reference fixtures remain stable or are
  updated with a clear reason.
- New modules have focused tests at the same abstraction level.
- Prefect-specific tests are isolated from high xdist fan-out.
- Non-integration tests pass for the touched area, plus the relevant broader
  lane before merge.
- Docs are updated when a change affects contribution flow, user-facing options,
  runtime behaviour, or operation semantics.

## Deferred Follow-Up

- Slurm/external-Dask validation remains deferred until it can be run inside a
  representative Slurm allocation.
- MPI WSClean validation remains deferred until it can be run with the intended
  MPI/WSClean deployment stack.
- Dask task-granularity/resource optimization should happen after the module
  boundaries are cleaner, using Dask dashboards and performance reports.
- Revisit `hybrid_screens` only if it becomes a supported target workflow.
- Revisit `shared_facet_rw` after WSClean shared-facet read/write behaviour is
  reliable in the intended environment.
