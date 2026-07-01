# Rapthor Architecture Refactor Plan

Status snapshot: 2026-07-01.

## Goal

Make the Prefect/Dask Rapthor pipeline easy to understand, extend, test, debug,
and scale while preserving the user-facing CLI workflow:

```bash
rapthor input.parset
```

This Prefect/Dask implementation has not been released, so prefer clean
production architecture over unreleased Python API compatibility, migration
aliases, compatibility shims, or test-only production surfaces.

## Current Status

The main architecture cleanup is complete enough to move from structural
refactoring into scalability and usability work.

Completed:

- Execution code is organized by owner package:
  `image`, `calibrate`, `concatenate`, `predict`, `mosaic`, and `pipeline`.
- Image and calibration operations are package-based adapters.
- Migrated helper-script logic lives in importable execution modules.
- Production flows call migrated Python helpers directly, except where shell
  isolation is still useful for external tools or third-party multiprocessing.
- Retired helper scripts and the old `plotrapthor` executable are guarded by architecture
  tests so production code and command fixtures do not reintroduce them.
- The installed `rapthor` command is exposed through `rapthor.cli:main`.
- `concat_linc_files` remains a supported installed utility through the package-owned
  `rapthor.execution.concatenate.linc_cli:main` entry point.
- Broad execution facades, normalized command wrappers, migration shims, and
  unused runtime abstractions have been removed.
- Command builders are deterministic and tested.
- Image and calibration command builders use option dataclasses where argument
  groups are stable.
- Payload contracts, builders, and validation live with operation-specific
  execution code:
  - `rapthor.execution.image.contracts/builders/validation.py`
  - `rapthor.execution.calibrate.contracts/builders/validation.py`
- Predict sector-model add/subtract now share Measurement Set mechanics through
  `rapthor.execution.predict.measurement_sets`, with direct unit coverage.
- Scheduler-independent work units are separated from Prefect flow wiring for
  the complex image and calibration paths.
- Prefect operation flow and task run names now include operation, calibration
  mode where relevant, cycle, and coarse task identifiers so the dashboard is
  easier to scan during rich demo and integration runs.
- The development architecture docs include a current Prefect/Dask
  orchestration diagram that matches the refactored owner-package layout.
- `calibration_strategy` is the only production interface for solve type and
  solve order. Legacy `do_fulljones_solve` and `do_slowgain_solve` flags are
  retired from production configuration.
- The default DD strategy is explicit:
  `{"dd": ["fast_phase", "medium_phase", "slow_gains", "medium_phase"], "di": []}`.
- Focused calibration integration tests pass for DD, DI, mixed DI/DD ordering,
  full-Jones, slow-gain, and calibration-option scenarios.
- The saved CWL equivalence matrix passes against the current scientific
  contract when run on a filesystem with enough space for WSClean FITS
  products.
- A fresh saved CWL equivalence baseline was run on 2026-07-01 at
  `/tmp/rapthor-equivalence-20260701-current` inside the dev container. All
  current-contract scenarios passed product comparison. The only warning was an
  expected non-scientific output-record summary difference for DI full-Jones:
  the CWL reference stored the full-Jones solution under legacy
  `fast_phase_solutions`, while the current cleaned contract records it as
  `fulljones_solutions`.
- Architecture docs and structure docs describe the current execution-owned
  module layout.
- Dev containers install docs dependencies by default.

Recent verification:

- `tests/execution/test_predict_measurement_sets.py`
- `tests/execution/test_predict_sector_models.py`
- `tests/execution/test_predict_flow.py`
- `tests/execution/test_run_names.py`
- `tests/architecture/test_import_boundaries.py`
- `python3 scripts/dev/run_saved_cwl_equivalence.py --run-root /tmp/rapthor-equivalence-20260701-current --stop-on-failure`
- `tox -e lint`
- Sphinx HTML build in the dev container, with existing documentation warnings
  tracked as docs cleanup rather than a blocked build.

Known caveats:

- Avoid running multiple pytest processes in parallel unless each run has a
  separate `RAPTHOR_TEST_RUN_ROOT`.
- Prefect can emit late logging shutdown warnings after passing flow tests.
  Track separately only if it becomes noisy in CI.
- Do not use the default `/tmp` run root for saved CWL equivalence on a nearly
  full container filesystem. WSClean writes large intermediate FITS products and
  can fail with CFITSIO write errors before scientific comparisons run.
- Pydantic remains a future option for configuration/payload validation. Keep
  contracts and builders clean enough that adopting it later would be
  incremental rather than a rewrite.

## Next Work Queue

### 1. Regression Guards For Scientific Contracts

Add focused integration/regression checks that protect the scientific behaviour
we stabilized during the migration.

Current coverage added:

- Solve-slot order, solution labels, per-chunk h5parm prefixes, and collected
  h5parm filenames are locked in focused operation tests.
- DD and DI calibration initial-solution handoff now checks that only current
  cycle h5parms are reused, including full-Jones DI solutions.
- Existing finalize tests cover final DD/DI solution destinations and auxiliary
  public solution products.
- Existing integration tests cover representative DD, DI, mixed DI/DD,
  full-Jones, slow-gain, and calibration-option command contracts.

Tasks:

- Add one focused integration assertion that a later cycle does not apply a
  previous-cycle h5parm in the emitted DP3 commands.
- Keep the saved CWL equivalence runner available as the heavier confidence
  check for larger scientific or script-migration changes.

Done when:

- Calibration strategy/output regressions fail in focused tests before they
  require a full manual equivalence investigation.

### 2. Builder And Helper Cleanup

Reduce small pockets of repeated structure that remain after the script and
flow migrations.

Tasks:

- Review repeated payload-builder and command-builder patterns in:
  - `rapthor.execution.image.builders`
  - `rapthor.execution.image.commands`
  - `rapthor.execution.calibrate.builders`
  - `rapthor.execution.calibrate.commands`
  - the smaller predict, mosaic, and concatenate command modules
- Extract only the shared pieces that remove real duplication without hiding
  scientific intent or making command construction harder to read.
- Prefer small option dataclasses or local helper functions for stable argument
  groups; keep Pydantic as a future validation option rather than introducing it
  during this cleanup.
- Run a focused dead-code scan for private helpers in `rapthor.execution` and
  `rapthor.operations`, especially compatibility leftovers and helpers only
  referenced by tests.
- Remove unused private helpers when production code no longer calls them, and
  move test-only setup into tests rather than keeping it in production modules.
- Add or update focused unit tests for any extracted builder/helper behavior so
  command strings, payload contracts, and scientific defaults stay locked.

Done when:

- Repeated builder logic has either been removed or intentionally left in place
  with a clear readability reason.
- No obvious unused private helpers remain in execution or operation modules.
- Command and payload tests still describe the production contracts directly.

### 3. Runtime Bootstrap For Prefect And Dask

Make `rapthor input.parset` succeed predictably whether or not the user has an
existing Prefect server or Dask cluster.

Tasks:

- Add a small runtime bootstrap layer before `pipeline_flow` starts:
  - use an existing Prefect API when `PREFECT_API_URL` or a parset value is set
    and reachable
  - otherwise allow Prefect to use its temporary local API/server
  - expose an explicit mode such as `prefect_api_mode = auto|external|ephemeral`
- Extend execution config and defaults for Prefect API selection without
  breaking the existing `rapthor input.parset` CLI.
- Preflight the runtime before launching long work:
  - check external Prefect API health when configured
  - check external Dask scheduler address and worker count
  - report local Dask worker/thread settings when no external scheduler is used
  - fail early with actionable messages for unreachable servers or schedulers
- Keep Dask task-runner selection simple:
  - `dask_scheduler` or `DASK_SCHEDULER` means `external_dask`
  - no scheduler means `local_dask`
  - explicit `prefect_task_runner` still overrides auto-selection
- Prefer one visible Dask scheduler for the whole `rapthor input.parset` run
  instead of short-lived per-operation local clusters, so the dashboard shows
  the full task stream.
- Add local worker sizing that is separate from node count, e.g.
  `local_dask_workers`, so single-node runs can use several workers without
  pretending those workers are separate nodes.
- Add tests for the startup matrix:
  - no Prefect server and no Dask cluster
  - existing Prefect server and no Dask cluster
  - no Prefect server and existing Dask cluster
  - existing Prefect server and existing Dask cluster
- Document the supported runtime modes and environment variables in the user
  docs.
- Add clear copy/paste quick-start docs with the lowest-friction commands for:
  - running Rapthor with the built-in ephemeral Prefect API and local Dask
  - starting a persistent local Prefect server and exporting `PREFECT_API_URL`
  - starting a local Dask scheduler/workers and setting `dask_scheduler` or
    `DASK_SCHEDULER`
  - using an existing Prefect server and Dask cluster on shared infrastructure

Done when:

- `rapthor input.parset` has a clear, tested runtime contract for local runs,
  external Prefect, local Dask, and external Dask.
- Startup failures happen before expensive pipeline work and explain exactly
  what the user should fix.
- A new user can copy commands from the docs and run Rapthor locally with
  minimal setup friction.

### 4. Dask Scalability Contracts

Prove that the pipeline can scale across multiple workers or nodes without
accidentally passing domain objects, huge nested state, or local-only paths.

Current findings:

- The Dask dashboard can look quiet because the current flow tasks are coarse:
  one imaging task per sector, one calibration task per time chunk, one mosaic
  task per image type, one concatenate task per epoch, and one predict task per
  model-data command or post-process group.
- Top-level pipeline operation ordering is intentionally sequential because
  later operations depend on field state and products from earlier operations.
  Parallelism should therefore be improved inside operation flows first.
- Long-running external tools such as DP3, WSClean, IDG, and PyBDSF should stay
  coarse resource-managed tasks. Dask should orchestrate around them rather
  than trying to split their internal work.

Tasks:

- Add payload-size and serialization guard tests for:
  - image sector tasks
  - calibration chunk tasks
  - predict model/post-process tasks
  - mosaic image tasks
  - concatenate epoch tasks
- Add tests that assert each flow submits the intended task units.
- Check that all worker payloads are plain serializable data, not `Field`,
  `Observation`, `Sector`, or operation instances.
- Extend resource-request coverage beyond image WSClean MPI paths.
- Split image-sector orchestration into clearer Dask task boundaries:
  - prepare one imaging Measurement Set per observation
  - concatenate prepared Measurement Sets
  - run or reuse WSClean
  - filter source/skymodel products
  - run diagnostics
  - build image cubes and normalization products
  - compress final images when requested
- Split mosaic orchestration into template, per-sector regrid, final mosaic,
  and optional compression tasks.
- Let predict post-processing for an observation start as soon as that
  observation's model-data outputs are ready, instead of waiting for all model
  outputs globally.
- Refine the current Prefect task names as finer task boundaries land, replacing
  generic indexes with stable sector, chunk, observation, image type, or epoch
  identifiers where those identifiers are available in payloads.
- Keep task granularity practical: do not split DP3, WSClean, IDG, or PyBDSF
  internals into Dask subtasks unless a proven library-level integration exists.
- Document which steps are distributed by Dask and which still run as coarse
  external commands or execution-owned module adapters.

Done when:

- Tests and docs make the Dask task boundaries visible.
- A developer can see what data each boundary receives.
- The tests fail if a future refactor starts sending rich domain objects or
  oversized payloads to workers.
- A representative demo run shows meaningful task-stream activity in the Dask
  dashboard without oversubscribing threaded or MPI external tools.

### 5. Runtime UX: Dry Run And Preflight

Make it easier for users to understand likely runtime failures before launching
a long pipeline run.

Tasks:

- Expand dry-run output to show:
  - planned operation order
  - task groups
  - resource hints
  - expected outputs
  - external tools and execution-owned module adapters
  - unsupported multi-node features
- Improve preflight messages for:
  - missing external tools
  - unsupported container configuration
  - Slurm/external-Dask mismatch
  - missing Dask scheduler
  - MPI WSClean assumptions
- Keep dry-run and preflight code independent of Prefect task objects where
  possible.

Done when:

- A user can run a preflight/dry-run path and understand likely runtime failures
  without reading flow code.

### 6. Contributor Documentation

Add short docs/checklists for common changes:

- adding a parset option
- modifying an operation
- adding an external command helper
- adding an execution-owned module adapter
- adding a new flow task boundary
- converting a legacy utility to an importable module

Include fast test lanes for each change type and point contributors to the
owner module for payloads, commands, outputs, operation adapters, and Prefect
flow wiring.

### 7. Deferred Targeted Refactors

Do not split these modules just for tidiness. Split them when changing behavior
or when a smaller extraction clearly reduces risk:

- `rapthor.execution.image.diagnostic_calculation`
  - later split into photometry, astrometry, plotting, and orchestration helpers
- `rapthor.execution.image.flux_normalization`
  - later split into catalog loading, source matching, SED fitting, and h5parm
    writing helpers
- `rapthor.execution.calibrate.h5parm_combination`
  - later move toward named combination strategies

Keep generated local noise (`__pycache__`, `.tox`, `.ruff_cache`, `runs`,
`htmlcov`, build outputs) out of repo decisions; clean locally when useful, but
do not treat it as source structure.
