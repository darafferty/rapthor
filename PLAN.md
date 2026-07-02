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

The main architecture cleanup and script-to-module migration are complete enough
to move into runtime bootstrap and scalability work.

Completed:

- Execution code is organized by owner package:
  `image`, `calibrate`, `concatenate`, `predict`, `mosaic`, and `pipeline`.
- Image and calibration operations are package-based adapters.
- Migrated helper-script logic lives in importable execution modules, and
  production flows call those modules directly except where shell isolation is
  still useful for external tools or third-party multiprocessing.
- Retired helper scripts and the old `plotrapthor` executable are guarded by
  architecture tests so production code and command fixtures do not reintroduce
  them.
- The installed `rapthor` command is exposed through `rapthor.cli:main`.
- `concat_linc_files` remains a supported installed utility through
  `rapthor.execution.concatenate.linc_cli:main`.
- Broad execution facades, normalized command wrappers, migration shims, and
  unused runtime abstractions have been removed.
- Payload contracts, builders, validation, commands, outputs, and flow wiring
  live with the operation-specific execution code.
- Command builders are deterministic and tested.
- Image and calibration command builders use option dataclasses where argument
  groups are stable.
- Predict sector-model add/subtract share Measurement Set mechanics through
  `rapthor.execution.predict.measurement_sets`, with direct unit coverage.
- Scheduler-independent work units are separated from Prefect flow wiring for
  the complex image and calibration paths.
- Prefect flow and task run names include operation, calibration mode where
  relevant, cycle, and coarse task identifiers so the dashboard is easier to
  scan during rich demo and integration runs.
- The development architecture docs include a current Prefect/Dask
  orchestration diagram matching the refactored owner-package layout.
- `calibration_strategy` is the only production interface for solve type and
  solve order. Legacy `do_fulljones_solve` and `do_slowgain_solve` flags are
  retired from production configuration.
- The default DD strategy is explicit:
  `{"dd": ["fast_phase", "medium_phase", "slow_gains", "medium_phase"], "di": []}`.
- Builder and helper cleanup is complete for the current pass:
  - shared payload validators are reused directly instead of being aliased as
    private local helpers
  - repeated payload validation for optional strings, basenames, string lists,
    optional file records, and required lists is centralized
  - non-obvious nested payload checks have concise docstrings
  - focused payload/contract tests cover the extracted behavior
  - no obvious compatibility-only private helpers remain in execution or
    operation modules from this cleanup pass
- The saved CWL equivalence helper now tolerates the observed sub-percent
  WSClean beam-fit jitter in image-cube beam sidecars while still comparing
  image data, h5parm solutions, sky models, operation markers, and output
  record shapes.
- Architecture docs and structure docs describe the current execution-owned
  module layout.
- Dev containers install docs dependencies by default.

## Recent Verification

Run in the dev container on 2026-07-01:

- `tox -e lint`
- Non-integration tests using the tox split against the prepared dev-container
  Python environment:
  - `tests/lib/test_field.py`: 26 passed
  - Prefect-marked non-integration tests: 42 passed
  - Remaining non-integration tests with xdist: 928 passed, 2 xfailed
- Full integration suite with generated runs on `/tmp`:
  - 25 passed, 1 skipped Slurm-only check, 1 expected xfail
- Saved CWL equivalence matrix:
  - all current-contract scenarios passed
  - report: `/tmp/rapthor-equivalence-20260701-builder-cleanup-final/equivalence-report.json`
- Rich Prefect/Dask demo:
  - command completed successfully with `examples/generated/prefect_demo_rich/prefect_demo_rich.parset`
  - run directory: `/tmp/rapthor-prefect-demo-20260701-builder-cleanup`
  - local Dask cluster started with 2 workers, Rapthor finished, and the helper
    shut down Dask and Prefect cleanly

Notes:

- `tox -e py310` is not currently a useful local verification path in this dev
  container because tox creates an isolated environment and tries to build
  `python-casacore` and `everybeam` without Casacore headers. Use the prepared
  dev-container Python environment for full local verification unless the tox
  environment is taught to use the container's system dependencies.
- Keep large integration, equivalence, and demo run roots on `/tmp` or another
  spacious filesystem. The workspace mount can fill quickly with WSClean FITS
  products.

## Known Caveats

- Avoid running multiple pytest processes in parallel unless each run has a
  separate `RAPTHOR_TEST_RUN_ROOT`.
- Prefect can emit late logging shutdown warnings after passing flow tests.
  Track separately only if it becomes noisy in CI.
- Pydantic remains a future option for configuration/payload validation. Keep
  contracts and builders clean enough that adopting it later would be
  incremental rather than a rewrite.

## Next Work Queue

### 1. Runtime Bootstrap For Prefect And Dask

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

### 2. Dask Scalability Contracts

Prove that the pipeline can scale across multiple workers or nodes without
accidentally passing domain objects, huge nested state, or local-only paths.

Tasks:

- Add payload-size and serialization guard tests for image, calibration,
  predict, mosaic, and concatenate task payloads.
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
  observation's model-data outputs are ready.
- Refine Prefect task names as finer task boundaries land, replacing generic
  indexes with stable sector, chunk, observation, image type, or epoch
  identifiers where available.
- Keep task granularity practical: do not split DP3, WSClean, IDG, or PyBDSF
  internals into Dask subtasks unless a proven library-level integration exists.
- Document which steps are distributed by Dask and which still run as coarse
  external commands or execution-owned module adapters.

Done when:

- Tests and docs make the Dask task boundaries visible.
- A developer can see what data each boundary receives.
- Tests fail if a future refactor sends rich domain objects or oversized
  payloads to workers.
- A representative demo run shows meaningful task-stream activity in the Dask
  dashboard without oversubscribing threaded or MPI external tools.

### 3. Runtime UX: Dry Run And Preflight

Make it easier for users to understand likely runtime failures before launching
a long pipeline run.

Tasks:

- Expand dry-run output to show planned operation order, task groups, resource
  hints, expected outputs, external tools, execution-owned module adapters, and
  unsupported multi-node features.
- Improve preflight messages for missing external tools, unsupported container
  configuration, Slurm/external-Dask mismatch, missing Dask scheduler, and MPI
  WSClean assumptions.
- Keep dry-run and preflight code independent of Prefect task objects where
  possible.

Done when:

- A user can run a preflight/dry-run path and understand likely runtime failures
  without reading flow code.

### 4. Contributor Documentation

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

### 5. Deferred Targeted Refactors

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
`htmlcov`, build outputs, temporary integration/equivalence/demo roots) out of
repo decisions; clean locally when useful, but do not treat it as source
structure.
