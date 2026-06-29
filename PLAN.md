# Rapthor Architecture Refactor Plan

Status snapshot: 2026-06-29.

## Goal

Make the Prefect/Dask Rapthor pipeline easy to understand, extend, test, debug,
and scale while preserving the user-facing CLI workflow:

```bash
rapthor input.parset
```

This Prefect/Dask implementation has not been released, so prefer the clean
production architecture over unreleased Python API compatibility, migration
aliases, compatibility shims, or test-only production surfaces.

## Current State

The main architecture cleanup is largely complete.

Completed foundations:

- Execution code is organized by owner package:
  `image`, `calibrate`, `concatenate`, `predict`, `mosaic`, and `pipeline`.
- Image and calibration operation adapters are package-based.
- Migrated helper-script logic lives in importable execution modules.
- Production flows call migrated Python helpers directly, except where shell
  isolation is still useful for external tools or third-party multiprocessing.
- Broad execution facades, normalized command wrappers, migration shims, and
  unused runtime abstractions have been removed.
- Command builders are deterministic and tested.
- Image and calibration command builders use option dataclasses where argument
  groups are stable.
- Payload contracts and validation live with operation-specific execution code.
- Scheduler-independent work units are separated from Prefect flow wiring for
  the complex image and calibration paths.
- Architecture boundary tests exist and should be tightened as the final
  script-entrypoint cleanup lands.

Known follow-ups:

- Run a Sphinx docs build once the docs environment has `sphinx` installed.
- Avoid running multiple pytest processes in parallel unless each run has a
  separate `RAPTHOR_TEST_RUN_ROOT`.
- Prefect can emit late logging shutdown warnings after passing flow tests;
  track separately only if it becomes noisy in CI.

## Immediate Next Work

Finish the remaining script-entrypoint cleanup before starting new scalability
work.

### 1. Move Remaining Shell Adapters Into Execution

Replace the remaining legacy executable/script entry points with
execution-owned module adapters.

- Done: `filter_skymodel.py` now runs through
  `python -m rapthor.execution.image.skymodel_filter_cli`, with the production
  logic still in `rapthor.execution.image.skymodel_filter.filter_image_skymodel`.
- Done: removed `rapthor/scripts/mpi_runner.sh`; there were no runtime
  references beyond packaging metadata.
- Done: removed the now-empty `rapthor/scripts` package marker and stale
  `scripts/*` package-data entry.
- Move `bin/plotrapthor` to execution-owned calibration plotting modules:
  - move plotting logic to `rapthor.execution.calibrate.plotting.plot_solutions`
  - add a small CLI adapter such as `rapthor.execution.calibrate.plotting_cli`
  - update `build_plot_solutions_command` to call the adapter with
    `python -m rapthor.execution.calibrate.plotting_cli ...`
  - move `tests/scripts/test_plotrapthor.py` into execution/calibration coverage
  - update command reference fixtures and remove `bin/plotrapthor` from package
    metadata

### 2. Final Script-Migration Polish

After the remaining entry point moves:

- Move or remove remaining `tests/scripts` files once their coverage lives under
  `tests/execution`.
- Add a shared command helper for `python -m rapthor.execution...` module
  adapters so filter and plotting commands are built consistently.
- Tighten architecture tests so production code, package metadata, and command
  fixtures cannot reintroduce retired helper scripts or `plotrapthor`.
- Replace script-era `print()` calls in execution helpers with logging,
  starting with:
  - `rapthor.execution.predict.sector_model_addition`
  - `rapthor.execution.predict.sector_model_subtraction`
  - `rapthor.execution.calibrate.h5parm_sources`
- Make adapter CLI defaults delegate to execution helper defaults where
  possible, so CLI parsing does not drift from the importable API.
- Decide whether `bin/concat_linc_files` is a supported utility or should move
  to the same module-adapter pattern.

### 3. Documentation Update

Update `docs/source/development/architecture.rst` so it reflects the current
architecture:

- `rapthor.scripts` is no longer a production pipeline layer.
- Migrated helper logic lives under `rapthor.execution.<owner>`.
- PyBDSF/lsmtool filtering may still run in a subprocess, but via an
  execution-owned adapter rather than a legacy script wrapper.
- Calibration solution plotting is owned by `rapthor.execution.calibrate` and
  may still run through shell execution via a module adapter.

### 4. Focused Verification

Run these after the entrypoint cleanup:

- `tests/architecture/test_import_boundaries.py`
- `tests/execution/test_image_flow.py`
- `tests/execution/test_calibrate_flow.py`
- moved filter-skymodel execution tests
- moved plot-solution execution tests
- `tests/integration/test_rapthor_restart.py` when the integration environment
  is available
- Sphinx build once the docs environment has `sphinx`

## Next Refactor Slice

Run this bounded maintainability pass before adding new Dask scalability
assertions.

### Payload Boundaries

Split the largest payload modules into clearer contracts, builders, and
validation modules:

- `rapthor.execution.image.payloads`
- `rapthor.execution.calibrate.payloads`

Target shape:

- `contracts.py`: typed payload aliases/classes and public payload schemas
- `builders.py`: parset/field/operation-to-payload mapping
- `validation.py`: payload validation and serialization checks

Keep imports direct and explicit. Avoid adding broad package-level facades.

### Predict Shared Helpers

Extract common Measurement Set helper code shared by:

- `rapthor.execution.predict.sector_model_addition`
- `rapthor.execution.predict.sector_model_subtraction`

Focus only on real duplication:

- chunk sizing and memory assumptions
- output naming
- table open/close handling
- common logging messages

### Defer Larger Scientific Splits

Do not split these modules just for tidiness. Split them when changing behavior
or when a smaller extraction clearly reduces risk:

- `rapthor.execution.image.diagnostic_calculation`
  - later split into photometry, astrometry, plotting, and orchestration helpers
- `rapthor.execution.image.flux_normalization`
  - later split into catalog loading, source matching, SED fitting, and h5parm
    writing helpers
- `rapthor.execution.calibrate.h5parm_combination`
  - later move toward named combination strategies

## Dask Scalability Contracts

After the entrypoint and payload cleanup, prove the pipeline can scale across
multiple workers/nodes without accidentally passing domain objects, huge nested
state, or local-only paths.

Tasks:

- Add payload-size and serialization guard tests for:
  - image sectors
  - calibration chunks
  - predict model tasks
  - mosaic image types
  - concatenate epochs
- Add tests that assert each flow submits the intended task units.
- Extend resource-request coverage beyond image WSClean MPI paths.
- Check that all worker payloads are plain serializable data, not `Field`,
  `Observation`, `Sector`, or operation instances.
- Document which steps are distributed by Dask and which still run as coarse
  external commands or execution-owned module adapters.

Done when:

- A developer can see, from tests and docs, what the Dask task boundaries are
  and what data each boundary receives.

## Runtime UX

Once scalability contracts are clearer, improve dry-run and preflight behavior.

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

## Contributor Documentation

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

## Target Environment Validation

Define opt-in validation runs for:

- external Dask
- Slurm-launched Dask
- MPI WSClean
- `prefect_command_profile = perf`

Keep these separate from the default non-integration suite. Document required
system tools, scheduler assumptions, and expected runtime artifacts.

## Testing Lanes

Use the narrowest lane that matches the change, then run a broader lane before
handing off larger slices.

Fast lanes:

- Architecture boundaries: `tests/architecture`
- Payload contracts: `tests/execution/test_payloads.py` and relevant flow tests
- Command builders: `tests/execution/test_commands.py`,
  `tests/execution/test_reference_fixtures.py`, or operation-specific flow tests
- Prefect wiring: relevant `tests/execution/test_*_flow.py`
- Operation adapters: relevant `tests/operations`

Broader lanes:

- Image execution: `tests/execution/test_image_flow.py`
- Calibration execution: `tests/execution/test_calibrate_flow.py`
- Smaller flow suites for predict, mosaic, and concatenate
- Selected `tests/integration` files only when required tools are available

Use the dev container for tests and formatting.

## Architecture Rules

- Preserve the CLI/parset user workflow.
- Do not preserve unreleased Python APIs, aliases, or compatibility shims unless
  they protect the CLI or a documented scientific contract.
- Keep `rapthor.lib` independent of Prefect, Dask, Slurm, shell execution, and
  artifact publication.
- Keep Prefect flows thin: task wiring, scheduling, runtime integration, and
  aggregation belong there.
- Keep operation adapters focused on lifecycle, reading live `Field` state,
  payload construction, flow invocation, and finalizer side effects.
- Keep command builders pure and deterministic.
- Keep Dask worker payloads plain and serializable.
- Use operation-owned packages for non-trivial execution code:
  - `commands.py`
  - `payloads.py`, or split `contracts.py`/`builders.py`/`validation.py`
  - `outputs.py` when useful
  - focused work-unit modules
  - `flow.py`
- Use operation-owned packages for non-trivial operation code:
  - `base.py`
  - `plan.py`
  - variant modules such as `initial.py` or `normalize.py`
  - diagnostics/finalizer helpers only when they have real ownership
- Prefer stdlib dataclasses for stable internal command argument groups.
- Keep Pydantic in mind as a future option for parset, payload, and boundary
  validation, but do not introduce it until a boundary has enough complexity to
  justify the dependency.
- Do not add broad facade exports or compatibility shims without a documented
  removal condition.
