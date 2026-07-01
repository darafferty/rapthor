# Rapthor Architecture Refactor Plan

Status snapshot: 2026-06-30.

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
- Focused calibration integration tests pass for DD, DI, mixed DI/DD ordering,
  full-Jones, slow-gain, and calibration-option scenarios.
- The saved CWL equivalence matrix passes against the current scientific
  contract when run on a filesystem with enough space for WSClean FITS
  products.
- Architecture boundary tests exist and should be tightened as the final
  script-entrypoint cleanup lands.

Known follow-ups:

- Avoid running multiple pytest processes in parallel unless each run has a
  separate `RAPTHOR_TEST_RUN_ROOT`.
- Prefect can emit late logging shutdown warnings after passing flow tests;
  track separately only if it becomes noisy in CI.

## Immediate Next Work

Retire the remaining legacy calibration-strategy options, then complete the
remaining script-entrypoint cleanup before starting new scalability work.

### 0. Calibration Strategy Equivalence (Complete; Add Regression Guards)

Keep the current contract explicit: if a cycle should run a medium phase solve
after slow gains, the strategy must request it directly as:

```python
{"dd": ["fast_phase", "medium_phase", "slow_gains", "medium_phase"]}
```

- Done: current default/flexible/demo strategy examples use the explicit
  post-slow medium solve.
- Done: the saved CWL equivalence runner reconstructs the slow-gain strategy
  explicitly and skips the stale phase-only `dd_slow_gain_calibration`
  reference by default.
- Done: output-record shape changes are treated as metadata warnings in the
  saved CWL equivalence runner, and beam-table text products are compared
  numerically with tolerance.
- Done: resolved the `field-solutions-medium1-phase.h5` differences. The
  migrated operation was applying the fast-phase antenna constraint to the
  medium1 solve; medium1 is now unconstrained again, while fast phase and the
  post-slow medium2 solve keep the station constraint. Normalization and
  peeling equivalence now use explicit fast/medium calibration strategies.
- Done: the default saved CWL equivalence matrix passes, excluding the stale
  phase-only `dd_slow_gain_calibration` reference.
- Done: focused calibration integration tests passed in the dev container:
  `tests/integration/test_dd_calibration.py`,
  `tests/integration/test_di_calibration.py`, and
  `tests/integration/test_calibration_options.py`.
- Done: the saved CWL equivalence matrix passed under
  `/app/runs/equivalence-20260630-1950-current`. Do not use the default `/tmp`
  run root for this check on a nearly full container filesystem; WSClean writes
  large intermediate FITS products and can fail with CFITSIO write errors before
  scientific comparisons run.
- Add integration regression checks for calibration solve products once the
  remaining calibration cleanup lands, so future refactors catch:
  - solve-slot order and h5parm filenames for explicit calibration strategies
  - final h5parm products used by later operations
  - auxiliary solution products that are intentionally public
  - current-cycle-only solution handoff between calibration cycles

### 0a. Remove Remaining Calibration Slot Semantics (Complete)

The solve plan should be driven by explicit solve metadata, not by assumptions
that `solve1`, `solve2`, `solve3`, or `solve4` mean fast, medium, slow, or
second-medium. DP3 step names may remain slot-based, but scientific behavior and
output routing should use solve type/order metadata.

Done:

- `CalibrationSolve` carries explicit `solution_label` metadata and
  `medium_index` for medium-phase solves.
- Calibration finalization and collection use solve metadata instead of
  filename-prefix checks such as `medium2_*`.
- The `dd_phase_slow` keep-model rule is solve-type/order based rather than
  `slot in {2, 3}` based.
- Calibration payloads no longer map initial-solution inputs through historical
  slot-number fallbacks.
- Payload chunk counting uses the first active solve slot rather than assuming
  `output_solve1_h5parm`.
- Combined h5parm payload keys describe the phase-combination sequence
  (`phase_1_2`, `phase_1_2_3`) instead of old `solve1_solve2` layouts.
- Calibration payloads require explicit `solve{slot}_type` and
  `solve{slot}_solution_label`; filename-based solve-type inference has been
  removed.

### 0b. Retire Legacy Calibration Solve Flags (Complete)

`calibration_strategy` is now the only calibration interface for solve type and
order. The legacy `do_fulljones_solve` and `do_slowgain_solve` inputs are
retired from production configuration and stripped as deprecated parameters if
they are still present in old strategy files.

Done:

- Added an explicit default calibration strategy:
  `{"dd": ["fast_phase", "medium_phase", "slow_gains", "medium_phase"], "di": []}`.
- Removed legacy boolean strategy resolution from `Field.set_calibration_strategy()`.
- Updated self-calibration, generated demo strategies, examples, test
  strategies, and docs to set `calibration_strategy` directly.
- Replaced operation/payload fields named after legacy flags with derived solve
  metadata such as `has_slow_gain_solve`.
- Added/updated regression coverage for:
  - default DD solve order, including the explicit post-slow medium solve
  - default DI having no full-Jones solve unless requested explicitly
  - mixed DI/DD strategy ordering
  - ignored legacy boolean attributes on field stubs
- Verified in the dev container:
  - `tests/lib/test_field.py`
  - `tests/lib/test_strategy.py`
  - `tests/operations/test_calibrate.py`
  - `tests/execution/test_calibrate_flow.py`
  - `tests/execution/test_prefect_demo_script.py`
  - `tests/execution/test_prefect_demo_data_generator.py`
  - `tests/integration/test_dd_calibration.py`
  - `tests/integration/test_di_calibration.py`
  - `tests/integration/test_calibration_options.py`

### 1. Move Remaining Shell Adapters Into Execution

Replace the remaining legacy executable/script entry points with
execution-owned module adapters.

- Done: `filter_skymodel.py` now runs through
  `python3 -m rapthor.execution.image.skymodel_filter_cli`, with the production
  logic still in `rapthor.execution.image.skymodel_filter.filter_image_skymodel`.
- Done: removed `rapthor/scripts/mpi_runner.sh`; there were no runtime
  references beyond packaging metadata.
- Done: removed the now-empty `rapthor/scripts` package marker and stale
  `scripts/*` package-data entry.
- Done: moved `bin/plotrapthor` to execution-owned calibration plotting:
  `rapthor.execution.calibrate.plotting.plot_solutions` plus the
  `rapthor.execution.calibrate.plotting_cli` module adapter.
- Done: `build_plot_solutions_command` now calls the plotting adapter with
  `python3 -m rapthor.execution.calibrate.plotting_cli ...`.
- Done: plot-solution coverage now lives under `tests/execution`, command
  reference fixtures use the module adapter, and architecture tests reject
  retired `plotrapthor` references.

### 2. Final Script-Migration Polish

After the remaining entry point moves:

- Done: moved or removed remaining `tests/scripts` coverage once equivalent
  coverage lived under `tests/execution`.
- Done: added `rapthor.execution.commands.python_module_command` so filter,
  image-cube catalog, and plotting commands build `python3 -m ...` module
  adapters consistently.
- Done: tightened architecture tests so production code, package metadata, and
  command fixtures cannot reintroduce retired helper scripts or `plotrapthor`.
- Done: replaced script-era `print()` calls in production execution helpers
  with logging in:
  - `rapthor.execution.predict.sector_model_addition`
  - `rapthor.execution.predict.sector_model_subtraction`
  - `rapthor.execution.calibrate.h5parm_sources`
  - `rapthor.execution.concatenate.measurement_sets`
- Done: adapter CLI defaults delegate to execution helper defaults where
  possible, so CLI parsing cannot drift from the importable API.
- Done: `concat_linc_files` remains a supported utility, now through the
  package-owned CLI adapter `rapthor.execution.concatenate.linc_cli:main`.

### 3. Documentation Update (Complete)

Update `docs/source/development/architecture.rst` so it reflects the current
architecture:

- Done: `rapthor.scripts` is no longer documented as a production pipeline
  layer.
- Done: migrated helper logic is documented under `rapthor.execution.<owner>`.
- Done: PyBDSF/lsmtool filtering may still run in a subprocess, but via an
  execution-owned adapter rather than a legacy script wrapper.
- Done: calibration solution plotting is documented as owned by
  `rapthor.execution.calibrate`, with optional shell execution through a module
  adapter.
- Done: `python_module_command` is documented as the standard way to build
  execution-owned module adapter commands.

### 4. Focused Verification (Complete)

Run these after the entrypoint cleanup:

- Done: `tests/architecture/test_import_boundaries.py`
- Done: `tests/execution/test_image_flow.py`
- Done: `tests/execution/test_calibrate_flow.py`
- Done: moved filter-skymodel execution tests
- Done: moved plot-solution execution tests
- Run `tests/integration/test_rapthor_restart.py` when the integration environment
  is available
- Done: Sphinx HTML build passes in the dev container. The build currently
  emits existing documentation warnings; track those as docs cleanup rather
  than as a blocked build.

### 5. Repository Structure Modernization

Clean the remaining packaging, docs, and legacy utility surfaces before the
next payload/scalability refactor slice.

- Done: moved the main CLI implementation to `rapthor.cli:main` and exposed it
  with `[project.scripts]` so the installed user command remains `rapthor`.
- Done: removed `bin/rapthor`; `python -m rapthor.cli` is the source-tree
  fallback and shows the same `Usage: rapthor <parset>` help text.
- Done: added Sphinx, numpydoc, and the Snowball stemmer constraint to the docs
  and dev dependency sets so rebuilt dev containers can build documentation by
  default.
- Done: updated stale structure docs, especially `docs/source/code.rst`, so
  they no longer describe `bin/rapthor` or `rapthor/scripts` as active
  production layout.
- Done: converted `bin/concat_linc_files` to the package-owned
  `rapthor.execution.concatenate.linc_cli:main` entry point and removed the
  extensionless script from package metadata.
- Done: replaced the manual `[tool.setuptools].packages` list with setuptools
  package discovery for `rapthor*`.
- Keep the generated local noise (`__pycache__`, `.tox`, `.ruff_cache`, `runs`,
  `htmlcov`, build outputs) out of repo decisions; clean locally when useful,
  but do not treat it as source structure.

## Next Refactor Slice

Run this bounded maintainability pass before adding new Dask scalability
assertions.

### Payload Boundaries

Split the largest payload modules into clearer contracts, builders, and
validation modules:

- Done: split `rapthor.execution.image.payloads` into direct owner modules:
  `rapthor.execution.image.contracts`, `rapthor.execution.image.builders`, and
  `rapthor.execution.image.validation`.
- Done: split `rapthor.execution.calibrate.payloads` into direct owner modules:
  `rapthor.execution.calibrate.contracts`,
  `rapthor.execution.calibrate.builders`, and
  `rapthor.execution.calibrate.validation`.

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
