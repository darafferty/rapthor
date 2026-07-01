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
- Retired helper scripts and `bin/plotrapthor` are guarded by architecture
  tests so production code and command fixtures do not reintroduce them.
- The installed `rapthor` command is exposed through `rapthor.cli:main`.
- `bin/concat_linc_files` remains a supported utility through the package-owned
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
- Architecture docs and structure docs describe the current execution-owned
  module layout.
- Dev containers install docs dependencies by default.

Recent verification:

- `tests/execution/test_predict_measurement_sets.py`
- `tests/execution/test_predict_sector_models.py`
- `tests/execution/test_predict_flow.py`
- `tests/architecture/test_import_boundaries.py`
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

Tasks:

- Check solve-slot order and h5parm filenames for explicit calibration
  strategies.
- Check final h5parm products used by later operations.
- Check auxiliary solution products that are intentionally public.
- Check current-cycle-only solution handoff between calibration cycles.
- Keep the saved CWL equivalence runner available as a heavier confidence check
  for larger scientific or script-migration changes.

Done when:

- Calibration strategy/output regressions fail in focused tests before they
  require a full manual equivalence investigation.

### 2. Dask Scalability Contracts

Prove that the pipeline can scale across multiple workers or nodes without
accidentally passing domain objects, huge nested state, or local-only paths.

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
- Document which steps are distributed by Dask and which still run as coarse
  external commands or execution-owned module adapters.

Done when:

- Tests and docs make the Dask task boundaries visible.
- A developer can see what data each boundary receives.
- The tests fail if a future refactor starts sending rich domain objects or
  oversized payloads to workers.

### 3. Runtime UX: Dry Run And Preflight

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
`htmlcov`, build outputs) out of repo decisions; clean locally when useful, but
do not treat it as source structure.
