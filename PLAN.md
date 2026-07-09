# Rapthor Architecture Refactor Plan

Status snapshot: 2026-07-09.

## Goal

Make the Prefect/Dask Rapthor pipeline easy to run, understand, test, debug,
benchmark, and scale while preserving the user-facing CLI workflow:

```bash
rapthor input.parset
```

The Prefect/Dask implementation has not been released, so prefer clean
production architecture over unreleased compatibility shims.

## Current State

The migration/refactor is now in the guarded scalability phase.

Completed and accepted:

- Execution code is organized by owner package:
  `image`, `calibrate`, `concatenate`, `predict`, `mosaic`, and `pipeline`.
- Operation adapters are thin; command builders, payload validation, output
  discovery, migrated helper logic, and flow wiring live under
  `rapthor/execution/<owner>/`.
- Runtime bootstrap, local/external Dask setup, CLI smoke coverage, readable
  run names, dashboard/report logging, and runtime parset materialization are
  in place.
- Master feature catch-up is complete for the known runtime/product behavior:
  MS time ordering, reset-directory guards, astrometry-corrected products,
  per-facet RMS diagnostics, residual visibilities, WSClean prediction,
  normalization semantics, parallel gridding, and shared-facet behavior.
- The flexible calibration strategy contract is implemented:
  `calibration_strategy` controls solve type/order; previous-cycle products
  may seed matching solves only as optimizer seeds when product role, cycle, and
  DD direction compatibility are valid.
- Image-only application semantics are covered: DI scalar phase, DI diagonal
  slow-gain, and DI full-Jones products are pre-applied; DD products are applied
  on the fly during imaging when matching directions are available.
- Science-equivalence evidence is summarized in `EQUIVALENCE_REPORT.md`, with
  historical compact reports under `docs/source/development/equivalence_runs/`.
- Benchmark scaffolding exists for CI-sized profiling, Dask report parsing,
  command timing, operation-boundary timing, JSON summaries, and Markdown
  reports.
- Image-sector execution has the accepted task split:
  `prepare -> filter_skymodel -> calculate_image_diagnostics -> finalize`.
- Post-split benchmarks support `filter_skymodel_ncores=15` while keeping the
  global `2x30`, `max_threads=30` resource shape. The default has been promoted
  and confirmed against the explicit `filter-only-15` profile. Explicit
  `filter_skymodel_ncores = 0` still means "use `max_threads`".
- The `calculate_image_diagnostics` task split has been benchmarked and
  accepted. It added the expected four Dask tasks, kept scheduler gap flat, and
  preserved successful command execution.

Keep in mind:

- Use the prepared dev-container Python environment for formatting, tests,
  demo runs, integration checks, equivalence checks, and benchmarks.
- Keep raw run directories, FITS/MS products, full Dask HTML reports, logs,
  `.tox`, `.ruff_cache`, `htmlcov`, and build products out of git.
- Compact curated reports, manifests, and short command logs may be tracked
  under `docs/source/development/` when they explain an important result.
- Preview artifacts are diagnostic aids only. Raw FITS/h5parm products,
  numeric diagnostics, catalogs, region files, command records, and report JSON
  remain the scientific contract.

## Active Queue

Do these in order unless a regression blocks progress.

1. **Add benchmark scenarios for currently hidden scaling paths.**
   Add or enable small repeatable profiles before changing those paths:
   flux-scale normalization / image cubes, WSClean-predict calibration,
   many-sector imaging, mosaic-heavy runs, and larger/multi-node runs. Keep
   preview artifacts disabled unless the scenario explicitly measures report
   overhead.

2. **Measure before splitting the next candidates.**
   Candidates are:
   `normalize_flux_scale` / `make_catalog_from_image_cube`,
   calibration post-processing (`collect_h5parms`, `process_slow_gains`,
   full-Jones normalization, `combine_h5parms`, `plot_solutions`),
   WSClean-predict loops, and mosaic per-sector regridding. Split only when a
   benchmark, dashboard trace, or failure mode shows the boundary is useful.

3. **Build the scalability/performance equivalence gate.**
   Compare current branch and master with identical inputs, resource shape,
   preview settings, run roots, and science checks. Start advisory: fail only
   on infrastructure errors, missing outputs, failed runs, or science
   equivalence failures; report performance as pass/warn/fail bands until
   variance is characterized.

4. **Guard the science-equivalence contract.**
   For documentation, preview-artifact, benchmark-report, or refactor-only
   changes, run focused tests. For calibration, prediction, imaging, h5parm,
   FITS, catalog, sky-model, or product-record changes, rerun the relevant
   saved-reference and branch-vs-master scenarios before judging the change.

5. **Polish runtime UX and contributor docs after the next scalability result.**
   Keep `TESTING.md`, `.agents/testing_playbook.md`, `AGENTS.md`, runtime docs,
   and this plan aligned. Improve preflight/dry-run output, missing-tool
   messages, runtime dashboard/resource summaries, and debugging docs as the
   runtime surface settles.

## Testing And Regression Workstream

Keep this workstream visible because the test suite is part of the refactor,
not just a final check.

- Add focused tests with each new task boundary: payload shape,
  serializability, task/run names, output records, restart markers, and command
  records.
- Keep science-regression coverage explicit for calibration strategy behavior,
  image-only cycles, DI pre-apply, DD on-the-fly apply, previous-cycle solution
  handling, and master feature catch-up cases.
- Add architecture tests only for contracts that should not drift silently:
  owner-package boundaries, thin operation adapters, runtime payload
  serializability, and benchmarkable task structure.
- Keep integration tests representative rather than exhaustive. Prefer a small
  set of scenario tests that protect scientific behavior and user workflows.
- Treat science-equivalence and performance-equivalence workflows as release
  gates, not everyday unit tests. Keep their reports easy to regenerate and
  easy for reviewers to interpret.
- When CI fails, improve the smallest useful test or fixture rather than adding
  broad slow coverage.

## Task-Boundary Policy

Make a step a Prefect task when it is slow, optional, scientifically meaningful,
externally resource-hungry, independently benchmarkable, or likely to fail in a
way users need to identify quickly.

Keep these as plain Python:

- validation
- payload construction
- deterministic command builders
- output-record assembly
- path discovery
- tiny helper functions

Do not split for Dask alone. A NumPy call inside a Prefect task runs as normal
NumPy in that worker process. Dask helps when Rapthor submits independent tasks
that can run concurrently, when external tools are given the right thread/core
limits, or when code is explicitly rewritten around Dask-aware collections such
as `dask.array` or carefully chunked delayed work. Avoid adding scheduler
overhead or oversubscribing DP3, WSClean, IDG/IDGCal, PyBDSF, or plotting
helpers.

Task naming:

- Use flow run names for operation context such as operation type, mode, and
  cycle: `calibrate_dd_4`, `predict_di_1`, `image_3`.
- Use short task definition names and task run names because the enclosing
  flow/subflow already provides context. Do not prefix task run names with the
  parent flow name.
- Prefer established legacy workflow vocabulary when still scientifically
  accurate: `filter_skymodel`, `calculate_image_diagnostics`,
  `combine_h5parms`, `collect_h5parms`, `process_slow_gains`,
  `plot_solutions`, `predict_model_data`, `make_mosaic`.
- Add only the smallest useful discriminator when several sibling tasks of the
  same kind can run in the same flow: `sector_1_filter_skymodel`,
  `chunk_1`, `screen_1`, `model_1`, `postprocess_1`.
- Prefer scientific labels over numerical suffixes when they are stable and
  meaningful, for example `image_type_I` instead of `image_type_1`.

## Benchmark And Equivalence Evidence

Use these as the historical record instead of expanding this plan:

- `EQUIVALENCE_REPORT.md`
- `docs/source/development/science_equivalence_contract.rst`
- `docs/source/development/performance_equivalence_contract.rst`
- `docs/source/development/equivalence_runs/`
- `docs/source/development/benchmark_baselines/`

Most relevant benchmark reports:

- `docs/source/development/benchmark_baselines/2026-07-08-default-filter-skymodel-confirmation.md`
- `docs/source/development/benchmark_baselines/2026-07-09-diagnostics-task-split.md`
- `docs/source/development/benchmark_baselines/2026-07-07-ci-benchmark-comparison.md`
- `docs/source/development/benchmark_baselines/2026-07-08-filter-skymodel-resource-profiles.md`
- `docs/source/development/benchmark_baselines/2026-07-08-filter-skymodel-only-profile.md`
- `docs/source/development/benchmark_baselines/2026-07-08-post-split-filter-skymodel-profile.md`

Benchmark decision rules:

- Compare absolute wall times only within paired runs on the same runner.
- Prefer medians over individual repetitions, but keep min/max visible.
- Track wall time, command totals, operation-minus-command gap,
  Dask duration-minus-compute gap, task count, task groups, worker/thread
  shape, and return codes.
- Do not use wall-clock thresholds as unit or architecture tests. Tests should
  protect benchmarkable runtime shape, not elapsed seconds.

## Verification Routine

After code changes:

```bash
python3 -m ruff check --fix --select I <touched-python-files>
python3 -m ruff format <touched-python-files>
python3 -m pytest <focused-tests>
```

Before merging significant runtime/science changes:

```bash
python3 -m pytest -m "not integration" tests
RAPTHOR_TEST_RUN_ROOT=/tmp/rapthor-integration-runs \
  python3 -m pytest -m integration -vv -ra --durations=0 \
  tests/integration tests/operations/integration
```

Use `scripts/dev/run_saved_cwl_equivalence.py` and branch-vs-master
equivalence/performance workflows after scientific logic changes, script/module
product changes, h5parm/FITS/catalog/skymodel changes, or performance-sensitive
task-boundary changes.

## Deferred Refactors

Do not split these modules for tidiness alone. Split them when changing
behavior or when profiling shows a real maintenance, observability, or
performance reason:

- `rapthor.execution.image.diagnostic_calculation`
- `rapthor.execution.image.flux_normalization`
- `rapthor.execution.calibrate.h5parm_combination`
- `rapthor.operations.calibrate.base`
- `rapthor.operations.image.base`
