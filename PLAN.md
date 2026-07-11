# Rapthor Architecture Refactor Plan

Status snapshot: 2026-07-11.

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

Accepted foundation:

- Owner-package architecture is in place:
  `image`, `calibrate`, `concatenate`, `predict`, `mosaic`, and `pipeline`.
  Operation adapters are thin, while command builders, payload validation,
  output discovery, migrated helper logic, and flow wiring live under
  `rapthor/execution/<owner>/`.
- Runtime bootstrap, local/external Dask setup, CLI smoke coverage, readable
  flow names, dashboard/report logging, and runtime parset materialization are
  in place.
- Known master feature catch-up is complete for runtime/product behavior:
  time ordering, reset-directory guards, astrometry products, per-facet RMS
  diagnostics, residual visibilities, WSClean prediction, normalization,
  parallel gridding, and shared-facet behavior.
- The calibration contract is strategy-driven: `calibration_strategy` controls
  solve type/order; previous-cycle products may seed matching solves only as
  optimizer seeds when product role, cycle, and DD direction compatibility are
  valid.
- Image-only application semantics are covered: DI scalar phase, DI diagonal
  slow-gain, and DI full-Jones products are pre-applied; DD products are
  applied on the fly during imaging when matching directions are available.
- Science-equivalence evidence is summarized in `EQUIVALENCE_REPORT.md`, with
  historical compact reports under `docs/source/development/equivalence_runs/`.

Accepted performance and task-boundary evidence:

- Benchmark scaffolding exists for CI-sized profiling, Dask report parsing,
  command timing, operation-boundary timing, JSON summaries, and Markdown
  reports.
- The preferred automatic benchmark shape is currently `4x15`
  (`local_dask_workers=4`, `cpus_per_task=15`, `max_threads=15`). It is neutral
  for the default single-sector benchmark and faster for many-sector mosaics.
- Accepted task splits so far:
  image-sector preparation, image-sector `filter_skymodel`, image diagnostics,
  optional image post-processing tasks, calibration post-processing, and
  calibration image-based/WSClean prediction setup.
- Task observability polish is in place: shared task run-name/tag helpers,
  readable calibration chunk names (`solve_chunk_*`, `screen_chunk_*`),
  tool/runtime tags, task-runtime JSONL records, task-aware command/profile
  artifacts, and durable postage-stamp PNGs under cycle image directories.
- The calibration post-processing split is accepted. It moved the real work out
  of `finalize_solutions_task`; plotting is now the main visible
  post-processing cost and is a secondary tuning target.
- The calibration image-based/WSClean prediction setup split is accepted.
  Benchmark `runs/benchmark-20260711-081953` confirmed clean return codes,
  stable external command counts, and the expected new visible task groups
  (`make_predict_region_task`, `wsclean_predict_facet_info_task`,
  `wsclean_predict_chunk_task`) in `ci-benchmark-wsclean-predict`.
- The image-sector preparation split is accepted. Benchmark
  `runs/benchmark-20260711-105033` completed with return code `0` for
  `ci-benchmark` and `ci-benchmark-image-products`. On the closest same-runner
  reference (`runs/benchmark-20260711-061530`, `lcs126`), default wall time
  changed by about `+3.1%`, command time changed by about `-6.1%`, command
  count stayed fixed, and the Dask gap stayed effectively unchanged. The split
  exposes per-observation DP3 preparation, concatenation, WSClean imaging,
  WSClean image finishing, optional residual-visibilities production, and the
  prepared-output join as separate tasks.
- Standalone prediction parallelism is accepted. Benchmark
  `runs/benchmark-20260711-162920` completed with return code `0` for
  `ci-benchmark`, `ci-benchmark-image-products`, and
  `ci-benchmark-predict-chunks`. The targeted chunked-predict scenario exposed
  the expected extra predict/postprocess work without material overhead:
  command time changed by about `+0.9%` relative to `ci-benchmark` in the same
  run, while wall time was not worse.
- WSClean-rendered model mosaics are the preferred path when sector sky-model
  inputs exist. Sparse FITS mapping remains a fallback for products that cannot
  be rendered by WSClean.

Operating rules:

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

1. **Freeze and run the baseline performance-equivalence gate.**
   Do this before further optimisation. The current branch has enough accepted
   task-boundary and science-equivalence evidence that the next useful
   checkpoint is a branch-vs-master runtime baseline. That gives later
   `filter_skymodel`, WSClean, and scheduling changes a stable "before"
   reference instead of comparing against a moving target.

   Baseline protocol:

   - use `docs/source/development/performance_equivalence_contract.rst` as the
     contract
   - compare current branch and `master` with matched inputs, resource shape,
     preview settings, run roots, and science checks
   - start advisory: fail only on infrastructure errors, missing outputs,
     failed runs, or science-equivalence failures; report performance as
     pass/warn/fail bands until variance is characterized
   - run `phase-only-core` first; add `dd-phase-plus-di-fulljones` once the
     first scenario is repeatable
   - use at least three repetitions per branch for decision-quality evidence
     when CI/runtime budget allows; otherwise clearly label a one-repetition
     run as a smoke baseline
   - archive compact reports under
     `docs/source/development/performance_equivalence_runs/` and keep raw run
     products out of git

   Do not change performance-sensitive pipeline behavior while this baseline
   gate is being captured.

   Current setup status:

   - `scripts/dev/run_branch_equivalence.py` now records elapsed seconds for
     each branch run and writes runtime min/median/max summaries.
   - The same runner parses `rapthor.log` operation-boundary timings and writes
     per-operation base/current median deltas to the compact JSON/Markdown
     reports.
   - The `phase-only-core` prepare-only smoke has been validated with
     `--repeatability-repetitions 3`; it writes the expected 15 planned pairs.
   - Baseline run instructions live in
     `docs/source/development/performance_equivalence_runs/README.md`.
   - The first full advisory baseline is archived in
     `docs/source/development/performance_equivalence_runs/2026-07-11-phase-only-core-baseline.md`.
     All six branch runs completed successfully and the current branch median
     runtime was about `47.5%` faster than `master`. The strict science
     comparator still exited failed because same-branch repeatability pairs also
     fail the current tolerances, so the next gate task is to calibrate
     pass/warn/fail bands against same-branch scatter before treating this as a
     formal gate pass.

2. **Target the next image-side performance bottlenecks.**
   Current benchmarks consistently show the largest costs are
   `filter_skymodel` and WSClean image runs. Treat calibration plotting as a
   secondary target, and only optimize it if larger/repeated runs keep showing
   it as a meaningful post-processing cost.

   Guidance:

   - Start with `filter_skymodel`: profile resource usage, check whether
     thread/core settings are optimal, and look for algorithmic or I/O wins
     before changing science behavior.
   - Keep WSClean image runs as a resource/concurrency target. They are visible
     as `image_sector_wsclean_task` and are the second-largest image-side
     command cost.
   - Do not optimize the image-sector `prepare_chunk`, concatenate, finish,
     prepare-output, or finalize tasks now; after the split they are visible
     and small in CI-sized benchmarks.
   - Treat `collect_h5parms_task` and calibration chunk timing jumps as
     runner/noise candidates until repeated benchmarks confirm they are real;
     their external command totals have not shown matching growth.
   - Keep WSClean per-facet prediction as a future targeted split. Today
     `wsclean_predict_chunk_*` is chunk-scattered, while Rapthor loops over
     frequency bands and facets inside the task. Investigate facet-level tasks
     only with a targeted benchmark and resource limits, because it can create
     many WSClean calls.

   Benchmark rule:

   - run default automatic coverage: `ci-benchmark`
   - for the current image-side bottleneck batch, keep
     `ci-benchmark-image-products` in CI as the targeted companion scenario
   - otherwise add `ci-benchmark-image-products` only when changing image
     products, image-sector post-processing, `filter_skymodel`, or WSClean
     image resource/concurrency behavior
   - add `ci-benchmark-predict-chunks` only when prediction chunking,
     predict post-processing dependencies, or related scheduling changes
   - add `ci-benchmark-wsclean-predict` only when calibration prediction setup
     or WSClean-predict paths are touched

3. **Keep mosaic science coverage explicit, but targeted.**
   The `multi-sector-mosaic` option-matrix scenario protects sector imaging,
   regridding, and mosaic assembly, but branch-vs-master equivalence is blocked
   because `master` fails before imaging commands run with the generated CWL
   image scatter. Keep current-branch coverage and promote it to a
   stored-reference mosaic science gate once a stable reference is captured.

   Manual current-branch smoke command after regenerating demo data:

   ```bash
   scripts/dev/generate-prefect-demo-data.py --force --include-multi-sector
   rapthor docs/source/development/equivalence_runs/2026-07-06-option-matrix/inputs/current/multi_sector_mosaic.parset
   ```

   Benchmark rule: keep `ci-benchmark-many-sector-mosaic` and
   `ci-benchmark-many-sector-mosaic-sparse-fallback` out of automatic CI unless
   changing mosaic behavior, WSClean-rendered model mosaics, sparse fallback,
   sector regridding, or scalability scheduling. When they are used, run them
   as targeted paired scenarios and archive compact evidence.

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
- Keep task-observability regression tests green: readable run names, stable
  sibling discriminators, tool tags, task-duration report rows,
  command-to-task association, and postage-stamp preview persistence.
- Keep science-regression coverage explicit for calibration strategy behavior,
  image-only cycles, DI pre-apply, DD on-the-fly apply, previous-cycle solution
  handling, and master feature catch-up cases.
- Keep targeted multi-sector mosaic coverage visible. Branch-vs-master
  equivalence is blocked by the master CWL scatter issue, so use
  current-branch integration/benchmark coverage now and add a stored-reference
  mosaic science gate when a stable reference is available.
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

Make large opaque work units Prefect tasks when they are slow, optional,
scientifically meaningful, externally resource-hungry, independently
benchmarkable, or likely to fail in a way users need to identify quickly. The
default posture is now to split these large units for observability, then use
focused tests and batch benchmarks as the guardrail.

Keep these as plain Python:

- validation
- payload construction
- deterministic command builders
- output-record assembly
- path discovery
- tiny helper functions

Do not split tiny helpers or split for Dask alone. A NumPy call inside a
Prefect task runs as normal NumPy in that worker process. Dask helps when
Rapthor submits independent tasks that can run concurrently, when external
tools are given the right thread/core limits, or when code is explicitly
rewritten around Dask-aware collections such as `dask.array` or carefully
chunked delayed work. Avoid adding noisy microtasks or oversubscribing DP3,
WSClean, IDG/IDGCal, PyBDSF, or plotting helpers.

Keep memory efficiency explicit when changing FITS/MS/image-heavy paths.
Rapthor must scale to large datasets and large mosaics, so prefer chunked or
sliced processing, memmap-friendly reads, in-place NumPy operations where they
remain readable, and bounded temporary arrays. Avoid full-image copies unless
they are required for correctness, and use benchmarks or focused profiling to
justify memory-expensive changes.

Task naming:

- Use flow run names for operation context such as operation type, mode, and
  cycle: `calibrate_dd_4`, `predict_di_1`, `image_3`.
- Use short task definition names and task run names because the enclosing
  flow/subflow already provides context. Do not prefix task run names with the
  parent flow name.
- Prefer established legacy workflow vocabulary when still scientifically
  accurate: `filter_skymodel`, `calculate_image_diagnostics`,
  `combine_h5parms`, `collect_h5parms`, `process_solutions`,
  `plot_solutions`, `make_mosaic`.
- Add only the smallest useful discriminator when several sibling tasks of the
  same kind can run in the same flow: `sector_1_filter_skymodel`,
  `prepare_chunk_1`, `solve_chunk_1`, `screen_chunk_1`,
  `dp3_predict_chunk_1`, `wsclean_predict_chunk_1`, `postprocess_1`.
- Use `prepare_chunk_*` for image-sector visibility preparation. The sector is
  the sky work unit; the chunk is the prepared visibility/MS slice inside that
  sector.
- Prefer scientific labels over numerical suffixes when they are stable and
  meaningful, for example `mosaic_I_image` instead of `mosaic_1`.
- Keep tool identity in Prefect tags rather than cramming it into every run
  name. Use lower-case tool tags such as `dp3`, `wsclean`, `python`, `fpack`,
  `pybdsf`, and `casacore`, plus secondary tags only when they help filtering
  without duplicating the flow name.
- Centralize task-run metadata instead of repeating literal strings in every
  flow. Prefer a tiny helper near `rapthor.execution.run_names` that returns
  sanitized run names and tool tags for `.with_options(...)`; keep payloads and
  command builders independent of Prefect metadata.

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
- `docs/source/development/benchmark_baselines/2026-07-10-pre-calibration-postprocess-split.md`
- `docs/source/development/benchmark_baselines/2026-07-11-wsclean-predict-task-split-confirmation.md`
- `docs/source/development/benchmark_baselines/2026-07-11-image-prepare-task-split-confirmation.md`
- `docs/source/development/benchmark_baselines/2026-07-11-predict-chunk-parallelism-confirmation.md`
- `docs/source/development/benchmark_baselines/2026-07-10-worker-thread-wsclean-model-benchmark.md`
- `docs/source/development/benchmark_baselines/2026-07-10-worker-shape-mosaic-method-comparison.md`

Benchmark decision rules:

- Compare absolute wall times only within paired runs on the same runner.
- Prefer medians over individual repetitions, but keep min/max visible.
- Track wall time, command totals, operation-minus-command gap,
  Dask duration-minus-compute gap, task count, task groups, worker/thread
  shape, and return codes.
- For hidden scaling paths, capture a pre-split baseline before changing task
  granularity, then compare each split batch against that baseline on the same
  scenario shape.
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
