# Rapthor Architecture Refactor Plan

Status snapshot: 2026-07-10.

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
- Hidden-path benchmark scenario definitions are in place for image products,
  WSClean-predict calibration, and many-sector mosaic work. The many-sector
  scenario uses a dedicated quadrant-balanced generated dataset and a
  four-sector grid so it exercises sectorized imaging and mosaicking without
  depending on source-boundary edge cases. It uses `dde_method = single` so
  each sector applies the nearest DD solution during imaging; the single-sector
  benchmark keeps the full-DD facet-imaging coverage.
- The pre-split hidden-path benchmark baseline is captured and accepted in
  `docs/source/development/benchmark_baselines/2026-07-09-hidden-path-pre-split-baseline.md`.
  It covers image-products, WSClean-predict, and many-sector mosaic paths with
  three successful repeats each on the CI-sized `2x30` resource profile.
- Sparse model mosaic execution now has focused guards: model-like mosaic
  products use sparse pixel mapping rather than continuous-image
  interpolation, preserving zero-valued background inside valid sector
  footprints and avoiding artificial nonzero-pixel growth.
- WSClean-rendered model mosaics are now wired for model products that have
  sector sky-model/component-list inputs. The sparse FITS model mapper remains
  as a fallback for products without matching sky models until the real
  multi-sector demo, science checks, and benchmark comparison prove the
  WSClean path can replace it.
- The first WSClean model-mosaic multi-sector demo smoke check is captured in
  `docs/source/development/benchmark_baselines/2026-07-09-wsclean-model-mosaic-demo.md`.
  `mosaic_1`, `mosaic_2`, and `mosaic_3` rendered `MFS-model-pb` successfully
  with WSClean, but the full demo failed later in `image_4` with a Prefect/Dask
  threaded settings-cache `KeyError`.
- The Prefect/Dask settings-cache runtime failure has a targeted fix: local and
  Slurm Dask workers now default to one Prefect task-engine thread per worker
  process, while `cpus_per_task` remains the external-command thread budget.
  A rerun confirmed separate worker processes for parallel sector tasks, no
  repeated settings-cache `KeyError`, and successful WSClean model-mosaic
  rendering through all four cycles. The full multi-sector demo completed after
  old run artifacts were cleared.
- The 2026-07-10 CI benchmark is captured in
  `docs/source/development/benchmark_baselines/2026-07-10-worker-thread-wsclean-model-benchmark.md`.
  It accepts the worker-thread runtime fix for the default and image-products
  profiles, keeps the WSClean-predict slowdown as a monitored variance item,
  and flags the many-sector mosaic wall-time increase for targeted follow-up.

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

1. **Keep targeted mosaic science coverage explicit.**
   The `multi-sector-mosaic` option-matrix scenario is defined under
   `docs/source/development/equivalence_runs/2026-07-06-option-matrix/`.
   It uses the generated quadrant-balanced multi-sector demo data, DD
   calibration, a 2x2 sector grid, and `dde_method = single` so it protects
   sector imaging, regridding, and mosaic assembly before more task-boundary
   refactors.

   Branch-vs-master equivalence for this exact scenario is currently skipped:
   `master` fails before imaging commands run because the generated CWL image
   scatter receives a single `parallel_gridding_tasks` value alongside
   per-sector lists. The Prefect/Dask branch runs this scenario successfully.
   Keep the current-branch integration/benchmark coverage active and promote
   this to a stored-reference mosaic science gate once a stable reference run is
   captured.

   Before running it, regenerate the demo data if the multi-sector files are
   missing:

   ```bash
   scripts/dev/generate-prefect-demo-data.py --force --include-multi-sector
   ```

   To re-check the current scenario setup manually, run only this scenario from
   the option matrix:

   ```bash
   scripts/dev/run_branch_option_matrix.py \
     --matrix docs/source/development/equivalence_runs/2026-07-06-option-matrix/option-matrix.json \
     --scenario multi-sector-mosaic \
     --run-root runs/branch-option-matrix-multi-sector-mosaic
   ```

   Preserve compact reports from current-branch and future stored-reference
   runs. The evidence should show that the expected mosaic image types are
   present, image arrays match within tolerance, WCS/header geometry is stable,
   finite/NaN masks are equivalent, and beam/axis metadata remains
   scientifically acceptable. Cover at least `MFS-image`, `MFS-image-pb`,
   `MFS-image-pb-ast`, `MFS-model-pb`, `MFS-residual`, and `MFS-dirty`, because
   these products share the mosaic template/regridding path.

   Sparse model mosaics have focused unit and flow guards plus a successful
   full multi-sector demo using WSClean-rendered model mosaics. Preserve this
   compact evidence and close the queue item only after the option-matrix or
   stored-reference mosaic scenario confirms the same product contract outside
   the ad hoc demo run.

   WSClean-rendered model mosaics are implemented for model products that carry
   matching sector sky-model/component lists. The full multi-sector demo now
   renders `MFS-model-pb` with WSClean in all four mosaic cycles and completes
   without the previous Prefect/Dask runtime failure.

   Next, run the option-matrix/current-branch mosaic scenario, compare the
   WSClean path against the sparse fallback where useful, and preserve compact
   evidence. Remove or demote the custom sparse mapper only after the demo,
   science checks, and benchmark comparison prove the WSClean path.

   Intermediate sector `*-MFS-model-pb.fits.fz` products showed horizontal
   stripe artifacts in CARTA. This was a product-level compression issue, not a
   preview stretch issue: default `fpack` quantizes/dithers sparse
   floating-point model images. Sparse sector model products now use lossless
   `fpack -g -q 0`, while regular image, residual, and dirty products keep the
   existing default compression. Rerun the multi-sector demo or the mosaic
   stored-reference scenario when clean sector model products are needed for
   manual inspection.

   Before treating WSClean-rendered model mosaics as the accepted performance
   baseline, run a targeted many-sector comparison that isolates sparse FITS
   model-mosaic fallback vs WSClean-rendered model mosaics, model products vs
   non-model mosaic products, and the worker-thread runtime shape. The latest
   CI benchmark shows reliable completion but a `+53.5%` many-sector wall-time
   increase despite lower aggregate command time, so the next question is
   scheduling/concurrency rather than basic correctness. The benchmark matrix
   now includes `ci-benchmark-many-sector-mosaic-sparse-fallback`, which uses
   the same generated multi-sector inputs as `ci-benchmark-many-sector-mosaic`
   but overrides `model_mosaic_method = sparse_fits`. Use the next benchmark
   report to decide whether the extra wall time is caused by WSClean-rendered
   model mosaics or by the broader many-sector scheduling/runtime shape.

2. **Systematically split large opaque work units into Prefect tasks.**
   The filter-skymodel and diagnostics benchmarks give enough evidence that
   meaningful task boundaries improve observability without harming this
   CI-sized performance shape. Split large steps by owner package, keeping each
   new task scientifically meaningful, restartable, serializable, and easy to
   identify in the dashboard. Compare each batch against the accepted
   pre-split hidden-path baseline and keep the mosaic science coverage from
   step 1 green when changing mosaic behavior.

   Priority order:

   - image post-processing: `normalize_flux_scale`, image-cube creation,
     `make_catalog_from_image_cube`, restoration/compression steps
   - calibration post-processing: `collect_h5parms`, `process_slow_gains`,
     full-Jones normalization, `combine_h5parms`, `plot_solutions`
   - prediction: WSClean-predict loops and sector-model post-processing
   - mosaic: WSClean-rendered model mosaics, per-sector regridding, mosaic
     assembly, and compression

3. **Benchmark after each owner-package split batch.**
   Rerun the relevant hidden-path scenarios after each batch. Keep the split
   when task count, scheduler gap, wall time, command totals, restart behavior,
   and raw/scientific outputs remain acceptable. Add compact reports under
   `docs/source/development/benchmark_baselines/`.

   Include the Dask worker-thread runtime fix in the next benchmark comparison:
   verify that moving from multi-threaded worker processes to single-threaded
   worker processes does not harm the accepted CI-sized resource shapes, and
   prefer adjusting worker count over restoring multiple Prefect task-engine
   threads per worker.

4. **Build the scalability/performance equivalence gate.**
   Compare current branch and master with identical inputs, resource shape,
   preview settings, run roots, and science checks. Start advisory: fail only
   on infrastructure errors, missing outputs, failed runs, or science
   equivalence failures; report performance as pass/warn/fail bands until
   variance is characterized.

5. **Guard the science-equivalence contract.**
   For documentation, preview-artifact, benchmark-report, or refactor-only
   changes, run focused tests. For calibration, prediction, imaging, h5parm,
   FITS, catalog, sky-model, or product-record changes, rerun the relevant
   saved-reference and branch-vs-master scenarios before judging the change.

6. **Polish runtime UX and contributor docs after the next scalability result.**
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
  `combine_h5parms`, `collect_h5parms`, `process_slow_gains`,
  `plot_solutions`, `predict_model_data`, `make_mosaic`.
- Add only the smallest useful discriminator when several sibling tasks of the
  same kind can run in the same flow: `sector_1_filter_skymodel`,
  `chunk_1`, `screen_1`, `model_1`, `postprocess_1`.
- Prefer scientific labels over numerical suffixes when they are stable and
  meaningful, for example `mosaic_I_image` instead of `mosaic_1`.

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
