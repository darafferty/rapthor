# Rapthor Architecture Refactor Plan

Status snapshot: 2026-07-08.

## Goal

Make the Prefect/Dask Rapthor pipeline easy to run, understand, test, debug,
benchmark, and scale while preserving the user-facing CLI workflow:

```bash
rapthor input.parset
```

This Prefect/Dask implementation has not been released, so prefer clean
production architecture over unreleased compatibility shims.

## Current Position

The architecture and science-equivalence work has moved from active migration
to guarded scalability work. The latest post-split filter-skymodel benchmark in
`runs/benchmark-20260708-121853/` has been analysed. It supports keeping the
global `2x30`, `max_threads=30` resource shape and promoting
`filter_skymodel_ncores=15` from benchmark profile to proposed production
default for the CI-sized configuration. Absolute wall times should be compared
within the paired run only because the July 8 runs used different CI runners.

Completed:

- Execution code is organized by owner package:
  `image`, `calibrate`, `concatenate`, `predict`, `mosaic`, and `pipeline`.
- Operation adapters are thin. Payload builders, validation, command builders,
  output discovery, migrated helper-script logic, and flow wiring live in
  execution owner packages.
- Prefect/Dask runtime bootstrap, local/external Dask setup, CLI smoke
  coverage, runtime parset materialization, readable run names, and dashboard
  / report logging are in place.
- Dask task-boundary guardrails cover owner-flow submissions, plain
  serializable worker payloads, readable task names, representative payload
  fixtures, and resource propagation into local Dask startup and operation
  payloads.
- Master feature catch-up is complete for the known runtime/product features:
  MS time ordering, reset-directory guards, astrometry-corrected products,
  per-facet RMS diagnostics, residual visibilities, WSClean-based prediction,
  normalization semantics, parallel gridding, and shared-facet behavior.
- The flexible calibration strategy contract is implemented and documented:
  `calibration_strategy` controls solve type/order; previous-cycle products may
  seed matching solves only as optimizer seeds when product role and DD
  direction compatibility are valid; preapply and imaging-time h5parm use
  remains current-cycle guarded except for explicit image-only carry-forward.
- Image-only application semantics are covered: DI scalar phase, DI diagonal
  slow-gain, and DI full-Jones products are pre-applied; DD products are
  applied on the fly during imaging when matching directions are available.
- DI full-Jones collection now matches legacy master post-processing:
  collected full-Jones gains are amplitude-normalized before plotting/finalizer
  handoff, without slow-gain flagging or smoothing.
- The strengthened saved-reference equivalence matrix, the active
  branch-vs-master scenarios, and the risk-based option rows are accepted for
  the covered contract and summarized in `EQUIVALENCE_REPORT.md`.
- The first low-risk image-cycle scalability slice has been implemented:
  image-sector work is split into post-WSClean preparation/finalization task
  boundaries with the same Dask-shaped graph for supported task-runner modes.
- Benchmark scaffolding exists for the `ci-benchmark` scenario, including
  timestamped run roots, GitLab metadata, Dask report parsing, command timing,
  operation-boundary timing, JSON summaries, and Markdown reports.
- Benchmark preview overhead is explained and guarded: generated CI benchmark
  parsets keep FITS and postage-stamp preview artifacts disabled, while demo
  parsets keep them enabled for visual dashboard inspection.
- A non-default `filter_skymodel_ncores` resource control is wired through the
  cluster parset, image operation payload, image execution payload, demo
  runner, and benchmark profile runner. Its default preserves previous
  behavior by resolving to `max_threads`.
- The targeted `filter-only-15` benchmark profile has been captured. It keeps
  WSClean, DP3, Dask workers, and `max_threads=30` at the `2x30` baseline while
  running `filter_skymodel` with `15` cores. It improves median wall time by
  `10.836 s` (`3.48%`) in `runs/benchmark-20260708-073531/`, but should be
  confirmed after the filter task boundary lands before changing defaults.
- `filter_skymodel` is now its own image-sector Prefect task named
  `filter_skymodel`. The image-sector graph is
  `image_sector_prepare -> filter_skymodel -> image_sector_finalize`, with
  sector-specific detail carried by `task_run_name`.
- The post-split benchmark confirms the filter-only optimization: median wall
  time improves from `213.347 s` to `183.968 s` (`13.77%`) and median
  `filter_skymodel` command time improves from `89.540 s` to `69.881 s` while
  the Dask task count remains `20`.
- `filter_skymodel_ncores=15` has been promoted to the proposed production
  default in the packaged parset defaults and regenerated benchmark parsets.
  Explicit user overrides are preserved: setting the value to `0` still uses
  `max_threads`.

Keep these caveats visible:

- Use the prepared dev-container Python environment for tests, formatting,
  integration checks, equivalence checks, demo runs, and benchmark runs. Local
  tox environments may fail to build astronomy dependencies.
- Keep large integration, equivalence, benchmark, and demo run roots outside
  git, preferably under `/tmp`, `runs/`, or CI artifacts.
- Do not commit raw run directories, FITS/MS products, full Dask HTML reports,
  Prefect logs, `.tox`, `.ruff_cache`, `htmlcov`, or build products. Compact
  curated reports, manifests, and short command logs may be tracked under
  `docs/source/development/` when they explain an important result.
- Preview artifacts remain diagnostic aids only. Raw FITS/h5parm products,
  numeric diagnostics, catalogs, region files, command records, and report JSON
  remain the scientific contract.

## Immediate Next Work, In Order

Use this section as the active queue.

1. **Run one confirmation benchmark for the promoted
   `filter_skymodel_ncores=15` default.**
   Keep the global `2x30`, `max_threads=30` worker/thread shape and preserve
   explicit user overrides. The confirmation run should show that the default
   profile matches the previous `ci-benchmark-filter-only-15` behavior and does
   not change the raw/scientific product contract.

2. **Keep the first scalability split for now, but treat it as an observability
   improvement rather than a proven speedup.**
   The July 7 split increases Dask task count from `12` to `16` and exposes
   `image_sector_prepare_task` and `image_sector_finalize_task`, while wall
   time remains close to the July 6 run. Revisit this if dashboard noise,
   focused tests, or product checks show a downside.

3. **Use an explicit task-boundary policy before splitting more flow steps.**
   A step should become a Prefect task when it is slow, optional,
   scientifically meaningful, externally resource-hungry, independently
   benchmarkable, or likely to fail in a way users need to identify quickly.
   Keep small validation, payload construction, output-record assembly, and
   path-discovery helpers as plain Python functions. Rank the next splits as
   follows:

   - **Split next when profiling confirms the default benchmark is stable:**
     `calculate_image_diagnostics` as a separate image-sector task after
     `filter_skymodel`. It loads FITS/catalog/skymodel products, writes JSON
     and plot artifacts, and is scientifically meaningful enough to deserve
     dashboard visibility even if the CI benchmark currently shows only a
     small finalization cost.
   - **Split when the option is enabled in a benchmark or manual profile:**
     `normalize_flux_scale` / `make_catalog_from_image_cube`. This path is
     optional, uses FITS/catalog/MS/h5parm data, and should be independently
     observable when flux-scale normalization is exercised.
   - **Measure before splitting:** calibration post-processing
     (`collect_h5parms`, `process_slow_gains`, full-Jones normalization,
     `combine_h5parms`, `plot_solutions`). The CI-sized benchmark shows
     h5parm collection itself is small, but plotting and Python helper time can
     matter in richer real runs and failures here are useful to isolate.
   - **Split only for WSClean-predict scenarios:** calibration
     `prepare_image_based_predict` / `wsclean_predict` loops. These can contain
     independent copied-MS, frequency-chunk, and patch prediction work, but the
     path is optional and should be benchmarked with a scenario that enables it.
   - **Defer unless profiling says otherwise:** mosaic regridding per sector
     image. Current CI mosaic time is small, but many sectors/image types could
     make per-regrid tasks useful later.
   - **Do not split for Dask alone:** validation, payload construction,
     deterministic command builders, output-record assembly, path discovery,
     and tiny helper functions. They should stay plain Python unless they hide
     a measured failure mode.

   NumPy operations are not automatically distributed or accelerated by Dask.
   A NumPy call inside a Prefect task runs in that worker process exactly as
   NumPy normally would. Dask helps when Rapthor submits independent tasks that
   can run concurrently, when external commands are given the right thread/core
   limits, or when code is explicitly rewritten to use Dask-aware collections
   such as `dask.array` or carefully chunked delayed work. Avoid adding task
   boundaries that only add scheduler overhead or oversubscribe external tools.

   Task names should be short step names because the enclosing flow/subflow
   already provides context. Prefer the established legacy CWL/workflow
   vocabulary when it is still scientifically accurate: use names such as
   `filter_skymodel`, `calculate_image_diagnostics`, `combine_h5parms`,
   `collect_h5parms`, `process_slow_gains`, `plot_solutions`,
   `predict_model_data`, and `make_mosaic`. Avoid redundant names such as
   `image_sector_filter_skymodel` inside the `image` flow unless two tasks in
   the same flow would otherwise collide. Use `task_run_name` for per-operation
   and per-sector specificity, for example `image_dd_3_sector_1_filter_skymodel`.

4. **Add a scalability/performance equivalence gate next to the science gate.**
   The branch-vs-master decision should have explicit performance evidence, not
   just successful science-product comparison. Build this as an advisory gate
   first, then promote it to a required release/merge gate once the scenarios
   and variance are stable. See "Scalability and Performance Equivalence Gate"
   below for the task list.

5. **Guard the accepted science-equivalence contract.**
   For documentation, preview-artifact, benchmark-report, or refactor-only
   changes, run focused tests. For calibration, prediction, imaging, h5parm,
   FITS, catalog, sky-model, or product-record changes, rerun the relevant
   saved-reference and branch-vs-master scenarios before judging the change.

6. **Resume maintainability and runtime-UX cleanup after the next scalability
   decision.**
   Keep `TESTING.md`, `.agents/testing_playbook.md`, `AGENTS.md`, runtime docs,
   and this plan aligned as the test and runtime surfaces settle.

## Benchmark Evidence

Compact analysis:

- `docs/source/development/benchmark_baselines/2026-07-07-ci-benchmark-comparison.md`
- `docs/source/development/benchmark_baselines/2026-07-07-ci-benchmark-comparison.summary.json`
- `docs/source/development/benchmark_baselines/2026-07-08-filter-skymodel-resource-profiles.md`
- `docs/source/development/benchmark_baselines/2026-07-08-filter-skymodel-resource-profiles.summary.json`
- `docs/source/development/benchmark_baselines/2026-07-08-filter-skymodel-only-profile.md`
- `docs/source/development/benchmark_baselines/2026-07-08-filter-skymodel-only-profile.summary.json`
- `docs/source/development/benchmark_baselines/2026-07-08-post-split-filter-skymodel-profile.md`
- `docs/source/development/benchmark_baselines/2026-07-08-post-split-filter-skymodel-profile.summary.json`

Latest raw artifacts:

- `runs/benchmark-20260704-122100/benchmark-report.md`
- `runs/benchmark-20260704-122100/benchmark-summary.json`
- `runs/benchmark-20260704-122100/benchmark-results.json`
- `runs/benchmark-20260704-122100/ci-benchmark/rep-*/benchmark-result.json`
- `runs/benchmark-20260704-122100/ci-benchmark/rep-*/dask-performance-report.html`
- `runs/benchmark-20260706-203026/benchmark-report.md`
- `runs/benchmark-20260706-203026/benchmark-summary.json`
- `runs/benchmark-20260706-203026/benchmark-results.json`
- `runs/benchmark-20260706-203026/ci-benchmark/rep-*/benchmark-result.json`
- `runs/benchmark-20260706-203026/ci-benchmark/rep-*/dask-performance-report.html`
- `runs/benchmark-20260707-153316/benchmark-report.md`
- `runs/benchmark-20260707-153316/benchmark-summary.json`
- `runs/benchmark-20260707-153316/benchmark-results.json`
- `runs/benchmark-20260707-153316/ci-benchmark/rep-*/benchmark-result.json`
- `runs/benchmark-20260707-153316/ci-benchmark/rep-*/dask-performance-report.html`
- `runs/benchmark-20260707-191141/benchmark-report.md`
- `runs/benchmark-20260707-191141/benchmark-summary.json`
- `runs/benchmark-20260707-191141/benchmark-results.json`
- `runs/benchmark-20260707-191141/ci-benchmark-*/rep-*/benchmark-result.json`
- `runs/benchmark-20260707-191141/ci-benchmark-*/rep-*/dask-performance-report.html`
- `runs/benchmark-20260708-073531/benchmark-report.md`
- `runs/benchmark-20260708-073531/benchmark-summary.json`
- `runs/benchmark-20260708-073531/benchmark-results.json`
- `runs/benchmark-20260708-073531/ci-benchmark-*/rep-*/benchmark-result.json`
- `runs/benchmark-20260708-073531/ci-benchmark-*/rep-*/dask-performance-report.html`
- `runs/benchmark-20260708-121853/benchmark-report.md`
- `runs/benchmark-20260708-121853/benchmark-summary.json`
- `runs/benchmark-20260708-121853/benchmark-results.json`
- `runs/benchmark-20260708-121853/ci-benchmark-*/rep-*/benchmark-result.json`
- `runs/benchmark-20260708-121853/ci-benchmark-*/rep-*/dask-performance-report.html`

Headline comparison:

| Run | Commit | Wall Median (s) | Dask Gap Median (s) | Task Count | Image Task Shape | Notes |
| --- | --- | ---: | ---: | ---: | --- | --- |
| `20260704-122100` | unavailable in report | `482.894` | `220.940` | `12` | `image_sector_task` x4 | Original compact baseline in this plan. |
| `20260706-203026` | `df67648f` | `305.320` | `70.480` | `12` | `image_sector_task` x4 | Large wall-time and scheduler-gap improvement with similar command time. |
| `20260707-153316` | `eb0a4033` | `308.563` | `72.570` | `16` | `image_sector_prepare_task` x4 plus `image_sector_finalize_task` x4 | First visible split in Dask task groups; wall time remains close to 2026-07-06. |
| `20260707-191141` | `32e64149` | `306.507` | `72.450` | `16` | Same split as July 7 baseline | Resource-profile baseline; still the best tested worker/thread shape. |
| `20260708-073531` | `2a86f8ac` | `311.315` baseline / `300.479` filter-only | `73.030` baseline / `71.410` filter-only | `16` | Same split as July 7 baseline | `filter_skymodel_ncores=15` is promising; rerun after the filter task boundary before changing defaults. |
| `20260708-121853` | `9f5ea488` | `213.347` baseline / `183.968` filter-only | `46.550` baseline / `45.990` filter-only | `20` | Adds `image_sector_filter_skymodel_task` x4 | Post-split result confirms `filter_skymodel_ncores=15`; promote to proposed default and rerun once. |

Resource-profile result from `20260707-191141`:

| Profile | Wall Median (s) | Delta vs Baseline | Command Median (s) | `filter_skymodel` (s) | WSClean (s) | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `baseline-2x30` | `306.507` | `0.000` | `228.392` | `109.566` | `87.066` | Keep as baseline. |
| `filter-threads-15` | `305.529` | `-0.32%` | `227.387` | `105.187` | `93.289` | Wall-time neutral; do not globally lower `max_threads`. |
| `filter-wide-1x60` | `347.278` | `+13.30%` | `271.584` | `119.811` | `113.504` | Reject. |
| `filter-workers-4x15` | `322.129` | `+5.10%` | `240.928` | `107.816` | `104.249` | Reject. |

Filter-only result from `20260708-073531`:

| Profile | Wall Median (s) | Delta vs Baseline | Command Median (s) | `filter_skymodel` (s) | WSClean (s) | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `baseline-2x30` | `311.315` | `0.000` | `230.657` | `109.771` | `89.082` | Keep the global `2x30`, `max_threads=30` baseline. |
| `filter-only-15` | `300.479` | `-3.48%` | `224.181` | `105.399` | `87.170` | Promising; rerun after the `filter_skymodel` task split before changing defaults. |

Post-split filter-only result from `20260708-121853`:

| Profile | Wall Median (s) | Delta vs Baseline | Command Median (s) | `filter_skymodel` (s) | WSClean (s) | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `baseline-2x30` | `213.347` | `0.000` | `157.881` | `89.540` | `53.592` | Keep the global `2x30`, `max_threads=30` baseline. |
| `filter-only-15` | `183.968` | `-13.77%` | `135.576` | `69.881` | `50.381` | Promote `filter_skymodel_ncores=15` to proposed default and rerun once. |

Common benchmark shape:

- Scenario: `ci-benchmark`
- Repetitions: `3`
- Return codes: `0, 0, 0`
- Historical baseline worker/thread shape: `2` workers, `60` total Dask
  threads, with `max_threads=30`
- Absolute timings vary with CI runner. Use paired comparisons within the same
  timestamped benchmark run for decisions.

Latest command timing by name:

| Run | Dominant Command | Median Total (s) | Median Count | Notes |
| --- | --- | ---: | ---: | --- |
| `20260706-203026` | `filter_skymodel` | `110.228` | `4` | Dominates image command time before the image-sector split. |
| `20260706-203026` | `wsclean` | `87.567` | `4` | Second-largest image command group. |
| `20260707-153316` | `filter_skymodel` | `110.312` | `4` | Still dominant after the image-sector split. |
| `20260707-153316` | `wsclean` | `88.961` | `4` | Stable relative to July 6. |
| `20260708-073531` baseline | `filter_skymodel` | `109.771` | `4` | Baseline for the targeted filter-only profile. |
| `20260708-073531` baseline | `wsclean` | `89.082` | `4` | Stable relative to the July 6/7 baselines. |
| `20260708-073531` filter-only | `filter_skymodel` | `105.399` | `4` | Positive signal from lowering only `filter_skymodel_ncores` to `15`. |
| `20260708-073531` filter-only | `wsclean` | `87.170` | `4` | No WSClean slowdown in the targeted profile. |
| `20260708-121853` baseline | `filter_skymodel` | `89.540` | `4` | Post-split baseline on a different runner; compare only with the paired filter-only row. |
| `20260708-121853` baseline | `wsclean` | `53.592` | `4` | No evidence that the split itself harms WSClean. |
| `20260708-121853` filter-only | `filter_skymodel` | `69.881` | `4` | Dedicated filter task group confirms the `15`-core improvement. |
| `20260708-121853` filter-only | `wsclean` | `50.381` | `4` | No WSClean regression in the paired profile. |

Resolved finding:

- The large July 4 to July 6 wall-time and Dask-gap improvement is most likely
  explained by FITS preview artifact rendering and Prefect artifact publication
  moving from always-on image/mosaic behavior to opt-in settings that default
  false in benchmark parsets. Command-profile time remains stable, so the
  improvement is mostly Python-side orchestration/artifact overhead rather than
  faster external commands.
- The July 8 targeted `filter-only-15` run shows a real positive signal while
  keeping the global worker/thread shape unchanged: median wall time improves by
  `10.836 s` (`3.48%`), median command time improves by `6.476 s`, and WSClean
  does not regress. Confirm this after the `filter_skymodel` task split before
  promoting it from benchmark profile to default.
- The post-split July 8 run confirms the filter-only setting with better
  observability: `filter_skymodel` is now visible as four Dask tasks, median
  wall time improves by `29.379 s` (`13.77%`), and median filter command time
  improves by `19.659 s`. Promote `filter_skymodel_ncores=15` as the proposed
  default, but run one confirmation benchmark after the default change.

Remaining questions:

- After the default changes, does the confirmation benchmark match the
  `filter-only-15` behavior?
- Can the benchmark runner interleave or randomize scenario order in a future
  pass to reduce cache/warm-up bias?
- Is wall time stable enough across repetitions for a hard performance gate, or
  should the gate remain advisory while variance is characterized?
- Is the `duration-minus-compute` gap dominated by Prefect/Dask orchestration,
  external-command blocking, task dependencies, or idle workers?
- Are image operation-minus-command gaps reduced after the split, or do they
  point to another boundary such as diagnostics, filtering, compression, cube
  generation, catalog extraction, or h5parm/product collection?
- Does the `2x30` baseline leave meaningful work idle after the post-split
  filter-only benchmark, or are the external tools already saturating the
  available resources?

Benchmark before and after any task-boundary, scheduler, dependency, or
performance-sensitive execution change.

Local smoke benchmark command for smaller workstations:

```bash
scripts/dev/run_benchmark_baseline.py \
  --scenario ci-benchmark \
  --repetitions 1 \
  --local-dask-workers 1 \
  --cpus-per-task 4 \
  --max-threads 4
```

## Scalability and Performance Equivalence Gate

Goal: provide the same kind of decision-quality evidence for runtime behavior
that the science-equivalence gate provides for scientific products. The gate
should answer: on the same inputs and hardware, does the current Prefect/Dask
branch preserve science outputs while matching or improving master wall time,
resource use, and scaling behavior?

Tasks:

1. **Define the comparison contract.**
   Use identical input data, strategy intent, output-product expectations,
   preview-artifact settings, thread limits, and run roots for master and the
   current branch. Keep generated preview artifacts disabled unless the scenario
   explicitly measures dashboard/reporting overhead. Record the exact git refs,
   container image, CPU count, worker/thread shape, environment variables, and
   parset materialization in every report. The first draft is documented in
   `docs/source/development/performance_equivalence_contract.rst`.

2. **Choose a small scenario matrix.**
   Start with one fast CI-sized scenario that both branches can represent, plus
   one rich scenario that exercises DI/DD calibration, imaging, mosaicking,
   sky-model filtering, and h5parm handling. Add larger-node or multi-node
   scenarios only after the local/CI runner gate is repeatable. Avoid scenarios
   where master has a known scientific bug unless the report labels it as a
   legacy limitation rather than a current-branch regression. The initial
   matrix is documented in
   `docs/source/development/performance_equivalence_contract.rst`:
   `phase-only-core`, `dd-phase-plus-di-fulljones`, and the supporting
   current-branch-only `ci-benchmark-current-profile`.

3. **Extend the branch-equivalence workflow with performance metadata.**
   Either add a performance mode to `scripts/dev/run_branch_equivalence.py` or
   create a sibling `scripts/dev/run_branch_performance_equivalence.py` that
   runs master and current with the same prepared inputs for `N` repetitions.
   Capture wall time, return code, command timing where available, operation
   timing where available, Dask task counts for the current branch, output
   product summaries, and the science-equivalence result for the same run.

4. **Make the first gate advisory, not a hard CI failure.**
   Store a compact Markdown report and JSON summary as CI artifacts and, for
   accepted milestone runs, under
   `docs/source/development/performance_equivalence_runs/`. Fail only on
   infrastructure errors, missing outputs, failed runs, or science-equivalence
   failures. Report performance deltas as pass/warn/fail bands until enough
   repeatability data exists to set strict thresholds.

5. **Define decision metrics before reading new results.**
   Compare median wall time, min/max spread, command-profile totals, dominant
   command groups, operation-minus-command gap, Dask duration-minus-compute gap,
   task count, and resource shape. Treat "current branch is faster by a stable
   margin", "current branch is neutral but cleaner/scales better", and "current
   branch is slower but scientifically safer" as separate decision outcomes.

6. **Add architecture guard tests for performance-sensitive contracts.**
   Tests should protect shape, not elapsed seconds: benchmark profiles remain
   configured, `ci-benchmark` keeps preview artifacts disabled and command
   profiling enabled, resource profiles override generic runtime args, expected
   Dask task groups stay visible, worker payloads remain serializable, and
   command logs keep the fields needed by reports.

7. **Document the reviewer workflow.**
   Add copy/paste commands for running the gate locally and on CI, where to
   find artifacts, which compact files are safe to commit, and how to interpret
   master-vs-current deltas. Link the gate from `TESTING.md`,
   `.agents/testing_playbook.md`, and the development architecture docs once it
   is stable.

Do not use absolute wall-clock thresholds as unit or architecture tests. Use
tests to protect the benchmarkable runtime shape, and use benchmark/equivalence
artifacts to make the performance decision.

## Scientific Equivalence Track

Current status:

- The saved-reference matrix passed with strengthened FITS checks on
  2026-07-04. The current report path and residual summary are recorded in
  `EQUIVALENCE_REPORT.md`.
- `scripts/dev/run_branch_equivalence.py` now compares arbitrary explicitly
  prepared base/current parsets across `master` and the current branch. It does
  not translate parsets or strategies; the input files are the scientific
  contract. Use `--prepare-only` to validate/report the chosen files before
  launching expensive runs. Prefer absolute paths inside these parsets for
  shared data and strategy files, because each side is run from its own branch
  checkout. Use `--setup-base-env` to create or reuse a virtualenv for the base
  ref so `master` or a chosen commit runs with its own Python dependencies
  while the current branch runs normally.
- The first manual `benchmark-default-like` branch-vs-master run used
  `master` commit `17448437b78583f1eaf38112a524b2dbe5f34bb8` as the reference.
  Both branches returned `0`, but the strengthened product comparison failed.
  The compact report, manifest, and command logs are tracked under
  `docs/source/development/equivalence_runs/2026-07-04-default-like-master-ref/`.
  The key finding is that master logs a `combine_h5parms.py ...
  p1p2a2_diagonal` broadcasting error during the slow-gain path but still
  completes, leaving the active facet h5parm phase-only.
- The manual `benchmark-phase-only` branch-vs-master run also used `master` as
  the reference. Both branches returned `0`; strict comparison failed, but the
  restored dirty/image/residual FITS deltas are small, the h5parm phase delta is
  about `1.54e-05`, and the remaining failures are mostly missing
  `field-MFS-image-pb-ast` products, sparse model-image differences,
  source-catalog/facet-region text differences, and output-record summaries.
  The compact report, manifest, and command logs are tracked under
  `docs/source/development/equivalence_runs/2026-07-04-phase-only-master-ref/`.
- The 2026-07-05 phase-only rerun used the same `master` reference after the
  current branch had caught up with the missing image-product behavior. Both
  branches returned `0`; strict comparison still failed. The rerun includes 32
  image diagnostic deltas and 20 side-by-side visual comparisons under
  `docs/source/development/equivalence_runs/2026-07-05-phase-only-master-ref/`.
  `field-MFS-image-pb-ast` products are now present on both branches. Top-level
  diagnostics are close, with matching source counts and largest relative
  deltas around `0.23%`, but h5 phase/source differences grow by cycle:
  exact in cycle 1, about `2.18e-07` in cycle 2, about `1.60e-03` in cycle 3,
  and about `6.12` plus source-coordinate differences in cycle 4.
- Investigation on 2026-07-05 found the first actionable semantic difference:
  master passes previous-cycle DD fast-phase h5parms into later calibration
  cycles as `fast_initialsolutions_h5parm`, while this branch's migrated
  calibration adapter treats those files as stale because their recorded cycle
  number is lower than the current calibration index. The current logs show
  explicit "Ignoring DD fast-phase/medium-phase h5parm ... produced in cycle
  N" warnings in cycles 2-4. Saved pipeline inputs confirm master provides
  previous-cycle fast-phase initial solutions for cycles 2-4, while current
  inputs set `solve1_initialsolutions_h5parm = None`. The phase h5parms are
  exact in cycle 1, nearly exact in cycle 2, then diverge cumulatively, which
  is consistent with different solve initialization rather than a parallel
  gridding/resource issue.

Initial-solution alignment status:

- Updated on 2026-07-05: calibration solve initialization now has a separate
  resolver from preapply products. Current-cycle same-solve products may seed
  matching solves. Previous-cycle same-mode/same-solve products may seed
  matching solves only as optimizer seeds when cycle and product role are
  valid; DD products additionally require compatible directions via fixed
  `facet_layout`, matching h5parm directions, or a future explicit remap/filter
  step. Future-cycle products are rejected. The strict current-cycle guard
  remains for calibration preapply products and imaging-time h5parm application.
- Focused operation and execution tests cover current-cycle, previous-cycle,
  future-cycle, phase-only DD, slow-gain DD, DI, wrong-mode, DD direction
  matching, fixed-`facet_layout` carry-over, and payload/command propagation
  cases.
- The rerun tracked under
  `docs/source/development/equivalence_runs/2026-07-05-phase-only-initial-solutions-master-ref/`
  confirms both branches pass previous-cycle DD fast-phase initial solutions in
  cycles 2-4 and leave phase-only medium/slow initial-solution slots unset.
  Re-run this scenario after the flexible carry-over contract is fully
  validated, because the current branch now intentionally allows matching
  phase-only medium seeds when directions are compatible.
- The first essential branch-vs-master mixed-mode scenario is tracked under
  `docs/source/development/equivalence_runs/2026-07-05-dd-phase-plus-di-fulljones-master-ref/`.
  Both branches return `0` for a single DD fast+medium phase-only cycle followed
  by DI full-Jones. Strict comparison still fails, but operation order, product
  presence, source counts, and high-level diagnostics are close; the largest
  image-diagnostic relative deltas are about `0.24%`, and the full-Jones
  amplitude h5parm delta is about `1.14e-03`.
- Follow-up investigation found that the DP3 solve commands, WSClean imaging
  commands, image preapply payloads, and full-Jones applycal settings match
  between branches. The systematic amplitude offset comes from post-processing:
  legacy master runs `process_gains.py` on the collected full-Jones h5parm with
  normalization enabled and flagging/smoothing disabled, while the current
  branch was previously handing the raw collected full-Jones gains to imaging.
  The current branch now ports that normalization behavior and focused tests
  cover the exact processing options. Rerun the focused full-Jones equivalence
  check before interpreting the old DD-plus-DI full-Jones repeatability envelope.
- The normalized focused rerun is tracked under
  `docs/source/development/equivalence_runs/2026-07-06-dd-phase-plus-di-fulljones-normalized-master-ref/`.
  Both branches return `0`; strict h5parm comparison now passes for all three
  h5parm products. The restored-image max absolute residual dropped from about
  `1.025e-02` to about `2.486e-05`, and the largest diagnostic relative delta
  is about `0.021%`. Remaining failures are small FITS image residuals, PyBDSF
  source-catalog uncertainty/shape columns, DS9 region text formatting, and
  legacy output-record metadata shape.
- The DI multi-cycle carry-over scenario is tracked under
  `docs/source/development/equivalence_runs/2026-07-05-di-multicycle-carryover-master-ref/`.
  It uses two selfcal/image cycles of master-compatible DD fast+medium
  phase-only calibration followed by DI full-Jones. This exposed a
  current-branch bug where a previous full-Jones h5parm was passed as an
  optimizer seed with `solve1.initialsolutions.soltab=[phase000]`; DP3 rejected
  the cycle-2 DI solve. The current branch now uses
  `[amplitude000,phase000]` for full-Jones initial solutions, with focused
  command and operation tests. The rerun returns `0` on both branches, reaches
  `calibrate_2`, `calibrate_di_2`, and `image_2`, and confirms image-cycle
  application of the cycle-2 full-Jones h5parm. Strict comparison still fails:
  cycle-1 diagnostics are close, while cycle-2 image RMS diagnostics differ by
  about `9-10%`, consistent with compounded differences after master carries
  unsafe DD seeds across regrouped directions and current skips them.
- The mode-boundary scenarios are tracked under
  `docs/source/development/equivalence_runs/2026-07-05-di-then-dd-mode-boundary-master-ref/`
  and
  `docs/source/development/equivalence_runs/2026-07-05-dd-then-di-mode-boundary-master-ref/`.
  Both use fixed facets and both branches return `0`. The DI-to-DD scenario
  confirms a policy difference: master carries the cycle-1 DI full-Jones
  product into cycle-2 imaging even though cycle 2 has no DI solve, while the
  current branch keeps imaging-time full-Jones application current-cycle
  guarded. The paired DD-to-DI scenario confirms that both branches apply the
  cycle-2 full-Jones product when cycle 2 runs DI; the remaining differences
  are dominated by DD seed policy and full-Jones numeric drift.
- The fixed-`facet_layout` carry-over scenario is tracked under
  `docs/source/development/equivalence_runs/2026-07-05-fixed-facet-carryover-master-ref/`.
  Both branches return `0` and run `calibrate_2`. Master passes only the
  previous fast-phase seed into cycle 2, while the current branch passes
  compatible fast and medium phase seeds because the fixed facet layout proves
  DD direction compatibility. Strict comparison still fails, as expected for
  this intentional flexible-strategy difference.
- The changing-facet carry-over scenario is tracked under
  `docs/source/development/equivalence_runs/2026-07-05-changing-facet-carryover-master-ref/`.
  Both branches return `0` and run `calibrate_2` after cycle 2 changes from
  five DD directions to three. Master still passes the previous fast-phase
  seed, while the current branch skips all previous-cycle DD initial-solution
  h5parms and logs the direction mismatch. Strict comparison still fails, but
  the safety behavior is the intended current-branch flexible-strategy
  contract.
- The slow-gain/default-like scenario is tracked under
  `docs/source/development/equivalence_runs/2026-07-05-slow-gain-default-like-master-ref/`.
  Both branches return `0` for a one-cycle calibration-only DD solve order
  equivalent to fast phase, medium phase, slow gains, and post-slow medium
  phase. Both produce standalone slow-gain and medium2 h5parms. Master still
  logs the `combine_h5parms` broadcast error and its final `field-solutions.h5`
  lacks `sol000/amplitude000`, while the current branch preserves the slow-gain
  amplitude soltab in the final combined h5parm.

Possible bugs on the master branch to investigate:

- Slow-gain h5parm combination can fail while the overall run still returns
  success. In the 2026-07-04 default-like branch-vs-master reference run,
  master logged a `combine_h5parms.py ... p1p2a2_diagonal` broadcasting error
  during the slow-gain path, but the CWL step/run completed and the active
  facet h5parm remained phase-only. The 2026-07-05 one-cycle slow-gain
  default-like run reproduced the issue in a tighter calibration-only case:
  master produced standalone slow-gain products but its final
  `field-solutions.h5` lacked `sol000/amplitude000`, while the current branch
  preserved it. Verify whether master should fail loudly, whether
  `p1p2a2_diagonal` should handle the product shapes, or whether this should
  remain a documented master reference limitation. The historical ICAL
  benchmark in `runs/2026-07-02-ical-1node/` used legacy Rapthor
  `2.2.dev117+g01a81e1`, ran the same `p1p2a2_diagonal` combine step for
  cycles 3-7, and completed successfully because its slow-gain h5parms had a
  full frequency axis. Treat the issue as shape/configuration-triggered rather
  than universal to every master slow-gain run.
- Previous-cycle DD initial-solution h5parms are carried across calibration
  patch/facet changes without direction compatibility checks. In the 2026-07-05
  phase-only master reference run, cycle 1 used `[Patch_rich_*]`, cycle 2 used
  `[Patch_0..4]`, and cycle 4 used ten sector-specific patch names, but master
  still passed the previous fast-phase h5parm as `fast_initialsolutions_h5parm`.
  The 2026-07-05 changing-facet carry-over run reproduced the issue in a
  tighter two-cycle case: cycle 2 used only `Patch_2..4`, but master still
  passed the previous five-direction fast h5parm. Verify whether DP3 safely
  ignores/remaps unmatched directions or whether this is a master bug; the
  current branch now rejects this unsafe carry-over unless direction
  compatibility is proven.
- Master carries a previous-cycle DI full-Jones product into later imaging when
  the later cycle does not run a DI solve. The 2026-07-05 DI-to-DD
  mode-boundary run shows master `image_2` applying the cycle-1
  `fulljones-solutions.h5`, while the current branch intentionally leaves
  `fulljones_h5parm` unset for `image_2` because the active cycle is DD-only.
  Treat this as legacy implicit-state behavior that should remain documented
  but should not be copied silently into the flexible strategy.

Science-equivalence gate checklist:

Complete this checklist before starting the first scalability slice. Keep the
tracked reports compact and move bulky FITS/MS/log products out of git.

1. **Use branch repeatability controls before tuning tolerances.**
   Use the refreshed normalized DD-plus-DI full-Jones repeatability envelope as
   the main tolerance evidence. When rerunning, use short `/tmp` paths for
   master runs, freeze inputs and resources, run three repeats per branch, and
   compare all same-branch and cross-branch pairs. Classify differences as
   strict-contract failures, repeatability-bounded numeric drift, or documented
   systematic branch differences.

2. **Document and enforce carry-forward policy for flexible strategy products.**
   The policy decision is to avoid silent carry-over after a new calibration
   step. A previous-cycle product may seed a matching solve when the product
   role and, for DD, directions are compatible. It may be reused by an explicit
   image-only cycle. It must not be applied during imaging after a new
   calibration step unless that product was part of the current cycle's
   calibration state. Current-cycle full-Jones imaging application is guarded by
   focused image-operation tests; keep DD direction compatibility checks strict.

3. **Keep option-specific equivalence scoped and explain skipped rows.**
   The initial risk-based matrix is complete for the options most likely to
   regress during this migration: provided normalization sky models, DP3
   image-based predict, WSClean predict, and BDA/averaging all pass against
   `master` on the generated rich demo data. Keep future additions scoped to one
   meaningful option family per row so failures remain attributable. Screens
   remain recorded as skipped until the target environment can run IDGCal/screen
   workflows reproducibly. Keep cubes, QUV/full-Stokes, astrometry, photometry,
   and peeling primarily in saved-reference or integration coverage unless a
   reviewer needs branch-vs-master parity for one of them.

4. **Tighten comparison rules only where the evidence supports it.**
   Add h5parm numeric statistics/tolerances for accepted phase-only drift while
   keeping solset/soltab names, axes, shapes, finite values, and source tables
   strict. Decide whether sparse `field-MFS-model-pb` differences need
   model-specific sparse-outlier tolerances or deeper deconvolution/model-
   selection investigation. Handle source-catalog differences with deterministic
   ordering, semantic comparison, or repeatability-bounded tolerances where
   possible.

5. **Make final master-reference decisions explicit.**
   Decide whether the slow-gain default-like reference should remain a
   documented master bug/legacy limitation, whether the master checkout should
   be patched for an intended-amplitude reference run, and how any intentional
   current-vs-master divergence should be labelled in reports.

6. **Keep the reviewer smoke workflow usable.**
   Update `scripts/dev/run_branch_equivalence.py`,
   `tests/execution/test_branch_equivalence.py`, `EQUIVALENCE_REPORT.md`, and
   tracked report docs as the comparison contract is tightened. Add a
   user-supplied parset smoke scenario: run `--prepare-only` to verify reported
   base/current parsets, strategy files, and work directories, then run one
   documented real comparison so reviewers can reuse the workflow with their
   own datasets.

For a workstation smoke check or user-supplied data, use `--prepare-only` first
to confirm the exact base/current parsets and work directories that will be
reported. The current benchmark strategy intentionally exercises DI-only phase
and full-Jones cycles that `master` cannot represent exactly; use explicit
DD-only/default-like branch-equivalence strategies for scientific
branch-vs-master gates.

For a specific base commit, pass `--base-ref <commit-ish>`. By default the base
checkout is placed under the run root at `checkouts/base` and the venv is
created at `checkouts/base/.venv`. Use `--base-venv` to reuse a different
environment, `--base-install-spec '.[dev]'` if development extras are needed,
`--base-system-site-packages` in prepared containers that provide compiled
astronomy packages globally, and `--reinstall-base-env` when dependencies should
be refreshed.

Keep:

- Product presence checks
- Operation order and `.done` marker checks
- Output-record shape checks
- h5parm dataset name, shape, and value checks
- Sky-model summaries
- Beam-table checks
- Exact text/region checks
- FITS aggregate checks
- FITS residual, finite-mask, WCS/header, pixel, and per-plane checks

Later:

- Add product-class-specific tolerances for dirty images, restored images,
  cubes, beam tables, and mosaics where needed.
- Add optional peak/source-aware checks for final science images.
- Keep bulky reference/current artifacts outside git; commit only compact
  curated reports and scenario manifests.

## Master Feature Catch-Up Track

Comparison basis: `git fetch origin master` on 2026-07-04 confirmed
`origin/master` and local `master` at
`17448437b78583f1eaf38112a524b2dbe5f34bb8` (`Generate residual visibilities`,
2026-07-01). `git log HEAD..origin/master` shows 23 master commits since this
branch diverged. The items below are the master runtime/product features that
are missing or only partially represented in the current Prefect/Dask branch.

Porting standard: preserve the intended behavior from master, but adapt each
feature to this branch's current ownership boundaries and Prefect/Dask execution
contracts. Do not copy legacy script or CWL-era structure when a cleaner
execution-module implementation is available. While porting, clean up local
code paths when that reduces complexity without changing behavior, and improve
tests so they are maintainable, human-readable, and cover the branch-specific
contracts as well as the master behavior being restored.

Port these in order:

1. **Apply small low-risk master fixes first.** Done on 2026-07-04.
   The Measurement Set time-concatenation ordering fix from `bf4608ef` is
   ported into `rapthor/execution/concatenate/measurement_sets.py`, using
   Measurement Set `TIME` values rather than caller order for TAQL
   concatenation. The `modifystate.py` missing-directory guard from `e25b4a6a`
   is also ported so reset does not fail when optional output directories such
   as `visibilities/` do not exist. Focused pytest coverage exists for both.

2. **Restore astrometry-corrected image products.** Implemented on 2026-07-04.
   The `ebe35408` astrometry-correction behavior is now represented under
   `rapthor/execution/image/`: image-sector execution creates the Stokes-I
   `image-pb-ast.fits(.fz)` product after diagnostics, compression preserves it,
   image finalization records `I_image_file_true_sky_astcorr`, and mosaic inputs
   include the product when all sectors provide it. Focused unit, flow,
   finalizer, mosaic, Dask-boundary, and reference-fixture tests pass. Re-run the
   branch-vs-master equivalence scenario to confirm the previously missing
   `field-MFS-image-pb-ast` products are now present in full pipeline outputs.

3. **Add per-facet RMS diagnostics.** Implemented on 2026-07-04.
   The facet-selection helper from `eb1b6f2f` is ported into
   `rapthor/lib/fitsimage.py`, and `calculate_image_diagnostics` now adds
   `facets_rms` for valid facet-region files while omitting it for missing,
   `none`, or unreadable region inputs. Focused coverage exists for FITS facet
   selection, facet RMS statistics, missing/invalid region handling, and the
   diagnostics JSON contract.

4. **Port imaging model-data and residual-visibility products.** Implemented on 2026-07-04.
   The `save_residual_visibilities` parset/default/docs path is ported. Final
   image cycles now ask WSClean to retain/update `MODEL_DATA` when residual
   visibilities are requested, DP3 writes `*_resid.ms` as DATA minus
   MODEL_DATA, residual products are returned by the image flow, and image
   finalization publishes them under `visibilities/image_X/sector_Y`. Focused
   command-builder, payload, flow, finalizer, pipeline-flag, parset, and
   observation tests pass.

5. **Port WSClean-based prediction.** Implemented on 2026-07-04.
   The `use_wsclean_predict` parset/default/docs path is ported. DD calibration
   preparation now creates a prediction region, reads patch names from it, copies
   each input MS before adding model columns, draws narrow-band WSClean model
   images, runs WSClean `-predict` per patch/frequency slice, and wires the
   resulting model columns into DP3 solve commands without restoring the legacy
   `rapthor/scripts/wsclean_predict.py` glue. Focused command-builder, payload,
   flow, operation, parset, and field tests pass.

6. **Align normalization and parallel-gridding semantics with master.** Implemented on 2026-07-05.
   The `fc79ef7f`, `3e4eca19`, `38abdb92`, and `01a81e11` behavior is now
   represented in the Prefect/Dask implementation. `do_normalize` outside the
   first cycle logs a warning instead of raising, flux-density normalization
   handles empty/no-match products more robustly, WSClean parallel gridding is
   modelled as task groups via `parallel_gridding_tasks`, no-DDE and MPI facet
   WSClean commands keep `-parallel-gridding`, and shared facet read/write is
   enabled only when the actual patch/facet count supports it. The LSMTool
   dependency was advanced to `176ef008534bdd929e58c57b00c0a60e3445ad68`; Rapthor
   call sites were updated for the current `read_skymodel(..., wcs=...)` API.

7. **Record what is intentionally already covered or not relevant.** Done on 2026-07-05.
   The remaining commits in `HEAD..origin/master` were audited against
   `origin/master` at `17448437b78583f1eaf38112a524b2dbe5f34bb8`.

   | Commit | Decision |
   | --- | --- |
   | `bf4608ef`, `e25b4a6a`, `eb1b6f2f`, `ebe35408`, `d90786e8`, `971a2b25`, `e8867abd`, `17448437` | Runtime/product behavior is ported in tasks 1-5 above. |
   | `fc79ef7f`, `3e4eca19`, `38abdb92`, `01a81e11` | Normalization, parallel-gridding, and shared-facet semantics are ported in task 6. |
   | `908f83c9` | Superseded by the migrated diagnostic module. Missing/`none` facet-region handling is covered by `compute_facet_rms_noise` tests, so the legacy script patch does not need a separate port. |
   | `37f6fc06`, `0e163498` | Already covered by this branch's Ubuntu 24.04 CI images and libdeflate runtime/build dependencies. The dev container may remain independently tuned for local development. |
   | `c853f707` | Covered or superseded. CI integration splitting/thread caps are represented by this branch's current CI/tox setup, `test_do_normalize` coverage exists here, and the old LSMTool pin is superseded by updating to LSMTool `176ef008534bdd929e58c57b00c0a60e3445ad68` plus current `read_skymodel(..., wcs=...)` call sites. |
   | `f826c262`, `13e3bd8e` | Formatting-only churn. No behavioral port required; formatting is enforced by the repository ruff workflow. |
   | `5f59fe87`, `6c58212b`, `0c422261`, `14377a61`, `4cc0b732` | Test-structure cleanup from master overlaps with this branch's ongoing test maintainability track. Do not copy it wholesale while larger Prefect/Dask-specific fixtures and helpers are still stabilizing; harvest useful patterns as part of the paused test-suite cleanup. |

   No further master-only runtime feature slice is currently known. Re-run the
   phase-only branch-equivalence scenario after these feature ports and update
   `EQUIVALENCE_REPORT.md` with the remaining intentional differences.

After each feature slice, run the focused unit tests for the touched owner
package, then `ruff check --fix --select I`, `ruff format`, and the relevant
saved/branch equivalence checks in the prepared dev container. Re-run the
phase-only branch-equivalence scenario after tasks 2-6 because those tasks
change the exact product set compared with master.

## First Scalability Slice

Status: implemented, benchmark interpretation recorded, keep for now as an
observability improvement.

Implemented slice:

- Image-sector execution has been split around post-WSClean
  preparation/finalization work.
- The task graph should remain Dask-shaped for supported task-runner modes,
  with `sync` reserved as a deterministic focused-test fallback.
- The scientific product contract should remain unchanged.

Remaining verification:

- Explain the July 4 to July 6 wall-time and Dask-gap improvement before
  adding another task boundary.
- Keep the July 7 split unless dashboard noise, focused tests, or product
  checks show a downside.
- If the slice causes product drift, rerun the relevant equivalence scenario
  and fix/revert before continuing.

Guardrails:

- Worker inputs stay plain serializable data, not `Field`, `Observation`,
  `Sector`, operation instances, open file handles, or subprocess state.
- Task names remain readable and include useful operation/sector identifiers.
- Output records and restart behavior remain unchanged.
- Existing image/calibrate product handoff semantics remain unchanged:
  `prepare_data_h5parm` is for DP3 pre-apply; `h5parm` is for imaging-time
  facet/DD application.
- External tools such as DP3, WSClean, IDG, and PyBDSF stay coarse commands
  unless a proven library-level integration exists.

Do not split a second image boundary until the first one has focused tests,
saved-equivalence evidence when needed, and an interpreted benchmark
comparison.

## Test Suite Track

Resume after the benchmark interpretation decides whether the first
scalability slice is kept, adjusted, or reverted. Continue opportunistic test
cleanup only when touching the relevant files.

Keep doing as files are touched:

- Run `ruff check --fix --select I` and `ruff format` for Python code changes.
- Use pytest fixtures, fixture factories, `tmp_path`, `monkeypatch`,
  `pytest.raises`, `caplog`, and table-driven parametrization with readable ids.
- Keep tests clean code and living documentation.
- Avoid duplicated setup and broad helpers that hide the behavior under test.
- Add docstrings when the test name does not explain the scientific or runtime
  reason for the case.
- Replace placeholder `pass` tests with real assertions or delete redundant
  tests. Keep `pass` only for deliberate no-op fake methods.
- Keep slow, external-tool, internet/data-download, Prefect-server, and
  integration behavior behind clear marks or fixtures.

Near-term cleanup backlog:

- Continue deduplicating large execution/operation test setup.
- Split very large flow/operation test modules when touching them:
  `tests/execution/test_image_flow.py`,
  `tests/execution/test_calibrate_flow.py`,
  `tests/execution/test_pipeline_flow.py`,
  `tests/operations/test_image.py`, and
  `tests/operations/test_calibrate.py`.
- Keep `TESTING.md` and `.agents/testing_playbook.md` aligned with test layout,
  marks, integration requirements, and Prefect harness usage.
- Use collection and durations checks during cleanup:

```bash
python -m pytest -m "not integration" tests --collect-only -q
python -m pytest -m "not integration" tests --durations=30 --durations-min=0.25
```

## Runtime UX And Docs

Do after the first scalability slice has settled.

Next useful improvements:

- Expand preflight/dry-run output to show operation order, task groups,
  external tools, resource hints, expected outputs, and unsupported features.
- Improve messages for missing tools, unsupported container setup,
  Slurm/external-Dask mismatch, missing Dask scheduler, and MPI WSClean
  assumptions.
- Show resolved Prefect API mode, Dask scheduler mode, local worker count, and
  dashboard URLs.
- Add concise contributor docs for adding parset options, operation changes,
  external command helpers, execution-owned module adapters, and new flow task
  boundaries.
- Add a short debugging guide covering logs, `.done`, `.outputs.json`,
  command records, Prefect run names, Dask dashboards, and external-command
  stderr.

## Deferred Refactors

Do not split these modules for tidiness alone. Split them when changing
behavior or when a small extraction clearly reduces risk:

- `rapthor.execution.image.diagnostic_calculation`
- `rapthor.execution.image.flux_normalization`
- `rapthor.execution.calibrate.h5parm_combination`
- `rapthor.operations.calibrate.base`
- `rapthor.operations.image.base`
