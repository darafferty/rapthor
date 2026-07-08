# 2026-07-08 Filter-Skymodel-Only Resource Profile

Source artifacts:

- `runs/benchmark-20260708-073531`

Raw CI artifacts, Dask HTML reports, command logs, generated products, and run
directories are intentionally not committed. This report keeps the compact
comparison needed to decide what to do after adding the non-default
`filter_skymodel_ncores` control.

## Run Identity

| Field | Value |
| --- | --- |
| Commit | `2a86f8ac0293e016f3fcd59e618e4452bc57532f` |
| CI job | `1833030` |
| Pipeline | `201045` |
| Runner | Shared Runner on `lcs126` |
| Ref | `gec-468-ai-migrate-to-prefect` |
| Scenarios | `ci-benchmark-baseline-2x30`, `ci-benchmark-filter-only-15` |
| Repetitions | `3` for each profile |
| Return codes | `0, 0, 0` for each profile |

## Summary

| Profile | Shape | Wall Median (s) | Wall Min-Max (s) | Wall Delta vs Baseline | Command Median (s) | Dask Gap Median (s) | Task Count |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| `baseline-2x30` | `2` workers x `30` threads, `max_threads=30`, `filter_skymodel_ncores=max_threads` | `311.315` | `307.440-359.387` | `0.000` (`0.00%`) | `230.657` | `73.030` | `16` |
| `filter-only-15` | `2` workers x `30` threads, `max_threads=30`, `filter_skymodel_ncores=15` | `300.479` | `299.678-303.515` | `-10.836` (`-3.48%`) | `224.181` | `71.410` | `16` |

The worst `filter-only-15` repetition was still faster than the best baseline
repetition by about `3.9 s`. The baseline includes one clear slow repetition,
but the filter-only profile is also consistently less variable.

## Command Timing

Median command totals:

| Profile | `filter_skymodel` (s) | `wsclean` (s) | `python3` (s) | `DP3` (s) | `fpack` (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `baseline-2x30` | `109.771` | `89.082` | `17.804` | `8.280` | `4.144` |
| `filter-only-15` | `105.399` | `87.170` | `17.730` | `8.211` | `4.137` |
| Delta | `-4.373` | `-1.911` | `-0.074` | `-0.069` | `-0.007` |

Per-repetition `filter_skymodel` totals:

| Profile | Rep 1 (s) | Rep 2 (s) | Rep 3 (s) |
| --- | ---: | ---: | ---: |
| `baseline-2x30` | `109.771` | `131.543` | `109.221` |
| `filter-only-15` | `105.609` | `105.190` | `105.399` |

Per-repetition WSClean totals:

| Profile | Rep 1 (s) | Rep 2 (s) | Rep 3 (s) |
| --- | ---: | ---: | ---: |
| `baseline-2x30` | `89.082` | `105.133` | `88.490` |
| `filter-only-15` | `87.933` | `87.170` | `86.885` |

The slow baseline repetition affected both `filter_skymodel` and WSClean, so
it should be treated as run noise rather than a filter-specific regression.
Even ignoring that outlier, `filter-only-15` remains a small positive signal.

## Decision

Keep the global `2x30`, `max_threads=30` worker/thread shape as the baseline.
The filter-only control is promising and should stay available, but do not
change the production default yet.

Before making `filter_skymodel_ncores=15` the default, split
`filter_skymodel` into its own Prefect task and rerun this comparison. The
split should make the dashboard and Dask report show whether the improvement is
actually isolated to source/skymodel filtering and whether the lower core count
continues to help once the task graph exposes the filter step directly.

## Next Step

Implement the `filter_skymodel` task boundary, preserving the scientific output
contract and existing command behavior, then rerun:

- `ci-benchmark-baseline-2x30`
- `ci-benchmark-filter-only-15`

If the post-split benchmark repeats this improvement without increasing command
failures, memory pressure, or science-product differences, promote
`filter_skymodel_ncores=15` from a benchmark profile to the proposed default.
