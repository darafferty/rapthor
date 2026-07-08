# 2026-07-08 Post-Split Filter-Skymodel Profile

Source artifacts:

- `runs/benchmark-20260708-121853`

Raw CI artifacts, Dask HTML reports, command logs, generated products, and run
directories are intentionally not committed. This report keeps the compact
comparison needed to decide what to do after splitting `filter_skymodel` into
its own image-sector Prefect task.

## Run Identity

| Field | Value |
| --- | --- |
| Commit | `9f5ea4885b76b6b2b8da03bd283e93dd687d39c2` |
| CI job | `1833289` |
| Pipeline | `201075` |
| Runner | `astron-docker-1` |
| Ref | `gec-468-ai-migrate-to-prefect` |
| Scenarios | `ci-benchmark-baseline-2x30`, `ci-benchmark-filter-only-15` |
| Repetitions | `3` for each profile |
| Return codes | `0, 0, 0` for each profile |

This run used a different runner from the earlier July 8 profile, so the paired
baseline-vs-filter comparison within this run is the decision-quality evidence.
Do not compare absolute wall times directly with runs from another runner.

## Summary

| Profile | Shape | Wall Median (s) | Wall Min-Max (s) | Wall Delta vs Baseline | Command Median (s) | Dask Gap Median (s) | Task Count |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| `baseline-2x30` | `2` workers x `30` threads, `max_threads=30`, `filter_skymodel_ncores=max_threads` | `213.347` | `194.119-233.012` | `0.000` (`0.00%`) | `157.881` | `46.550` | `20` |
| `filter-only-15` | `2` workers x `30` threads, `max_threads=30`, `filter_skymodel_ncores=15` | `183.968` | `183.907-210.826` | `-29.379` (`-13.77%`) | `135.576` | `45.990` | `20` |

The task split is visible in Dask: task count is now `20`, including four
`image_sector_filter_skymodel_task` tasks.

## Filter Task Evidence

Median Dask task-group totals:

| Profile | `image_sector_prepare_task` (s) | `image_sector_filter_skymodel_task` (s) | `image_sector_finalize_task` (s) |
| --- | ---: | ---: | ---: |
| `baseline-2x30` | `56.247` | `89.881` | `5.180` |
| `filter-only-15` | `51.772` | `70.154` | `5.154` |
| Delta | `-4.474` | `-19.727` | `-0.026` |

Median command totals:

| Profile | `filter_skymodel` (s) | `wsclean` (s) | `python3` (s) | `DP3` (s) | `fpack` (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `baseline-2x30` | `89.540` | `53.592` | `9.776` | `2.712` | `2.106` |
| `filter-only-15` | `69.881` | `50.381` | `9.668` | `2.687` | `2.097` |
| Delta | `-19.659` | `-3.211` | `-0.108` | `-0.025` | `-0.009` |

Per-repetition wall times:

| Profile | Rep 1 (s) | Rep 2 (s) | Rep 3 (s) |
| --- | ---: | ---: | ---: |
| `baseline-2x30` | `233.012` | `213.347` | `194.119` |
| `filter-only-15` | `210.826` | `183.907` | `183.968` |

Per-repetition `filter_skymodel` command totals:

| Profile | Rep 1 (s) | Rep 2 (s) | Rep 3 (s) |
| --- | ---: | ---: | ---: |
| `baseline-2x30` | `111.127` | `89.540` | `77.389` |
| `filter-only-15` | `94.799` | `69.623` | `69.881` |

Both profiles have a slower first repetition, so cache/warm-up effects are
still visible. The paired comparison nevertheless repeats the earlier positive
signal, and the improvement is now visible in the dedicated filter task group.

## Decision

The post-split benchmark supports promoting `filter_skymodel_ncores=15` from a
benchmark profile to the proposed production default for this CI-sized
configuration.

Keep the global `2x30`, `max_threads=30` worker/thread shape. The improvement is
isolated to the filter step without a WSClean regression, and the Dask
duration-minus-compute gap is essentially unchanged.

## Next Step

Change the default so `filter_skymodel` uses `15` cores unless the user
overrides it explicitly, then run one confirmation benchmark. The confirmation
run should show that the default profile matches the previous
`filter-only-15` behavior and should keep the raw/scientific product contract
unchanged.

