# 2026-07-08 Filter-Skymodel Resource Profile Benchmark

Source artifacts:

- `runs/benchmark-20260707-191141`

Raw CI artifacts, Dask HTML reports, command logs, generated products, and run
directories are intentionally not committed. This report keeps the compact
comparison needed to decide whether any tested worker/thread profile should
replace the current `2x30` CI benchmark shape.

## Run Identity

| Field | Value |
| --- | --- |
| Commit | `32e64149cd948dc12e21fec8b7533166858edae1` |
| CI job | `1832616` |
| Pipeline | `201009` |
| Runner | Shared Runner on `lcs126` |
| Ref | `gec-468-ai-migrate-to-prefect` |
| Scenario | `ci-benchmark` |
| Repetitions | `3` for each resource profile |
| Return codes | `0, 0, 0` for each resource profile |

## Summary

| Profile | Shape | Wall Median (s) | Wall Delta vs Baseline | Command Median (s) | Dask Gap Median (s) | Task Count |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `baseline-2x30` | `2` workers x `30` threads, `max_threads=30` | `306.507` | `0.000` (`0.00%`) | `228.392` | `72.450` | `16` |
| `filter-threads-15` | `2` workers x `30` threads, `max_threads=15` | `305.529` | `-0.977` (`-0.32%`) | `227.387` | `72.470` | `16` |
| `filter-wide-1x60` | `1` worker x `60` threads, `max_threads=60` | `347.278` | `+40.772` (`+13.30%`) | `271.584` | `71.080` | `16` |
| `filter-workers-4x15` | `4` workers x `15` threads, `max_threads=15` | `322.129` | `+15.622` (`+5.10%`) | `240.928` | `75.600` | `16` |

## Command Timing

| Profile | `filter_skymodel` Delta | `wsclean` Delta | `DP3` Delta | Interpretation |
| --- | ---: | ---: | ---: | --- |
| `filter-threads-15` | `-4.378 s` | `+6.223 s` | `-2.857 s` | Slightly faster filtering is cancelled by slower WSClean; wall time is neutral. |
| `filter-wide-1x60` | `+10.245 s` | `+26.438 s` | `+6.434 s` | One wide worker is clearly worse for this benchmark. |
| `filter-workers-4x15` | `-1.750 s` | `+17.183 s` | `-2.866 s` | More smaller workers slow WSClean enough to lose wall time. |

Median command totals:

| Profile | `filter_skymodel` (s) | `wsclean` (s) | `python3` (s) | `DP3` (s) | `fpack` (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `baseline-2x30` | `109.566` | `87.066` | `17.784` | `8.230` | `4.151` |
| `filter-threads-15` | `105.187` | `93.289` | `17.812` | `5.372` | `4.159` |
| `filter-wide-1x60` | `119.811` | `113.504` | `17.841` | `14.664` | `4.149` |
| `filter-workers-4x15` | `107.816` | `104.249` | `17.827` | `5.364` | `4.128` |

## Decision

Keep the current `baseline-2x30` resource shape for now.

The benchmark does not support changing global worker/thread defaults:

- `filter-threads-15` is effectively tied with baseline on wall time, and its
  small `filter_skymodel` improvement is offset by slower WSClean.
- `filter-wide-1x60` and `filter-workers-4x15` are slower by a clear margin.
- The Dask duration-minus-compute gap remains about `72 s` for the baseline and
  `filter-threads-15`, so the next bottleneck is external command/resource
  behavior rather than scheduler gap.
- Task count remains `16`, confirming that these profiles tested resource
  shape, not graph granularity.

## Next Hypothesis

The resource profiles in this run only varied `cluster.max_threads`, which
affects WSClean, DP3-related command builders, and `filter_skymodel` together.
The next benchmark should use the new non-default, filter-only resource control
so we can compare:

- WSClean/DP3 remaining at the current `2x30`, `max_threads=30` baseline.
- `filter_skymodel` running with a smaller `ncores` value such as `15`.

Do not adopt this as a default until a follow-up benchmark shows stable wall
time improvement without increasing memory pressure or reducing scientific
coverage.
