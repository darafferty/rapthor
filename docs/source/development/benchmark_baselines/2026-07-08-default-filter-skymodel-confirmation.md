# 2026-07-08 Default Filter-Skymodel Confirmation Benchmark

Source artifacts:

- `runs/benchmark-20260708-191446`

Raw CI artifacts, Dask HTML reports, command logs, generated products, and run
directories are intentionally not committed. This report keeps the compact
evidence needed to decide whether the promoted `filter_skymodel_ncores=15`
default behaves like the previous explicit `filter-only-15` benchmark profile.

## Run Identity

| Field | Value |
| --- | --- |
| Commit | `bdea140c003c729a844b7d698fe45b0fe90c7e1a` |
| CI job | `1833982` |
| Pipeline | `201143` |
| Runner | Shared Runner on `lcs126` |
| Ref | `gec-468-ai-migrate-to-prefect` |
| Scenarios | `ci-benchmark-baseline-2x30`, `ci-benchmark-filter-only-15` |
| Repetitions | `3` for each profile |
| Return codes | `0, 0, 0` for each profile |

Both materialized runtime parsets used:

- `local_dask_workers = 2`
- `cpus_per_task = 30`
- `max_threads = 30`
- `filter_skymodel_ncores = 15`
- `prefect_publish_fits_previews = False`

This means the `baseline-2x30` scenario is now the packaged-default behavior,
and `filter-only-15` is the same setting applied explicitly by the benchmark
profile.

## Summary

| Profile | Shape | Wall Median (s) | Wall Min-Max (s) | Wall Delta vs Default | Command Median (s) | Dask Gap Median (s) | Task Count |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| `baseline-2x30` | `2` workers x `30` threads, `max_threads=30`, `filter_skymodel_ncores=15` by default | `319.800` | `319.084-319.983` | `0.000` (`0.00%`) | `239.036` | `73.750` | `20` |
| `filter-only-15` | `2` workers x `30` threads, `max_threads=30`, `filter_skymodel_ncores=15` explicitly | `314.896` | `311.505-320.070` | `-4.904` (`-1.53%`) | `233.976` | `73.330` | `20` |

The two profiles are within normal CI-run variance. The explicit profile is
slightly faster in this paired run, but both profiles now use the same resource
settings and produce the same task graph size.

## Image-Sector Task Evidence

Median Dask task-group totals:

| Profile | `image_sector_prepare_task` (s) | `image_sector_filter_skymodel_task` (s) | `image_sector_finalize_task` (s) |
| --- | ---: | ---: | ---: |
| `baseline-2x30` | `98.868` | `111.409` | `11.888` |
| `filter-only-15` | `95.395` | `109.677` | `11.905` |
| Delta | `-3.472` | `-1.732` | `+0.016` |

Median command totals:

| Profile | `filter_skymodel` (s) | `wsclean` (s) | `python3` (s) | `DP3` (s) | `fpack` (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `baseline-2x30` | `110.814` | `94.928` | `18.252` | `8.299` | `4.149` |
| `filter-only-15` | `109.095` | `92.004` | `18.309` | `9.627` | `4.149` |
| Delta | `-1.718` | `-2.924` | `+0.056` | `+1.328` | `+0.001` |

The Dask duration-minus-compute gap is effectively unchanged (`73.750 s` vs
`73.330 s`), so the default promotion did not add scheduler overhead.

## Decision

The confirmation benchmark passes.

The packaged default now matches the explicit `filter-only-15` resource setting,
the paired wall time is within CI variance, and the image-sector task split
continues to expose `filter_skymodel` as a measurable task group without
increasing the Dask task count beyond the expected `20`.

Treat the current image-sector split as the stable baseline for the next
scalability slice.

## Next Step

Move to the next planned observability split: make
`calculate_image_diagnostics` its own image-sector task. Add focused tests for
payload shape, serializability, task names, output records, and restart behavior,
then rerun the focused image tests and this CI-sized benchmark.
