# 2026-07-09 Image Diagnostics Task-Split Benchmark

Source artifacts:

- `runs/benchmark-20260709-065455`

Raw CI artifacts, Dask HTML reports, command logs, generated products, and run
directories are intentionally not committed. This report keeps the compact
evidence needed to decide whether splitting `calculate_image_diagnostics` into
its own image-sector task is acceptable.

## Run Identity

| Field | Value |
| --- | --- |
| Commit | `5659abdacf66b954684a56645190ee70337c67c4` |
| CI job | `1834483` |
| Pipeline | `201181` |
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

The `filter-only-15` profile was still present in this CI run but is now
redundant because `filter_skymodel_ncores = 15` is the packaged default.

## Comparison Against Previous Baseline

Previous reference:

- `docs/source/development/benchmark_baselines/2026-07-08-default-filter-skymodel-confirmation.md`

| Profile | Wall Median (s) | Delta vs Previous | Command Median (s) | Delta vs Previous | Dask Gap Median (s) | Task Count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline-2x30` | `307.153` | `-12.647` (`-3.95%`) | `228.222` | `-10.814` (`-4.52%`) | `73.410` | `24` |
| `filter-only-15` | `317.159` | `+2.263` (`+0.72%`) | `238.526` | `+4.550` (`+1.94%`) | `73.500` | `24` |

The task count increased from `20` to `24`, which is exactly the expected four
new `calculate_image_diagnostics` tasks. The Dask duration-minus-compute gap
stayed effectively flat (`73.750 s` to `73.410 s` for the default profile), so
the extra boundary did not introduce measurable scheduler overhead.

The default profile is faster than the previous paired baseline. The redundant
explicit `filter-only-15` profile is slightly slower than its previous value but
still within the overlapping CI variance band from the earlier run.

## Image-Sector Task Evidence

| Metric | Previous Default (s) | Current Default (s) | Delta |
| --- | ---: | ---: | ---: |
| `image_sector_prepare_task` total | `98.868` | `94.266` | `-4.602` |
| `image_sector_filter_skymodel_task` total | `111.409` | `105.872` | `-5.537` |
| `image_sector_finalize_task` total | `11.888` | `4.833` | `-7.055` |
| `image_sector_diagnostics_task` total | n/a | `7.633` | n/a |
| diagnostics + finalize total | `11.888` | `12.466` | `+0.578` |

The split moves the diagnostics work out of `finalize` and makes it visible as a
separate Dask task group. The combined diagnostics-plus-finalize total is only
`0.578 s` above the previous combined task, while the whole default-profile
wall time improved by `12.647 s`.

Median command totals for the default profile:

| Command | Previous (s) | Current (s) | Delta |
| --- | ---: | ---: | ---: |
| `filter_skymodel` | `110.814` | `105.303` | `-5.511` |
| `wsclean` | `94.928` | `91.090` | `-3.838` |

The leaf command sequence remained stable at `44` command records per
repetition.

## Decision

The diagnostics task split is accepted.

It improves dashboard observability, gives `calculate_image_diagnostics` its own
benchmarkable task group, preserves successful runs and command counts, and does
not add measurable scheduler overhead in this CI-sized benchmark.

## Next Step

Use `ci-benchmark-baseline-2x30` as the default benchmark scenario going
forward. The next scalability work should add targeted scenarios for currently
hidden scaling paths before splitting more tasks.
