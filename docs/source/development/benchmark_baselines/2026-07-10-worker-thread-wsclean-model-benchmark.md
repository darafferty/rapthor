# 2026-07-10 Worker-Thread And WSClean Model Benchmark

Source artifacts:

- `runs/benchmark-20260710-072456`

Raw CI artifacts, Dask HTML reports, command logs, generated products, and run
directories are intentionally not committed. This report preserves the compact
evidence needed to judge the Dask worker-thread runtime change and the
WSClean-rendered model-mosaic path against the accepted hidden-path benchmark
baseline.

This run was produced from commit `0b4db147ff2f67257c991c790d31b9eb30385842`.
It includes the one-thread-per-Dask-worker runtime fix and WSClean-rendered
model mosaics. It does not include the later lossless `fpack -g -q 0` sector
model-image compression fix from commit `3fb5d91c`.

## Run Identity

| Field | Value |
| --- | --- |
| Commit | `0b4db147ff2f67257c991c790d31b9eb30385842` |
| CI job | `1835926` |
| Pipeline | `201347` |
| Runner | Shared Runner on `lcs126` |
| Ref | `gec-468-ai-migrate-to-prefect` |
| Scenarios | default, image-products, WSClean-predict, many-sector mosaic |
| Repetitions | `3` for each profile |
| Return codes | `0, 0, 0` for every profile |

All materialized runtime parsets used the CI-sized resource shape:

- `local_dask_workers = 2`
- `cpus_per_task = 30`
- `max_threads = 30`
- `filter_skymodel_ncores = 15`
- `prefect_publish_fits_previews = False`

The Dask reports showed `2` workers and `2` total threads, consistent with one
Prefect task-engine thread per worker process.

## Comparison Against 2026-07-09 Hidden-Path Baseline

| Scenario | Wall Median Before (s) | Wall Median After (s) | Wall Delta | Command Median Before (s) | Command Median After (s) | Command Delta | Task Count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline-2x30` | `305.360` | `306.210` | `+0.3%` | `226.839` | `224.891` | `-0.9%` | `24 -> 24` |
| `image-products-baseline-2x30` | `343.803` | `343.648` | `0.0%` | `256.737` | `257.043` | `+0.1%` | `28 -> 28` |
| `WSClean-predict-baseline-2x30` | `288.767` | `305.434` | `+5.8%` | `209.787` | `227.347` | `+8.4%` | `24 -> 24` |
| `many-sector-mosaic-baseline-2x30` | `412.910` | `633.808` | `+53.5%` | `784.321` | `720.884` | `-8.1%` | `120 -> 120` |

## Headline Result

The worker-thread runtime fix is acceptable for the default and image-products
profiles: wall time, aggregate command time, command counts, and task counts
remain effectively unchanged.

The WSClean-predict profile is slower by about `5.8%` wall time. The command
delta is concentrated in WSClean (`91.709 s` to `107.449 s` aggregate median),
with the same command and task counts. Treat this as a watch item rather than a
blocker, because the run-to-run resource shape and task graph stayed stable.

The many-sector mosaic profile needs follow-up before it can be considered an
accepted performance baseline. It completed reliably, but wall time increased
from about `413 s` to `634 s` even though aggregate profiled command time fell
from about `784 s` to `721 s`. That means the regression is not explained by
leaf-command cost alone.

## Many-Sector Mosaic Evidence

Median many-sector task-group timing from the new run:

| Task Group | Count | Aggregate Task Time (s) | Median Max Task (s) |
| --- | ---: | ---: | ---: |
| `image_sector_prepare_task` | `16` | `334.807` | `25.827` |
| `image_sector_filter_skymodel_task` | `16` | `307.143` | `24.448` |
| `mosaic_task` | `24` | `300.059` | `17.193` |
| `image_sector_finalize_task` | `16` | `12.827` | `1.137` |
| `predict_model_data_task` | `18` | `9.763` | `1.122` |
| `image_sector_diagnostics_task` | `16` | `6.676` | `0.497` |
| `calibrate_chunk_task` | `4` | `6.357` | `2.909` |
| `mosaic_template_task` | `4` | `6.076` | `2.142` |
| `predict_postprocess_task` | `6` | `1.520` | `0.284` |

Median command totals for the same profile:

| Command | New Median (s) | Baseline Median (s) | Delta |
| --- | ---: | ---: | ---: |
| `wsclean` | `327.481` | `307.048` | `+20.433` |
| `filter_skymodel` | `304.867` | `330.306` | `-25.439` |
| `DP3` | `21.654` | `106.825` | `-85.171` |
| `fpack` | `19.770` | `19.032` | `+0.738` |
| `python3` | `17.717` | `17.610` | `+0.107` |

The new `mosaic_task` aggregate timing is the strongest signal. The profile now
uses WSClean to render model mosaics, which is scientifically cleaner than
interpolating sparse model pixels, but it introduces extra mosaic-task work.
The wall-time increase should therefore be investigated as a scheduling and
mosaic-concurrency issue rather than assumed to be an external-command
regression.

## Decision

Accept the worker-thread runtime fix for the default and image-products paths.
Keep the WSClean-predict slowdown as a monitored variance item.

Do not yet accept this run as the new many-sector mosaic performance baseline.
Use it as evidence that the WSClean-rendered model-mosaic path is reliable, but
isolate the many-sector wall-time increase before splitting additional mosaic
work or using this profile for performance-equivalence claims.

## Next Step

Run a targeted many-sector comparison that separates:

- sparse FITS model-mosaic fallback vs WSClean-rendered model mosaics
- one-thread-per-worker task execution vs the previous worker-thread shape, if
  the Prefect settings-cache failure can be avoided in a controlled run
- model-mosaic products vs non-model mosaic products

Keep the lossless sector-model compression fix in the next benchmark run, but
interpret it separately: it protects intermediate `*-model-pb.fits.fz`
correctness and should have small file-size impact for sparse model products.
