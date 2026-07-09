# 2026-07-09 Hidden-Path Pre-Split Benchmark Baseline

Source artifacts:

- `runs/benchmark-20260709-171403`

Raw CI artifacts, Dask HTML reports, command logs, generated products, and run
directories are intentionally not committed. This report preserves the compact
evidence needed to compare later task-splitting batches against the current
pre-split hidden-path baseline.

## Run Identity

| Field | Value |
| --- | --- |
| Commit | `5a6b86202c900ad65f6fc7feca0cfb8fb807aea0` |
| CI job | `1835525` |
| Pipeline | `201315` |
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

## Headline Results

| Scenario | Wall Median (s) | Wall Min-Max (s) | Command Median (s) | Command Count | Dask Duration (s) | Dask Compute (s) | Dask Gap (s) | Task Count |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ci-benchmark-baseline-2x30` | `305.360` | `304.486-310.879` | `226.839` | `44` | `292.050` | `219.470` | `72.480` | `24` |
| `ci-benchmark-image-products-baseline-2x30` | `343.803` | `343.643-345.578` | `256.737` | `51` | `330.450` | `253.280` | `77.210` | `28` |
| `ci-benchmark-wsclean-predict-baseline-2x30` | `288.767` | `287.805-290.586` | `209.787` | `56` | `275.470` | `197.700` | `78.170` | `24` |
| `ci-benchmark-many-sector-mosaic-baseline-2x30` | `412.910` | `411.132-418.849` | `784.321` | `135` | `399.210` | n/a | n/a | `120` |

The many-sector command median is larger than wall time because sector commands
run concurrently; command seconds are summed leaf-command time, not elapsed
pipeline time. The Dask report parser recovered duration, task count, and task
groups for the many-sector profile, but did not recover compute/gap values from
that HTML report.

## Scenario Coverage

| Scenario | What It Exercises | Key Task Groups |
| --- | --- | --- |
| `ci-benchmark-baseline-2x30` | Current default CI-sized rich demo path | 4 image-sector prepare/filter/diagnostics/finalize tasks, 4 calibrate chunks, 2 predict task pairs |
| `ci-benchmark-image-products-baseline-2x30` | Image post-processing hidden paths: normalization, image cubes, catalog creation, restoration/compression | 5 image-sector task sets, 4 calibrate chunks, 2 predict task pairs |
| `ci-benchmark-wsclean-predict-baseline-2x30` | WSClean prediction-heavy calibration path | Same visible task structure as default, but 16 WSClean command records |
| `ci-benchmark-many-sector-mosaic-baseline-2x30` | 2x2 sector imaging, per-sector prediction, regridding, mosaic assembly, compression | 16 image-sector task sets, 24 mosaic tasks, 18 predict-model tasks |

## Bottleneck Evidence

Median command totals:

| Scenario | `filter_skymodel` (s) | `wsclean` (s) | `DP3` (s) | `fpack` (s) | Other notable command |
| --- | ---: | ---: | ---: | ---: | --- |
| `baseline` | `105.778` | `89.149` | `8.219` | `4.172` | `python3`: `17.867 s` |
| `image-products` | `114.274` | `101.246` | `8.498` | `7.184` | `make_catalog_from_image_cube`: `6.160 s` |
| `WSClean-predict` | `86.675` | `91.709` | `7.694` | `4.142` | `python3`: `17.790 s` |
| `many-sector mosaic` | `330.306` | `307.048` | `106.825` | `19.032` | `python3`: `17.610 s` |

Median operation-level observations:

- Image-products adds a visible `normalize_1` operation of about `19.137 s`,
  including `16.026 s` of profiled commands.
- The image-products scenario adds one extra image-sector path and one
  `make_catalog_from_image_cube` command, raising task count from `24` to `28`.
- WSClean-predict increases command count from `44` to `56`, but does not add
  new visible task groups; this is a good candidate for finer calibration
  prediction task boundaries.
- The many-sector mosaic scenario is the best current scaling-path baseline:
  it adds 16 image-sector task sets and 24 mosaic tasks and completes reliably
  in about `413 s` wall time.
- Many-sector image and predict operations show negative
  operation-minus-command values because multiple sector commands run in
  parallel and command totals are aggregate command CPU/wall time, not serial
  elapsed time.

## Decision

The pre-split hidden-path benchmark baseline is accepted.

It covers the currently hidden paths we wanted before splitting more large work
units: image post-processing, WSClean-predict calibration, and multi-sector
mosaic/regridding. All profiles completed successfully across three repeats,
with stable wall-time ranges and useful command/task-group evidence.

## Next Step

Use this report as the comparison baseline for future task-splitting batches.
Before changing mosaic execution, fix and guard the sparse `MFS-model-pb` mosaic
artifact noted in `PLAN.md`, then split owner-package work in batches and rerun
the relevant hidden-path profile after each batch.
