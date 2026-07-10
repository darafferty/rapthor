# 2026-07-10 Worker-Shape And Mosaic-Method Comparison

Source artifacts:

- `runs/benchmark-20260710-185040`

Raw CI artifacts, Dask HTML reports, command logs, generated products, and run
directories are intentionally not committed. This report preserves the compact
evidence needed to choose the next task-splitting and benchmarking shape.

## Run Identity

| Field | Value |
| --- | --- |
| Commit | `82c2390d0d212f50ba37fdd75a2da249594e902c` |
| CI job | `1836574` |
| Pipeline | `201409` |
| Runner | Shared Runner on `lcs126` |
| Ref | `gec-468-ai-migrate-to-prefect` |
| Repetitions | `1` per scenario/profile |
| Return codes | `0` for all six runs |

The run compared two resource profiles:

- `baseline-2x30`: `local_dask_workers=2`, `cpus_per_task=30`,
  `max_threads=30`
- `filter-workers-4x15`: `local_dask_workers=4`, `cpus_per_task=15`,
  `max_threads=15`

Both profiles keep the same total external-command thread budget on the
60-core CI runner, but the `4x15` shape exposes more independent Prefect/Dask
work at once.

## Headline Results

| Scenario | Profile | Wall (s) | Command Total (s) | Dask Duration (s) | Task Count | Commands |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `ci-benchmark` | `2x30` | `312.153` | `227.536` | `294.970` | `24` | `47` |
| `ci-benchmark` | `4x15` | `310.356` | `229.581` | `297.040` | `24` | `47` |
| `ci-benchmark-many-sector-mosaic` | `2x30` | `543.793` | `578.655` | `530.160` | `120` | `151` |
| `ci-benchmark-many-sector-mosaic` | `4x15` | `372.806` | `632.163` | `359.070` | `120` | `151` |
| `ci-benchmark-many-sector-mosaic-sparse-fallback` | `2x30` | `615.050` | `707.348` | n/a | `120` | `147` |
| `ci-benchmark-many-sector-mosaic-sparse-fallback` | `4x15` | `428.808` | `797.318` | `415.110` | `120` | `147` |

## Paired Comparisons

| Comparison | Wall Change | Command Change | Interpretation |
| --- | ---: | ---: | --- |
| Default `ci-benchmark`, `4x15` vs `2x30` | `-0.6%` | `+0.9%` | Neutral. The default single-sector-rich path does not expose enough independent work for the extra workers to matter. |
| Many-sector WSClean model mosaics, `4x15` vs `2x30` | `-31.4%` | `+9.2%` | Strong throughput gain. Individual command totals rise slightly, but more sector/mosaic work overlaps. |
| Many-sector sparse fallback, `4x15` vs `2x30` | `-30.3%` | `+12.7%` | Same scheduling signal: more workers improve elapsed time despite higher aggregate command seconds. |
| WSClean model mosaics vs sparse fallback, `2x30` | `-11.6%` | `-18.2%` | WSClean-rendered model mosaics are faster than sparse fallback in the same worker shape. |
| WSClean model mosaics vs sparse fallback, `4x15` | `-13.1%` | `-20.7%` | WSClean remains faster under the higher-concurrency worker shape. |

## Task-Group Evidence

The many-sector profiles expose the expected task groups:

- `16` image-sector prepare/filter/diagnostics/finalize task groups
- `24` mosaic tasks
- `18` prediction model-data tasks
- `4` calibration chunk tasks

For `ci-benchmark-many-sector-mosaic-filter-workers-4x15`, dominant aggregate
task-group times were:

| Task Group | Count | Aggregate Task Time (s) | Median Task Time (s) |
| --- | ---: | ---: | ---: |
| `image_sector_filter_skymodel_task` | `16` | `328.626` | `25.345` |
| `mosaic_task` | `24` | `259.253` | `12.751` |
| `image_sector_prepare_task` | `16` | `255.148` | `15.322` |
| `predict_model_data_task` | `18` | `14.869` | `0.478` |
| `image_sector_finalize_task` | `16` | `12.684` | `0.999` |

The aggregate command totals are larger than elapsed operation time in the
many-sector runs because sector commands execute concurrently. That is the
desired scheduling behavior for this scalability track.

## Decision

This run resolves the immediate follow-up from the previous worker-thread and
WSClean model-mosaic benchmark:

- The six-way comparison completed successfully with return code `0` for every
  scenario/profile.
- `4x15` is effectively neutral for the default single-sector benchmark and is
  clearly better for many-sector mosaics on this runner.
- WSClean-rendered model mosaics are faster than the sparse FITS fallback in
  both worker shapes, so WSClean should remain the preferred path. Keep sparse
  mapping as a fallback for products without sky-model/component-list inputs.

Use `4x15` as the preferred comparison shape for future many-sector
observability/scalability batches. Keep `2x30` in the benchmark matrix when we
need a paired worker-shape comparison, but do not block task-splitting work on
the older `2x30` many-sector shape.

## Next Step

Proceed with the next calibration task split:

- split `process_slow_gains`
- split full-Jones normalization
- split `combine_h5parms`
- split `plot_solutions`

Benchmark after the batch with the relevant hidden-path calibration scenarios
and keep the many-sector `4x15` profile available for changes that touch image
or mosaic scheduling.
