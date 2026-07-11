# 2026-07-10 Pre-Calibration-Postprocess-Split Benchmark

Source artifacts:

- `runs/benchmark-20260710-214235`

Raw CI artifacts, Dask HTML reports, command logs, generated products, and run
directories are intentionally not committed. This report preserves the compact
baseline needed to judge the calibration post-processing task split.

## Run Identity

| Field | Value |
| --- | --- |
| Commit | `50b11de646ab503c24ba2806deeabdc5ed1bd221` |
| CI job | `1836621` |
| Pipeline | `201416` |
| Runner | Shared Runner on `lcs126` |
| Ref | `gec-468-ai-migrate-to-prefect` |
| Repetitions | `1` per scenario |
| Return codes | `0` for both runs |

The run used the preferred automatic benchmark shape:

- `local_dask_workers=4`
- `cpus_per_task=15`
- `max_threads=15`
- `filter_skymodel_ncores=15`

## Headline Results

| Scenario | Wall (s) | Command Total (s) | Dask Duration (s) | Dask Compute (s) | Dask Gap (s) | Dask Tasks | Commands |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ci-benchmark` | `316.499` | `229.340` | `299.110` | `245.430` | `53.680` | `40` | `47` |
| `ci-benchmark-wsclean-predict` | `294.675` | `214.185` | `281.360` | `225.060` | `56.300` | `40` | `59` |

## Calibration Evidence

Before the calibration post-processing split, all per-solve processing,
solution plotting, h5parm combination, DD source adjustment, and output-record
assembly were hidden inside `finalize_solutions_task`.

| Scenario | `finalize_solutions_task` Count | Aggregate Task Time (s) | Median Task Time (s) | Max Task Time (s) |
| --- | ---: | ---: | ---: | ---: |
| `ci-benchmark` | `4` | `19.873` | `4.099` | `9.847` |
| `ci-benchmark-wsclean-predict` | `4` | `19.636` | `4.021` | `9.774` |

The calibration operation timings provide a stable before-split comparison:

| Scenario | Operation | Elapsed (s) | Command Total (s) | Operation-Command Gap (s) | Commands |
| --- | --- | ---: | ---: | ---: | ---: |
| `ci-benchmark` | `calibrate_di_1` | `6.152` | `4.078` | `2.074` | `5` |
| `ci-benchmark` | `calibrate_2` | `7.133` | `5.532` | `1.601` | `5` |
| `ci-benchmark` | `calibrate_3` | `13.177` | `11.470` | `1.707` | `10` |
| `ci-benchmark` | `calibrate_di_4` | `3.133` | `1.933` | `1.200` | `3` |
| `ci-benchmark-wsclean-predict` | `calibrate_di_1` | `6.131` | `4.097` | `2.034` | `5` |
| `ci-benchmark-wsclean-predict` | `calibrate_2` | `9.136` | `7.593` | `1.543` | `11` |
| `ci-benchmark-wsclean-predict` | `calibrate_3` | `15.152` | `13.639` | `1.513` | `16` |
| `ci-benchmark-wsclean-predict` | `calibrate_di_4` | `3.145` | `1.923` | `1.222` | `3` |

## Decision For The Next Run

Use this run as the before-split calibration baseline. The next benchmark
should keep `ci-benchmark` and `ci-benchmark-wsclean-predict` so command totals,
operation timings, and total wall time remain comparable.

Add a narrow `ci-benchmark-calibration-postprocess` scenario to isolate the
new task boundaries without image/filter/mosaic wall time dominating the
profile. The next run should show:

- new visible task groups: `process_solutions_task`, `plot_solutions_task`,
  and `combine_h5parms_task`
- a much thinner `finalize_solutions_task`
- unchanged calibration command counts and command totals within normal CI
  variance
- acceptable Dask duration-minus-compute gap despite the extra task count

If the broad scenarios remain comparable and the narrow scenario confirms the
new task groups are readable, proceed to the next task-split candidate:
prediction WSClean loops and sector-model post-processing.
