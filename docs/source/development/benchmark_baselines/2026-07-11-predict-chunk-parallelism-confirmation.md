# 2026-07-11 Predict-Chunk Parallelism Confirmation

Source artifacts:

- `runs/benchmark-20260711-162920`

Raw CI artifacts, Dask HTML reports, command logs, generated products, and run
directories are intentionally not committed. This report preserves the compact
evidence needed to decide whether the standalone prediction dependency split is
safe to keep.

## Run Identity

| Field | Value |
| --- | --- |
| Commit | `3f20401a315f3c0dc13fcc14e669529278143e8a` |
| CI job | `1837311` |
| Pipeline | `201460` |
| Runner | `Shared Runner on lcs126` |
| Ref | `gec-468-ai-migrate-to-prefect` |
| Repetitions | `1` per scenario |
| Return codes | `0` for all runs |

The run used the preferred automatic benchmark shape:

- `local_dask_workers=4`
- `cpus_per_task=15`
- `max_threads=15`
- `filter_skymodel_ncores=15`

## Headline Results

| Scenario | Wall (s) | Command Total (s) | Dask Duration (s) | Dask Compute (s) | Dask Gap (s) | Dask Tasks | Commands |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ci-benchmark` | `341.475` | `227.459` | `324.060` | `290.300` | `33.760` | `77` | `47` |
| `ci-benchmark-image-products` | `367.424` | `260.188` | `354.050` | `319.000` | `35.050` | `91` | `55` |
| `ci-benchmark-predict-chunks` | `330.485` | `229.447` | `316.910` | `287.870` | `29.040` | `89` | `57` |

The targeted `ci-benchmark-predict-chunks` scenario forces observation
chunking with `cluster.max_nodes = 2`, so it exercises multiple
`dp3_predict_chunk` / `postprocess` dependency groups.

Compared with `ci-benchmark` in the same CI run, the predict-chunks scenario:

- increased external command count from `47` to `57`
- increased Dask task count from `77` to `89`
- increased external command time by only about `0.9%`
- reduced wall time by about `3.2%`
- reduced the Dask duration-minus-compute gap by about `14%`

Do not treat the wall-time reduction as a proven speedup from a single
repetition, but it is strong evidence that keeping model-data futures live and
feeding each postprocess task only the matching futures does not harm
performance.

## Same-Runner Stability Check

Compared with the immediately preceding same-runner benchmark
(`runs/benchmark-20260711-105033` on `Shared Runner on lcs126`):

| Scenario | Metric | Previous | This run | Change |
| --- | --- | ---: | ---: | ---: |
| `ci-benchmark` | Wall (s) | `331.956` | `341.475` | `+2.9%` |
| `ci-benchmark` | Command total (s) | `228.891` | `227.459` | `-0.6%` |
| `ci-benchmark` | Dask tasks | `77` | `77` | `0.0%` |
| `ci-benchmark-image-products` | Wall (s) | `367.737` | `367.424` | `-0.1%` |
| `ci-benchmark-image-products` | Command total (s) | `261.218` | `260.188` | `-0.4%` |
| `ci-benchmark-image-products` | Dask tasks | `91` | `91` | `0.0%` |

The default benchmark wall time drift is not matched by external command time
or task count, so it is best treated as CI/runtime variance rather than a code
regression.

## Task Evidence

The targeted scenario exposed the expected additional task groups:

| Task group | Count | Total (s) | Median (s) | Max (s) |
| --- | ---: | ---: | ---: | ---: |
| `predict_model_data_task` | `4` | `3.899` | `0.484` | `2.502` |
| `predict_postprocess_task` | `4` | `3.196` | `0.248` | `2.472` |

The current report was generated before the dashboard/task-run rename from
`predict_model_data_*` to `dp3_predict_chunk_*`; the underlying task function
and dependency structure are the same.

## Dominant Costs

| Scenario | Dominant cost | Time (s) | Count |
| --- | --- | ---: | ---: |
| `ci-benchmark` | `filter_skymodel` | `105.155` | `4` |
| `ci-benchmark` | `wsclean` | `93.360` | `4` |
| `ci-benchmark` | `python3` | `17.828` | `10` |
| `ci-benchmark-image-products` | `filter_skymodel` | `113.841` | `5` |
| `ci-benchmark-image-products` | `wsclean` | `107.811` | `5` |
| `ci-benchmark-image-products` | `python3` | `18.125` | `10` |
| `ci-benchmark-predict-chunks` | `filter_skymodel` | `105.067` | `4` |
| `ci-benchmark-predict-chunks` | `wsclean` | `92.643` | `4` |
| `ci-benchmark-predict-chunks` | `python3` | `17.985` | `10` |

The next optimisation targets remain image-side:

- `filter_skymodel`, still the largest command cost
- WSClean image runs, the second-largest image-side command cost
- calibration plotting, if larger runs keep showing it as a visible
  post-processing cost

Do not spend effort further splitting image visibility preparation,
concatenation, finish, or finalize tasks unless larger real data shows they
become bottlenecks.

## Decision

Accepted. Keep the standalone prediction dependency split. Use
`ci-benchmark-predict-chunks` as a targeted scenario when prediction chunking,
predict post-processing dependencies, or related scheduling behavior changes.
It does not need to stay in every automatic benchmark once this batch is
accepted.
