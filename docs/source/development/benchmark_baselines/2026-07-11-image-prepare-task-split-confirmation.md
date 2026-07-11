# 2026-07-11 Image Prepare Task-Split Confirmation

Source artifacts:

- `runs/benchmark-20260711-105033`

Raw CI artifacts, Dask HTML reports, command logs, generated products, and run
directories are intentionally not committed. This report preserves the compact
evidence needed to decide whether the image-sector preparation split is safe to
keep.

## Run Identity

| Field | Value |
| --- | --- |
| Commit | `81716b2d7d9f78b88bbaac80442df70f8d1a6c0c` |
| CI job | `1837109` |
| Pipeline | `201449` |
| Runner | `Shared Runner on lcs126` |
| Ref | `gec-468-ai-migrate-to-prefect` |
| Repetitions | `1` per scenario |
| Return codes | `0` for both runs |

The run used the preferred automatic benchmark shape:

- `local_dask_workers=4`
- `cpus_per_task=15`
- `max_threads=15`
- `filter_skymodel_ncores=15`

## Same-Runner Comparison

The immediately preceding benchmark (`runs/benchmark-20260711-081953`) ran on a
different runner (`astron-docker-3`), so raw wall times should not be compared
directly. The closest same-runner reference is `runs/benchmark-20260711-061530`
on `Shared Runner on lcs126`.

| Scenario | Run | Wall (s) | Command Total (s) | Dask Duration (s) | Dask Compute (s) | Dask Gap (s) | Dask Tasks | Commands |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ci-benchmark` | before prepare split | `322.078` | `243.881` | `303.440` | `267.310` | `36.130` | `61` | `47` |
| `ci-benchmark` | after prepare split | `331.956` | `228.891` | `314.360` | `279.090` | `35.270` | `77` | `47` |

Interpretation:

- Wall time increased by about `3.1%`, which is small enough to treat as normal
  CI/runtime variance for a single repetition.
- External command count stayed fixed at `47`.
- External command time decreased by about `6.1%`; the split did not add shell
  work.
- Dask duration-minus-compute gap stayed effectively unchanged (`36.13 s` to
  `35.27 s`), so the additional task boundaries did not introduce an obvious
  scheduling penalty.
- Task count increased from `61` to `77`, as expected, because the previous
  opaque image `prepare` task is now visible as separate task groups.

## Image-Products Scenario

The image-products scenario was included to exercise the optional image-product
paths affected by the preparation split.

| Scenario | Wall (s) | Command Total (s) | Dask Duration (s) | Dask Compute (s) | Dask Gap (s) | Dask Tasks | Commands |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ci-benchmark-image-products` | `367.737` | `261.218` | `354.350` | `316.240` | `38.110` | `91` | `55` |

This scenario completed successfully and gives us a useful post-split baseline
for future image-product changes. It should remain automatic while accepting
this split, then become targeted again unless the next change touches image
products or image-sector preparation.

## Task-Split Evidence

The split exposes the following formerly hidden task groups:

| Task group | Count | Total (s) | Max (s) |
| --- | ---: | ---: | ---: |
| `image_sector_prepare_visibility_task` | `4` | `7.893` | `2.759` |
| `image_sector_concatenate_task` | `4` | `0.782` | `0.198` |
| `image_sector_wsclean_task` | `4` | `96.156` | `25.505` |
| `image_sector_finish_wsclean_task` | `4` | `0.729` | `0.184` |
| `image_sector_prepare_outputs_task` | `4` | `0.688` | `0.182` |

The expensive part of the former `prepare` task is clearly WSClean imaging.
Per-observation DP3 preparation is visible and inexpensive in this CI-sized
scenario; concatenation and join tasks are very small.

## Dominant Costs After The Split

| Scenario | Dominant cost | Time (s) | Count |
| --- | --- | ---: | ---: |
| `ci-benchmark` | `filter_skymodel` | `105.665` | `4` |
| `ci-benchmark` | `wsclean` | `94.096` | `4` |
| `ci-benchmark` | `python3` | `17.987` | `10` |
| `ci-benchmark-image-products` | `filter_skymodel` | `113.837` | `5` |
| `ci-benchmark-image-products` | `wsclean` | `108.680` | `5` |
| `ci-benchmark-image-products` | `python3` | `18.245` | `10` |

The next optimisation targets are therefore:

- `filter_skymodel`: still the largest single command cost and a good
  candidate for resource/concurrency tuning or algorithmic profiling.
- WSClean image runs: now visible as their own task group, suitable for
  resource-shape comparisons and concurrency checks.
- Prediction post-processing: review dependency shape before optimizing; it is
  not dominant in this default scenario, but it can hide waiting if grouped too
  coarsely.

Do not spend effort further splitting the image `prepare` join, concatenation,
or finish tasks unless larger real data shows they become bottlenecks.

## Decision

Accepted. Keep the image-sector preparation split and proceed to reviewing
standalone prediction parallelism. Continue to benchmark `ci-benchmark` by
default. Keep `ci-benchmark-image-products` available as a targeted scenario
after this acceptance unless subsequent changes touch image-product or
image-sector preparation behavior.
