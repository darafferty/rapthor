# 2026-07-11 WSClean-Predict Task-Split Confirmation

Source artifacts:

- `runs/benchmark-20260711-081953`

Raw CI artifacts, Dask HTML reports, command logs, generated products, and run
directories are intentionally not committed. This report preserves the compact
evidence needed to decide whether the calibration image-based/WSClean
prediction setup split is safe to keep.

## Run Identity

| Field | Value |
| --- | --- |
| Commit | `498231dd7df90b48b980c9eb34a81e4799e126c2` |
| CI job | `1837072` |
| Pipeline | `201445` |
| Runner | `astron-docker-3` |
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
| `ci-benchmark` | `182.960` | `140.900` | `172.670` | `152.460` | `20.210` | `61` | `47` |
| `ci-benchmark-wsclean-predict` | `171.067` | `131.277` | `162.880` | `144.490` | `18.390` | `67` | `59` |

The runner changed from the previous 2026-07-11 benchmark (`lcs126`) to
`astron-docker-3`, so the wall-time improvement should not be attributed solely
to code changes. The stronger acceptance signal is structural:

- both scenarios exited successfully
- command counts stayed stable compared with the before-split baseline
- `ci-benchmark-wsclean-predict` exposed the expected prediction setup task
  groups without increasing command count
- the Dask gap stayed modest at about 18-20 s

## Task-Split Evidence

The WSClean-predict scenario now exposes these task groups:

| Task group | Count |
| --- | ---: |
| `make_predict_region_task` | `2` |
| `wsclean_predict_facet_info_task` | `2` |
| `wsclean_predict_chunk_task` | `2` |

The broad default scenario stayed at `61` Dask tasks and `47` external
commands. The WSClean-predict scenario increased from `61` to `67` Dask tasks
because the previously hidden prediction setup work is now visible, while the
external command count stayed at `59`.

## Dominant Costs

| Scenario | Dominant cost | Time (s) | Count |
| --- | --- | ---: | ---: |
| `ci-benchmark` | `filter_skymodel` | `68.795` | `4` |
| `ci-benchmark` | `wsclean` | `49.851` | `4` |
| `ci-benchmark` | `python3` | `17.497` | `10` |
| `ci-benchmark-wsclean-predict` | `filter_skymodel` | `56.537` | `4` |
| `ci-benchmark-wsclean-predict` | `wsclean` | `52.526` | `16` |
| `ci-benchmark-wsclean-predict` | `python3` | `17.559` | `10` |

The next performance targets should therefore be image-preparation and imaging
observability first, followed by targeted optimisation of `filter_skymodel`,
WSClean resource/concurrency choices, and calibration plotting/Python
post-processing costs.

## Decision

Accepted. Keep the calibration image-based/WSClean prediction setup split and
proceed to the image-sector `prepare` split. Continue benchmarking with
`ci-benchmark`; add `ci-benchmark-image-products` for the image-preparation
split batch. Keep `ci-benchmark-wsclean-predict` only when WSClean-predict
inputs or calibration prediction setup paths are touched.
