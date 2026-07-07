# 2026-07-07 CI Benchmark Comparison

Source artifacts:

- `runs/benchmark-20260704-122100`
- `runs/benchmark-20260706-203026`
- `runs/benchmark-20260707-153316`

Raw CI artifacts, Dask HTML reports, command logs, generated products, and run
directories are intentionally not committed. This report keeps the compact
comparison needed to decide whether the first image-sector scalability slice is
worth keeping and what to optimize next.

## Run Identity

| Run | Commit | CI job | Runner | Scenario | Repetitions | Return codes |
| --- | --- | --- | --- | --- | ---: | --- |
| `20260704-122100` | `b8ed119e` from the existing compact baseline | not present in downloaded report | larger GitLab runner, 2 workers / 60 Dask threads | `ci-benchmark` | 3 | `0, 0, 0` |
| `20260706-203026` | `df67648f` | `1831569` | Shared Runner on `lcs126`, 2 workers / 60 Dask threads | `ci-benchmark` | 3 | `0, 0, 0` |
| `20260707-153316` | `eb0a4033` | `1832025` | Shared Runner on `lcs126`, 2 workers / 60 Dask threads | `ci-benchmark` | 3 | `0, 0, 0` |

## Summary

| Run | Wall Median (s) | Wall Min-Max (s) | Command Median (s) | Dask Duration Median (s) | Dask Compute Median (s) | Dask Gap Median (s) | Dask Tasks |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `20260704-122100` | `482.894` | `481.534-488.178` | `230.551` | `469.550` | `247.350` | `220.940` | `12` |
| `20260706-203026` | `305.320` | `304.572-308.257` | `229.843` | `291.400` | `220.580` | `70.480` | `12` |
| `20260707-153316` | `308.563` | `308.300-314.398` | `230.799` | `295.040` | `222.470` | `72.570` | `16` |

Derived deltas:

| Comparison | Wall Change | Dask Gap Change | Image Operation Gap Change |
| --- | ---: | ---: | ---: |
| `20260704` to `20260706` | `-36.8%` | `-68.1%` | `-68.2%` |
| `20260706` to `20260707` | `+1.1%` | `+3.0%` | `+22.3%` |

The command profile sum is stable at about `230 s`, so the July 4 to July 6
wall-time improvement is not primarily explained by faster external commands.
It is mostly a reduction in Dask/report idle gap and Rapthor operation overhead.

## Dask Task Shape

| Run | Task Group | Median Count | Median Total Compute (s) | Median Max Task (s) |
| --- | --- | ---: | ---: | ---: |
| `20260704-122100` | `calibrate_chunk_task` | `4` | `6.892` | `3.475` |
| `20260704-122100` | `image_sector_task` | `4` | `239.083` | `69.374` |
| `20260704-122100` | `predict_model_data_task` | `2` | `0.920` | `0.467` |
| `20260704-122100` | `predict_postprocess_task` | `2` | `0.452` | `0.273` |
| `20260706-203026` | `calibrate_chunk_task` | `4` | `6.887` | `3.473` |
| `20260706-203026` | `image_sector_task` | `4` | `212.316` | `61.625` |
| `20260706-203026` | `predict_model_data_task` | `2` | `0.928` | `0.476` |
| `20260706-203026` | `predict_postprocess_task` | `2` | `0.456` | `0.271` |
| `20260707-153316` | `calibrate_chunk_task` | `4` | `6.958` | `3.484` |
| `20260707-153316` | `image_sector_prepare_task` | `4` | `92.258` | `24.534` |
| `20260707-153316` | `image_sector_finalize_task` | `4` | `122.276` | `39.337` |
| `20260707-153316` | `predict_model_data_task` | `2` | `0.921` | `0.469` |
| `20260707-153316` | `predict_postprocess_task` | `2` | `0.458` | `0.272` |

The July 7 run is the first one where the image-sector split is visible in the
Dask task groups. It increases the Dask task count from `12` to `16` by
splitting the four image-sector tasks into four prepare tasks and four finalize
tasks.

## Operation Gap

Median operation-minus-command gap, summed over image operations:

| Run | Image Operation Gap (s) |
| --- | ---: |
| `20260704-122100` | `39.136` |
| `20260706-203026` | `12.442` |
| `20260707-153316` | `15.220` |

The July 7 split improves dashboard/task visibility but is not the source of
the large July 4 to July 6 improvement. Relative to July 6, it is close to
neutral on wall time and Dask gap, with a small increase in image operation
gap.

## Command Timing

Median command totals by name for the two most recent benchmark runs:

| Run | Command | Median Total (s) | Median Count |
| --- | --- | ---: | ---: |
| `20260706-203026` | `filter_skymodel` | `110.228` | `4` |
| `20260706-203026` | `wsclean` | `87.567` | `4` |
| `20260706-203026` | `python3` | `17.837` | `10` |
| `20260706-203026` | `DP3` | `8.217` | `10` |
| `20260707-153316` | `filter_skymodel` | `110.312` | `4` |
| `20260707-153316` | `wsclean` | `88.961` | `4` |
| `20260707-153316` | `python3` | `17.955` | `10` |
| `20260707-153316` | `DP3` | `8.279` | `10` |

The next scalability work should focus on `filter_skymodel` resource use and
placement. It is currently isolated as a subprocess when needed because PyBDSF
and daemonic Dask workers do not make safe direct in-worker execution a simple
default. The right next question is therefore not "where can we add another
tidy Prefect box?", but "what worker/thread shape and task placement gives
`filter_skymodel` enough resources without over-subscribing the machine?"

The benchmark runner now supports named resource profiles for this comparison.
Use the same scientific `ci-benchmark` scenario and vary only the resource
shape:

```bash
scripts/dev/run_benchmark_baseline.py \
  --scenario ci-benchmark \
  --resource-profile baseline-2x30 \
  --resource-profile filter-threads-15 \
  --resource-profile filter-workers-4x15 \
  --resource-profile filter-wide-1x60 \
  --repetitions 3 \
  --prepare-inputs
```

The GitLab benchmark job is configured to run these profiles by default.
Profiles override the job's `--local-dask-workers`, `--cpus-per-task`, and
`--max-threads` arguments so the matrix entries remain distinct.

## Interpretation

The first scalability slice should be kept for now as an observability
improvement, but it should not be treated as a proven performance improvement.
The evidence says:

- The major performance win happened between the July 4 and July 6 runs, before
  the split appeared as separate Dask task groups.
- The July 7 split exposes a better task graph (`image_sector_prepare_task` and
  `image_sector_finalize_task`) with only about a `1%` wall-time regression
  relative to July 6.
- The command profile is stable, so the remaining opportunity is orchestration
  shape, dependency gaps, and work that is still hidden inside image-sector
  commands/tasks.
- The next performance investigation should target the dominant
  `filter_skymodel` command group before adding another task boundary.

## July 4 to July 6 Improvement

The most likely explanation is that FITS preview artifact rendering moved from
always-on behavior to explicit opt-in behavior between `b8ed119e` and
`df67648f`.

Evidence:

- The July 4 image-sector and mosaic code called
  `publish_fits_image_artifacts(...)` unconditionally after producing FITS
  products.
- By July 6, image-sector and mosaic preview publication were guarded by
  `config.publish_fits_previews`, with the default
  `prefect_publish_fits_previews = False`.
- The July 6 benchmark parsets explicitly contain
  `prefect_publish_fits_previews = False` and
  `prefect_publish_postage_stamp_previews = False`.
- The command-profile median is essentially unchanged at about `230 s`, while
  wall time, Dask gap, and image operation-minus-command gap all drop sharply.
  That is the expected shape for removing Python-side preview rendering,
  artifact-file writing, and Prefect artifact publication from the benchmark
  path, rather than speeding up DP3/WSClean commands themselves.
- The command logs for the first commands in the July 4 and July 6 runs have
  the same command shape apart from timestamped run-root paths, supporting the
  conclusion that the main improvement is outside the external-command profile.

This explanation has high confidence for the orchestration-gap reduction. Other
changes in the same range, such as parallel-gridding option updates and pinned
external-tool versions, may contribute to the smaller image-task compute
difference, but they do not explain the much larger reduction in wall time and
operation overhead by themselves.

## Recommended Next Steps

1. Keep the current image-sector split unless dashboard noise or focused tests
   show a clear downside.
2. Keep the existing generated-parset guard that disables preview artifacts for
   CI benchmarks, while allowing demo/debug parsets to enable previews for
   dashboard inspection.
3. Benchmark `filter_skymodel` resource settings and task placement before
   adding another image-sector task boundary. Candidate comparisons should keep
   the same `ci-benchmark` scenario and vary only the worker/thread shape or
   filter-skymodel resource allocation.
4. Compare one more benchmark after any scalability change using the same
   `ci-benchmark` resource shape.
5. Choose any further task boundary from measured command and Dask report
   evidence, not from tidiness alone.
