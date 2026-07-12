# Rapthor Branch Equivalence

Scenario: `slow-gain-default-like`
Run root: `/app/runs/rbe-slow-gain-default-like-20260705`

## Branch Runs

| Side | Ref | Return Code | Parset | Work Dir | Log | Input Snapshot |
| --- | --- | ---: | --- | --- | --- | --- |
| base | `master` | 0 | `/app/docs/source/development/science_equivalence_runs/2026-07-05-slow-gain-default-like-master-ref/inputs/base/master_slow_gain_default_like.parset` | `/tmp/rbe-m-sg-v1-w` | `/app/runs/rbe-slow-gain-default-like-20260705/base/rapthor-command.log` | parset: `inputs/base/master_slow_gain_default_like.parset`, strategy: `inputs/base/master_slow_gain_default_like_strategy.py` |
| current | `current` | 0 | `/app/docs/source/development/science_equivalence_runs/2026-07-05-slow-gain-default-like-master-ref/inputs/current/current_slow_gain_default_like.parset` | `/tmp/rbe-c-sg-v1-w` | `/app/runs/rbe-slow-gain-default-like-20260705/current/rapthor-command.log` | parset: `inputs/current/current_slow_gain_default_like.parset`, strategy: `inputs/current/current_slow_gain_default_like_strategy.py` |

## Comparison Summary

| Result | Ops | Records | FITS | Image HDUs | Table HDUs | H5 | Text | Diagnostics | Visuals |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fail | 1 | 1 | 0 | 0 | 0 | 5 | 7 | 0 | 2 |

## FITS Residual Metrics

| Product | Max Abs Delta | P99 Abs Delta | Residual RMS | RMS / Ref RMS | RMS / Ref MAD |
| --- | ---: | ---: | ---: | ---: | ---: |
| n/a | n/a | n/a | n/a | n/a | n/a |

## Visual Comparisons

### Solution: `calibrate_1/fast_phase_dir[Patch_0].png`

![calibrate_1/fast_phase_dir[Patch_0].png](visual-comparisons/solutions/calibrate_1-fast_phase_dir-patch_0-.png.png)

### Solution: `calibrate_1/medium1_phase_dir[Patch_0].png`

![calibrate_1/medium1_phase_dir[Patch_0].png](visual-comparisons/solutions/calibrate_1-medium1_phase_dir-patch_0-.png.png)


## Warnings

- output-record summary differs for calibrate_1

## Failures

- HDF5 dataset names differ for field-solutions.h5
