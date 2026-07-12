# Rapthor Branch Repeatability

Scenario: `dd-phase-plus-di-fulljones-performance-gate`
Run root: `/tmp/rddfj`
Repetitions per side: 3

## Branch Runs

| Side | Repetition | Ref | Return Code | Elapsed (s) | Parset | Work Dir | Log |
| --- | --- | --- | ---: | ---: | --- | --- | --- |
| base | `rep-01` | `master` | 0 | 175.248 | `/tmp/rddfj/inputs/base/rep-01/master_dd_phase_plus_di_fulljones.parset` | `/tmp/wddfj/base/rep-01` | `/tmp/rddfj/base/rep-01/rapthor-command.log` |
| base | `rep-02` | `master` | 0 | 151.183 | `/tmp/rddfj/inputs/base/rep-02/master_dd_phase_plus_di_fulljones.parset` | `/tmp/wddfj/base/rep-02` | `/tmp/rddfj/base/rep-02/rapthor-command.log` |
| base | `rep-03` | `master` | 0 | 148.245 | `/tmp/rddfj/inputs/base/rep-03/master_dd_phase_plus_di_fulljones.parset` | `/tmp/wddfj/base/rep-03` | `/tmp/rddfj/base/rep-03/rapthor-command.log` |
| current | `rep-01` | `current` | 0 | 95.067 | `/tmp/rddfj/inputs/current/rep-01/current_dd_phase_plus_di_fulljones.parset` | `/tmp/wddfj/current/rep-01` | `/tmp/rddfj/current/rep-01/rapthor-command.log` |
| current | `rep-02` | `current` | 0 | 94.004 | `/tmp/rddfj/inputs/current/rep-02/current_dd_phase_plus_di_fulljones.parset` | `/tmp/wddfj/current/rep-02` | `/tmp/rddfj/current/rep-02/rapthor-command.log` |
| current | `rep-03` | `current` | 0 | 86.962 | `/tmp/rddfj/inputs/current/rep-03/current_dd_phase_plus_di_fulljones.parset` | `/tmp/wddfj/current/rep-03` | `/tmp/rddfj/current/rep-03/rapthor-command.log` |

## Gate Decision

Overall status: **pass**

| Decision Area | Status | Notes |
| --- | --- | --- |
| Run validity | pass | 0 failed run(s) |
| Science/product validity | pass | 4 of 9 cross-branch pair(s) repeatability-bounded |
| Performance | pass | Current branch median runtime is not slower than master. |

## Runtime Summary

| Side | Runs | Min (s) | Median (s) | Max (s) |
| --- | ---: | ---: | ---: | ---: |
| base | 3 | 148.245 | 151.183 | 175.248 |
| current | 3 | 86.962 | 94.004 | 95.067 |

Current-vs-base median delta: -37.821%

## Operation Runtime Summary

| Operation | Base Runs | Base Median (s) | Current Runs | Current Median (s) | Delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| `calibrate_1` | 3 | 22.568 | 3 | 4.132 | -81.691% |
| `calibrate_di_1` | 3 | 19.311 | 3 | 6.128 | -68.269% |
| `image_1` | 3 | 87.845 | 3 | 58.258 | -33.681% |
| `mosaic_1` | 3 | 3.436 | 3 | 2.997 | -12.770% |
| `predict_di_1` | 3 | 15.691 | 3 | 4.095 | -73.904% |

## Pair Summary

| Pair | Group | Result | Failures | Warnings | FITS | H5 | Text | Diagnostics | Max Abs Delta | P99 Abs Delta | Residual RMS | Diagnostic Rel Delta | Report |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `base-rep-01_vs_base-rep-02` | `base-base` | repeatability-reference | 28 | 0 | 7 | 3 | 10 | 1 | 2.739e-03 | 1.058e-05 | 4.193e-06 | 0.011% | `pairs/base-rep-01_vs_base-rep-02/branch-equivalence-report.json` |
| `base-rep-01_vs_base-rep-03` | `base-base` | repeatability-reference | 28 | 0 | 7 | 3 | 10 | 1 | 2.739e-03 | 1.059e-05 | 4.193e-06 | 0.008% | `pairs/base-rep-01_vs_base-rep-03/branch-equivalence-report.json` |
| `base-rep-02_vs_base-rep-03` | `base-base` | repeatability-reference | 0 | 0 | 7 | 3 | 10 | 1 | 9.537e-07 | 7.451e-08 | 2.798e-08 | 0.010% | `pairs/base-rep-02_vs_base-rep-03/branch-equivalence-report.json` |
| `current-rep-01_vs_current-rep-02` | `current-current` | repeatability-reference | 28 | 0 | 7 | 4 | 10 | 1 | 2.739e-03 | 1.059e-05 | 4.194e-06 | 0.027% | `pairs/current-rep-01_vs_current-rep-02/branch-equivalence-report.json` |
| `current-rep-01_vs_current-rep-03` | `current-current` | repeatability-reference | 28 | 0 | 7 | 4 | 10 | 1 | 2.739e-03 | 1.058e-05 | 4.194e-06 | 0.020% | `pairs/current-rep-01_vs_current-rep-03/branch-equivalence-report.json` |
| `current-rep-02_vs_current-rep-03` | `current-current` | repeatability-reference | 0 | 0 | 7 | 4 | 10 | 1 | 1.431e-06 | 8.196e-08 | 2.934e-08 | 0.008% | `pairs/current-rep-02_vs_current-rep-03/branch-equivalence-report.json` |
| `base-rep-01_vs_current-rep-01` | `base-current` | pass | 0 | 2 | 7 | 3 | 10 | 1 | 1.431e-06 | 8.009e-08 | 2.908e-08 | 0.024% | `pairs/base-rep-01_vs_current-rep-01/branch-equivalence-report.json` |
| `base-rep-01_vs_current-rep-02` | `base-current` | repeatability-bounded | 28 | 2 | 7 | 3 | 10 | 1 | 2.739e-03 | 1.059e-05 | 4.194e-06 | 0.004% | `pairs/base-rep-01_vs_current-rep-02/branch-equivalence-report.json` |
| `base-rep-01_vs_current-rep-03` | `base-current` | repeatability-bounded | 28 | 2 | 7 | 3 | 10 | 1 | 2.739e-03 | 1.059e-05 | 4.194e-06 | 0.006% | `pairs/base-rep-01_vs_current-rep-03/branch-equivalence-report.json` |
| `base-rep-02_vs_current-rep-01` | `base-current` | repeatability-bounded | 28 | 2 | 7 | 3 | 10 | 1 | 2.739e-03 | 1.058e-05 | 4.193e-06 | 0.017% | `pairs/base-rep-02_vs_current-rep-01/branch-equivalence-report.json` |
| `base-rep-02_vs_current-rep-02` | `base-current` | pass | 0 | 2 | 7 | 3 | 10 | 1 | 1.431e-06 | 8.941e-08 | 2.993e-08 | 0.011% | `pairs/base-rep-02_vs_current-rep-02/branch-equivalence-report.json` |
| `base-rep-02_vs_current-rep-03` | `base-current` | pass | 0 | 2 | 7 | 3 | 10 | 1 | 9.537e-07 | 7.567e-08 | 2.842e-08 | 0.017% | `pairs/base-rep-02_vs_current-rep-03/branch-equivalence-report.json` |
| `base-rep-03_vs_current-rep-01` | `base-current` | repeatability-bounded | 28 | 2 | 7 | 3 | 10 | 1 | 2.739e-03 | 1.058e-05 | 4.193e-06 | 0.027% | `pairs/base-rep-03_vs_current-rep-01/branch-equivalence-report.json` |
| `base-rep-03_vs_current-rep-02` | `base-current` | pass | 0 | 2 | 7 | 3 | 10 | 1 | 1.431e-06 | 8.941e-08 | 3.093e-08 | 0.008% | `pairs/base-rep-03_vs_current-rep-02/branch-equivalence-report.json` |
| `base-rep-03_vs_current-rep-03` | `base-current` | pass | 0 | 2 | 7 | 3 | 10 | 1 | 1.192e-06 | 8.196e-08 | 2.917e-08 | 0.014% | `pairs/base-rep-03_vs_current-rep-03/branch-equivalence-report.json` |

## Interpretation

Use same-branch pairs to estimate run-to-run scatter before accepting or tightening product-specific tolerances. Cross-branch differences that are consistently larger than same-branch scatter need a scientific explanation, a bug fix, or an explicit intentional-difference label.
