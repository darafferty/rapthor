# Rapthor Branch Repeatability

Scenario: `phase-only-core-performance-baseline`
Run root: `/tmp/r2`
Repetitions per side: 3

## Branch Runs

| Side | Repetition | Ref | Return Code | Elapsed (s) | Parset | Work Dir | Log |
| --- | --- | --- | ---: | ---: | --- | --- | --- |
| base | `rep-01` | `master` | 0 | 426.877 | `/tmp/r2/inputs/base/rep-01/master_benchmark_phase_only.parset` | `/tmp/w2/base/rep-01` | `/tmp/r2/base/rep-01/rapthor-command.log` |
| base | `rep-02` | `master` | 0 | 429.557 | `/tmp/r2/inputs/base/rep-02/master_benchmark_phase_only.parset` | `/tmp/w2/base/rep-02` | `/tmp/r2/base/rep-02/rapthor-command.log` |
| base | `rep-03` | `master` | 0 | 429.835 | `/tmp/r2/inputs/base/rep-03/master_benchmark_phase_only.parset` | `/tmp/w2/base/rep-03` | `/tmp/r2/base/rep-03/rapthor-command.log` |
| current | `rep-01` | `current` | 0 | 291.588 | `/tmp/r2/inputs/current/rep-01/current_benchmark_phase_only.parset` | `/tmp/w2/current/rep-01` | `/tmp/r2/current/rep-01/rapthor-command.log` |
| current | `rep-02` | `current` | 0 | 303.160 | `/tmp/r2/inputs/current/rep-02/current_benchmark_phase_only.parset` | `/tmp/w2/current/rep-02` | `/tmp/r2/current/rep-02/rapthor-command.log` |
| current | `rep-03` | `current` | 0 | 308.499 | `/tmp/r2/inputs/current/rep-03/current_benchmark_phase_only.parset` | `/tmp/w2/current/rep-03` | `/tmp/r2/current/rep-03/rapthor-command.log` |

## Gate Decision

Overall status: **pass**

| Decision Area | Status | Notes |
| --- | --- | --- |
| Run validity | pass | 0 failed run(s) |
| Science/product validity | pass | 9 of 9 cross-branch pair(s) repeatability-bounded |
| Performance | pass | Current branch median runtime is not slower than master. |

## Runtime Summary

| Side | Runs | Min (s) | Median (s) | Max (s) |
| --- | ---: | ---: | ---: | ---: |
| base | 3 | 426.877 | 429.557 | 429.835 |
| current | 3 | 291.588 | 303.160 | 308.499 |

Current-vs-base median delta: -29.425%

## Operation Runtime Summary

| Operation | Base Runs | Base Median (s) | Current Runs | Current Median (s) | Delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| `calibrate_1` | 3 | 21.394 | 3 | 8.131 | -61.995% |
| `calibrate_2` | 3 | 21.186 | 3 | 8.134 | -61.608% |
| `calibrate_3` | 3 | 21.261 | 3 | 6.106 | -71.282% |
| `calibrate_4` | 3 | 23.169 | 3 | 10.130 | -56.279% |
| `image_1` | 3 | 78.853 | 3 | 54.441 | -30.959% |
| `image_2` | 3 | 79.803 | 3 | 61.330 | -23.149% |
| `image_3` | 3 | 80.841 | 3 | 60.219 | -25.509% |
| `image_4` | 3 | 85.053 | 3 | 61.235 | -28.004% |
| `mosaic_1` | 3 | 3.291 | 3 | 1.162 | -64.677% |
| `mosaic_2` | 3 | 3.275 | 3 | 1.116 | -65.922% |
| `mosaic_3` | 3 | 3.280 | 3 | 1.119 | -65.893% |
| `mosaic_4` | 3 | 3.321 | 3 | 1.107 | -66.661% |

## Pair Summary

| Pair | Group | Result | Failures | Warnings | FITS | H5 | Text | Diagnostics | Max Abs Delta | P99 Abs Delta | Residual RMS | Diagnostic Rel Delta | Report |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `base-rep-01_vs_base-rep-02` | `base-base` | repeatability-reference | 122 | 0 | 28 | 8 | 37 | 4 | 4.395e-01 | 7.896e-03 | 4.666e-03 | 0.224% | `pairs/base-rep-01_vs_base-rep-02/branch-equivalence-report.json` |
| `base-rep-01_vs_base-rep-03` | `base-base` | repeatability-reference | 27 | 0 | 28 | 8 | 37 | 4 | 4.363e-01 | 9.924e-06 | 4.559e-03 | 0.011% | `pairs/base-rep-01_vs_base-rep-03/branch-equivalence-report.json` |
| `base-rep-02_vs_base-rep-03` | `base-base` | repeatability-reference | 123 | 0 | 28 | 8 | 37 | 4 | 4.335e-01 | 7.896e-03 | 4.593e-03 | 0.216% | `pairs/base-rep-02_vs_base-rep-03/branch-equivalence-report.json` |
| `current-rep-01_vs_current-rep-02` | `current-current` | repeatability-reference | 18 | 0 | 28 | 12 | 37 | 4 | 1.809e-02 | 9.939e-06 | 7.633e-06 | 0.010% | `pairs/current-rep-01_vs_current-rep-02/branch-equivalence-report.json` |
| `current-rep-01_vs_current-rep-03` | `current-current` | repeatability-reference | 120 | 0 | 28 | 12 | 37 | 4 | 8.599e-02 | 7.896e-03 | 2.037e-03 | 0.224% | `pairs/current-rep-01_vs_current-rep-03/branch-equivalence-report.json` |
| `current-rep-02_vs_current-rep-03` | `current-current` | repeatability-reference | 120 | 0 | 28 | 12 | 37 | 4 | 8.599e-02 | 7.896e-03 | 2.037e-03 | 0.214% | `pairs/current-rep-02_vs_current-rep-03/branch-equivalence-report.json` |
| `base-rep-01_vs_current-rep-01` | `base-current` | repeatability-bounded | 128 | 4 | 28 | 8 | 37 | 4 | 2.220e-01 | 7.896e-03 | 3.252e-03 | 0.233% | `pairs/base-rep-01_vs_current-rep-01/branch-equivalence-report.json` |
| `base-rep-01_vs_current-rep-02` | `base-current` | repeatability-bounded | 128 | 4 | 28 | 8 | 37 | 4 | 2.220e-01 | 7.896e-03 | 3.252e-03 | 0.223% | `pairs/base-rep-01_vs_current-rep-02/branch-equivalence-report.json` |
| `base-rep-01_vs_current-rep-03` | `base-current` | repeatability-bounded | 32 | 4 | 28 | 8 | 37 | 4 | 2.220e-01 | 9.935e-06 | 3.252e-03 | 0.011% | `pairs/base-rep-01_vs_current-rep-03/branch-equivalence-report.json` |
| `base-rep-02_vs_current-rep-01` | `base-current` | repeatability-bounded | 34 | 4 | 28 | 8 | 37 | 4 | 2.220e-01 | 9.939e-06 | 3.286e-03 | 0.009% | `pairs/base-rep-02_vs_current-rep-01/branch-equivalence-report.json` |
| `base-rep-02_vs_current-rep-02` | `base-current` | repeatability-bounded | 35 | 4 | 28 | 8 | 37 | 4 | 2.220e-01 | 9.937e-06 | 3.286e-03 | 0.002% | `pairs/base-rep-02_vs_current-rep-02/branch-equivalence-report.json` |
| `base-rep-02_vs_current-rep-03` | `base-current` | repeatability-bounded | 128 | 4 | 28 | 8 | 37 | 4 | 2.220e-01 | 7.896e-03 | 3.287e-03 | 0.215% | `pairs/base-rep-02_vs_current-rep-03/branch-equivalence-report.json` |
| `base-rep-03_vs_current-rep-01` | `base-current` | repeatability-bounded | 128 | 4 | 28 | 8 | 37 | 4 | 2.220e-01 | 7.896e-03 | 3.240e-03 | 0.224% | `pairs/base-rep-03_vs_current-rep-01/branch-equivalence-report.json` |
| `base-rep-03_vs_current-rep-02` | `base-current` | repeatability-bounded | 128 | 4 | 28 | 8 | 37 | 4 | 2.220e-01 | 7.896e-03 | 3.240e-03 | 0.215% | `pairs/base-rep-03_vs_current-rep-02/branch-equivalence-report.json` |
| `base-rep-03_vs_current-rep-03` | `base-current` | repeatability-bounded | 35 | 4 | 28 | 8 | 37 | 4 | 2.220e-01 | 9.939e-06 | 3.239e-03 | 0.001% | `pairs/base-rep-03_vs_current-rep-03/branch-equivalence-report.json` |

## Interpretation

Use same-branch pairs to estimate run-to-run scatter before accepting or tightening product-specific tolerances. Cross-branch differences that are consistently larger than same-branch scatter need a scientific explanation, a bug fix, or an explicit intentional-difference label.
