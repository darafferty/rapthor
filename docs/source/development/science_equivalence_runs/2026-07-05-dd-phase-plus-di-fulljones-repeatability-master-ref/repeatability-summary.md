# Rapthor Branch Repeatability

Scenario: `dd-phase-plus-di-fulljones-repeatability`
Run root: `/tmp/rfjr`
Repetitions per side: 3

## Branch Runs

| Side | Repetition | Ref | Return Code | Parset | Work Dir | Log |
| --- | --- | --- | ---: | --- | --- | --- |
| base | `rep-01` | `master` | 0 | `/tmp/rfjr/inputs/base/rep-01/master_dd_phase_plus_di_fulljones.parset` | `/tmp/rfjw/base/rep-01` | `/tmp/rfjr/base/rep-01/rapthor-command.log` |
| base | `rep-02` | `master` | 0 | `/tmp/rfjr/inputs/base/rep-02/master_dd_phase_plus_di_fulljones.parset` | `/tmp/rfjw/base/rep-02` | `/tmp/rfjr/base/rep-02/rapthor-command.log` |
| base | `rep-03` | `master` | 0 | `/tmp/rfjr/inputs/base/rep-03/master_dd_phase_plus_di_fulljones.parset` | `/tmp/rfjw/base/rep-03` | `/tmp/rfjr/base/rep-03/rapthor-command.log` |
| current | `rep-01` | `current` | 0 | `/tmp/rfjr/inputs/current/rep-01/current_dd_phase_plus_di_fulljones.parset` | `/tmp/rfjw/current/rep-01` | `/tmp/rfjr/current/rep-01/rapthor-command.log` |
| current | `rep-02` | `current` | 0 | `/tmp/rfjr/inputs/current/rep-02/current_dd_phase_plus_di_fulljones.parset` | `/tmp/rfjw/current/rep-02` | `/tmp/rfjr/current/rep-02/rapthor-command.log` |
| current | `rep-03` | `current` | 0 | `/tmp/rfjr/inputs/current/rep-03/current_dd_phase_plus_di_fulljones.parset` | `/tmp/rfjw/current/rep-03` | `/tmp/rfjr/current/rep-03/rapthor-command.log` |

## Pair Summary

| Pair | Group | Result | Failures | Warnings | FITS | H5 | Text | Diagnostics | Max Abs Delta | P99 Abs Delta | Residual RMS | Diagnostic Rel Delta | Report |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `base-rep-01_vs_base-rep-02` | `base-base` | pass | 0 | 0 | 7 | 3 | 10 | 1 | 1.431e-06 | 8.941e-08 | 3.340e-08 | 0.015% | `pairs/base-rep-01_vs_base-rep-02/branch-equivalence-report.json` |
| `base-rep-01_vs_base-rep-03` | `base-base` | pass | 0 | 0 | 7 | 3 | 10 | 1 | 9.537e-07 | 8.941e-08 | 3.254e-08 | 0.019% | `pairs/base-rep-01_vs_base-rep-03/branch-equivalence-report.json` |
| `base-rep-02_vs_base-rep-03` | `base-base` | pass | 0 | 0 | 7 | 3 | 10 | 1 | 1.192e-06 | 8.196e-08 | 3.042e-08 | 0.004% | `pairs/base-rep-02_vs_base-rep-03/branch-equivalence-report.json` |
| `current-rep-01_vs_current-rep-02` | `current-current` | fail | 29 | 0 | 7 | 4 | 10 | 1 | 2.745e-03 | 1.061e-05 | 4.204e-06 | 0.010% | `pairs/current-rep-01_vs_current-rep-02/branch-equivalence-report.json` |
| `current-rep-01_vs_current-rep-03` | `current-current` | fail | 29 | 0 | 7 | 4 | 10 | 1 | 2.745e-03 | 1.061e-05 | 4.203e-06 | 0.014% | `pairs/current-rep-01_vs_current-rep-03/branch-equivalence-report.json` |
| `current-rep-02_vs_current-rep-03` | `current-current` | pass | 0 | 0 | 7 | 4 | 10 | 1 | 9.537e-07 | 8.941e-08 | 3.008e-08 | 0.009% | `pairs/current-rep-02_vs_current-rep-03/branch-equivalence-report.json` |
| `base-rep-01_vs_current-rep-01` | `base-current` | fail | 48 | 2 | 7 | 3 | 10 | 1 | 1.025e-02 | 9.260e-04 | 3.872e-04 | 0.224% | `pairs/base-rep-01_vs_current-rep-01/branch-equivalence-report.json` |
| `base-rep-01_vs_current-rep-02` | `base-current` | fail | 61 | 2 | 7 | 3 | 10 | 1 | 1.024e-02 | 9.260e-04 | 3.872e-04 | 0.224% | `pairs/base-rep-01_vs_current-rep-02/branch-equivalence-report.json` |
| `base-rep-01_vs_current-rep-03` | `base-current` | fail | 61 | 2 | 7 | 3 | 10 | 1 | 1.024e-02 | 9.260e-04 | 3.872e-04 | 0.224% | `pairs/base-rep-01_vs_current-rep-03/branch-equivalence-report.json` |
| `base-rep-02_vs_current-rep-01` | `base-current` | fail | 47 | 2 | 7 | 3 | 10 | 1 | 1.025e-02 | 9.259e-04 | 3.871e-04 | 0.229% | `pairs/base-rep-02_vs_current-rep-01/branch-equivalence-report.json` |
| `base-rep-02_vs_current-rep-02` | `base-current` | fail | 61 | 2 | 7 | 3 | 10 | 1 | 1.024e-02 | 9.259e-04 | 3.872e-04 | 0.235% | `pairs/base-rep-02_vs_current-rep-02/branch-equivalence-report.json` |
| `base-rep-02_vs_current-rep-03` | `base-current` | fail | 61 | 2 | 7 | 3 | 10 | 1 | 1.024e-02 | 9.259e-04 | 3.871e-04 | 0.226% | `pairs/base-rep-02_vs_current-rep-03/branch-equivalence-report.json` |
| `base-rep-03_vs_current-rep-01` | `base-current` | fail | 48 | 2 | 7 | 3 | 10 | 1 | 1.025e-02 | 9.259e-04 | 3.871e-04 | 0.233% | `pairs/base-rep-03_vs_current-rep-01/branch-equivalence-report.json` |
| `base-rep-03_vs_current-rep-02` | `base-current` | fail | 61 | 2 | 7 | 3 | 10 | 1 | 1.024e-02 | 9.260e-04 | 3.872e-04 | 0.239% | `pairs/base-rep-03_vs_current-rep-02/branch-equivalence-report.json` |
| `base-rep-03_vs_current-rep-03` | `base-current` | fail | 61 | 2 | 7 | 3 | 10 | 1 | 1.024e-02 | 9.259e-04 | 3.871e-04 | 0.230% | `pairs/base-rep-03_vs_current-rep-03/branch-equivalence-report.json` |

## Interpretation

Use same-branch pairs to estimate run-to-run scatter before accepting or tightening product-specific tolerances. Cross-branch differences that are consistently larger than same-branch scatter need a scientific explanation, a bug fix, or an explicit intentional-difference label.
