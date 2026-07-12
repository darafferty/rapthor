# Rapthor Branch Repeatability

Scenario: `dd-phase-plus-di-fulljones-normalized-repeatability`
Run root: `/tmp/rfjnr`
Repetitions per side: 3

## Branch Runs

| Side | Repetition | Ref | Return Code | Parset | Work Dir | Log |
| --- | --- | --- | ---: | --- | --- | --- |
| base | `rep-01` | `master` | 0 | `/tmp/rfjnr/inputs/base/rep-01/master_dd_phase_plus_di_fulljones.parset` | `/tmp/rfjnw/base/rep-01` | `/tmp/rfjnr/base/rep-01/rapthor-command.log` |
| base | `rep-02` | `master` | 0 | `/tmp/rfjnr/inputs/base/rep-02/master_dd_phase_plus_di_fulljones.parset` | `/tmp/rfjnw/base/rep-02` | `/tmp/rfjnr/base/rep-02/rapthor-command.log` |
| base | `rep-03` | `master` | 0 | `/tmp/rfjnr/inputs/base/rep-03/master_dd_phase_plus_di_fulljones.parset` | `/tmp/rfjnw/base/rep-03` | `/tmp/rfjnr/base/rep-03/rapthor-command.log` |
| current | `rep-01` | `current` | 0 | `/tmp/rfjnr/inputs/current/rep-01/current_dd_phase_plus_di_fulljones.parset` | `/tmp/rfjnw/current/rep-01` | `/tmp/rfjnr/current/rep-01/rapthor-command.log` |
| current | `rep-02` | `current` | 0 | `/tmp/rfjnr/inputs/current/rep-02/current_dd_phase_plus_di_fulljones.parset` | `/tmp/rfjnw/current/rep-02` | `/tmp/rfjnr/current/rep-02/rapthor-command.log` |
| current | `rep-03` | `current` | 0 | `/tmp/rfjnr/inputs/current/rep-03/current_dd_phase_plus_di_fulljones.parset` | `/tmp/rfjnw/current/rep-03` | `/tmp/rfjnr/current/rep-03/rapthor-command.log` |

## Pair Summary

| Pair | Group | Result | Failures | Warnings | FITS | H5 | Text | Diagnostics | Max Abs Delta | P99 Abs Delta | Residual RMS | Diagnostic Rel Delta | Report |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `base-rep-01_vs_base-rep-02` | `base-base` | pass | 0 | 0 | 7 | 3 | 10 | 1 | 9.537e-07 | 7.823e-08 | 2.834e-08 | 0.017% | `pairs/base-rep-01_vs_base-rep-02/branch-equivalence-report.json` |
| `base-rep-01_vs_base-rep-03` | `base-base` | pass | 0 | 0 | 7 | 3 | 10 | 1 | 9.537e-07 | 8.941e-08 | 3.157e-08 | 0.008% | `pairs/base-rep-01_vs_base-rep-03/branch-equivalence-report.json` |
| `base-rep-02_vs_base-rep-03` | `base-base` | pass | 0 | 0 | 7 | 3 | 10 | 1 | 9.537e-07 | 8.941e-08 | 3.097e-08 | 0.021% | `pairs/base-rep-02_vs_base-rep-03/branch-equivalence-report.json` |
| `current-rep-01_vs_current-rep-02` | `current-current` | pass | 0 | 0 | 7 | 4 | 10 | 1 | 9.537e-07 | 8.941e-08 | 2.973e-08 | 0.008% | `pairs/current-rep-01_vs_current-rep-02/branch-equivalence-report.json` |
| `current-rep-01_vs_current-rep-03` | `current-current` | pass | 0 | 0 | 7 | 4 | 10 | 1 | 9.537e-07 | 7.451e-08 | 2.810e-08 | 0.010% | `pairs/current-rep-01_vs_current-rep-03/branch-equivalence-report.json` |
| `current-rep-02_vs_current-rep-03` | `current-current` | pass | 0 | 0 | 7 | 4 | 10 | 1 | 1.431e-06 | 8.335e-08 | 2.925e-08 | 0.010% | `pairs/current-rep-02_vs_current-rep-03/branch-equivalence-report.json` |
| `base-rep-01_vs_current-rep-01` | `base-current` | pass | 0 | 2 | 7 | 3 | 10 | 1 | 1.431e-06 | 8.941e-08 | 3.068e-08 | 0.012% | `pairs/base-rep-01_vs_current-rep-01/branch-equivalence-report.json` |
| `base-rep-01_vs_current-rep-02` | `base-current` | pass | 0 | 2 | 7 | 3 | 10 | 1 | 9.537e-07 | 7.451e-08 | 2.856e-08 | 0.008% | `pairs/base-rep-01_vs_current-rep-02/branch-equivalence-report.json` |
| `base-rep-01_vs_current-rep-03` | `base-current` | pass | 0 | 2 | 7 | 3 | 10 | 1 | 1.431e-06 | 8.941e-08 | 2.961e-08 | 0.018% | `pairs/base-rep-01_vs_current-rep-03/branch-equivalence-report.json` |
| `base-rep-02_vs_current-rep-01` | `base-current` | pass | 0 | 2 | 7 | 3 | 10 | 1 | 1.431e-06 | 8.941e-08 | 3.055e-08 | 0.010% | `pairs/base-rep-02_vs_current-rep-01/branch-equivalence-report.json` |
| `base-rep-02_vs_current-rep-02` | `base-current` | pass | 0 | 2 | 7 | 3 | 10 | 1 | 9.537e-07 | 7.451e-08 | 2.837e-08 | 0.017% | `pairs/base-rep-02_vs_current-rep-02/branch-equivalence-report.json` |
| `base-rep-02_vs_current-rep-03` | `base-current` | pass | 0 | 2 | 7 | 3 | 10 | 1 | 9.537e-07 | 8.196e-08 | 2.903e-08 | 0.019% | `pairs/base-rep-02_vs_current-rep-03/branch-equivalence-report.json` |
| `base-rep-03_vs_current-rep-01` | `base-current` | pass | 0 | 2 | 7 | 3 | 10 | 1 | 1.192e-06 | 7.963e-08 | 2.921e-08 | 0.012% | `pairs/base-rep-03_vs_current-rep-01/branch-equivalence-report.json` |
| `base-rep-03_vs_current-rep-02` | `base-current` | pass | 0 | 2 | 7 | 3 | 10 | 1 | 9.537e-07 | 8.941e-08 | 3.079e-08 | 0.004% | `pairs/base-rep-03_vs_current-rep-02/branch-equivalence-report.json` |
| `base-rep-03_vs_current-rep-03` | `base-current` | pass | 0 | 2 | 7 | 3 | 10 | 1 | 9.537e-07 | 8.568e-08 | 2.992e-08 | 0.010% | `pairs/base-rep-03_vs_current-rep-03/branch-equivalence-report.json` |

## Interpretation

Use same-branch pairs to estimate run-to-run scatter before accepting or tightening product-specific tolerances. Cross-branch differences that are consistently larger than same-branch scatter need a scientific explanation, a bug fix, or an explicit intentional-difference label.
