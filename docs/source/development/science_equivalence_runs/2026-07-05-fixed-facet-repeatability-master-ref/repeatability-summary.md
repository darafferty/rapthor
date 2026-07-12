# Rapthor Branch Repeatability

Scenario: `fixed-facet-carryover-repeatability`
Run root: `/tmp/rffr`
Repetitions per side: 3

## Branch Runs

| Side | Repetition | Ref | Return Code | Parset | Work Dir | Log |
| --- | --- | --- | ---: | --- | --- | --- |
| base | `rep-01` | `master` | 0 | `/tmp/rffr/inputs/base/rep-01/master_fixed_facet_carryover.parset` | `/tmp/rffw/base/rep-01` | `/tmp/rffr/base/rep-01/rapthor-command.log` |
| base | `rep-02` | `master` | 0 | `/tmp/rffr/inputs/base/rep-02/master_fixed_facet_carryover.parset` | `/tmp/rffw/base/rep-02` | `/tmp/rffr/base/rep-02/rapthor-command.log` |
| base | `rep-03` | `master` | 0 | `/tmp/rffr/inputs/base/rep-03/master_fixed_facet_carryover.parset` | `/tmp/rffw/base/rep-03` | `/tmp/rffr/base/rep-03/rapthor-command.log` |
| current | `rep-01` | `current` | 0 | `/tmp/rffr/inputs/current/rep-01/current_fixed_facet_carryover.parset` | `/tmp/rffw/current/rep-01` | `/tmp/rffr/current/rep-01/rapthor-command.log` |
| current | `rep-02` | `current` | 0 | `/tmp/rffr/inputs/current/rep-02/current_fixed_facet_carryover.parset` | `/tmp/rffw/current/rep-02` | `/tmp/rffr/current/rep-02/rapthor-command.log` |
| current | `rep-03` | `current` | 0 | `/tmp/rffr/inputs/current/rep-03/current_fixed_facet_carryover.parset` | `/tmp/rffw/current/rep-03` | `/tmp/rffr/current/rep-03/rapthor-command.log` |

## Pair Summary

| Pair | Group | Result | Failures | Warnings | FITS | H5 | Text | Diagnostics | Max Abs Delta | P99 Abs Delta | Residual RMS | Diagnostic Rel Delta | Report |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `base-rep-01_vs_base-rep-02` | `base-base` | fail | 9 | 0 | 7 | 4 | 16 | 1 | 2.325e-01 | 9.939e-06 | 3.454e-03 | 0.000% | `pairs/base-rep-01_vs_base-rep-02/branch-equivalence-report.json` |
| `base-rep-01_vs_base-rep-03` | `base-base` | fail | 9 | 0 | 7 | 4 | 16 | 1 | 2.330e-01 | 9.932e-06 | 3.480e-03 | 0.000% | `pairs/base-rep-01_vs_base-rep-03/branch-equivalence-report.json` |
| `base-rep-02_vs_base-rep-03` | `base-base` | fail | 9 | 0 | 7 | 4 | 16 | 1 | 2.260e-01 | 9.924e-06 | 3.455e-03 | 0.000% | `pairs/base-rep-02_vs_base-rep-03/branch-equivalence-report.json` |
| `current-rep-01_vs_current-rep-02` | `current-current` | fail | 8 | 0 | 7 | 6 | 16 | 1 | 2.317e-01 | 9.939e-06 | 3.479e-03 | 0.000% | `pairs/current-rep-01_vs_current-rep-02/branch-equivalence-report.json` |
| `current-rep-01_vs_current-rep-03` | `current-current` | fail | 9 | 0 | 7 | 6 | 16 | 1 | 2.331e-01 | 9.924e-06 | 3.455e-03 | 0.000% | `pairs/current-rep-01_vs_current-rep-03/branch-equivalence-report.json` |
| `current-rep-02_vs_current-rep-03` | `current-current` | fail | 9 | 0 | 7 | 6 | 16 | 1 | 2.347e-01 | 9.924e-06 | 3.491e-03 | 0.000% | `pairs/current-rep-02_vs_current-rep-03/branch-equivalence-report.json` |
| `base-rep-01_vs_current-rep-01` | `base-current` | fail | 10 | 2 | 7 | 4 | 16 | 1 | 2.303e-01 | 9.939e-06 | 3.478e-03 | 0.000% | `pairs/base-rep-01_vs_current-rep-01/branch-equivalence-report.json` |
| `base-rep-01_vs_current-rep-02` | `base-current` | fail | 10 | 2 | 7 | 4 | 16 | 1 | 2.348e-01 | 9.939e-06 | 3.437e-03 | 0.000% | `pairs/base-rep-01_vs_current-rep-02/branch-equivalence-report.json` |
| `base-rep-01_vs_current-rep-03` | `base-current` | fail | 8 | 2 | 7 | 4 | 16 | 1 | 2.295e-01 | 9.939e-06 | 3.449e-03 | 0.000% | `pairs/base-rep-01_vs_current-rep-03/branch-equivalence-report.json` |
| `base-rep-02_vs_current-rep-01` | `base-current` | fail | 10 | 2 | 7 | 4 | 16 | 1 | 2.321e-01 | 9.924e-06 | 3.450e-03 | 0.000% | `pairs/base-rep-02_vs_current-rep-01/branch-equivalence-report.json` |
| `base-rep-02_vs_current-rep-02` | `base-current` | fail | 10 | 2 | 7 | 4 | 16 | 1 | 2.315e-01 | 9.928e-06 | 3.461e-03 | 0.000% | `pairs/base-rep-02_vs_current-rep-02/branch-equivalence-report.json` |
| `base-rep-02_vs_current-rep-03` | `base-current` | fail | 10 | 2 | 7 | 4 | 16 | 1 | 2.309e-01 | 9.932e-06 | 3.432e-03 | 0.000% | `pairs/base-rep-02_vs_current-rep-03/branch-equivalence-report.json` |
| `base-rep-03_vs_current-rep-01` | `base-current` | fail | 10 | 2 | 7 | 4 | 16 | 1 | 2.287e-01 | 9.937e-06 | 3.462e-03 | 0.000% | `pairs/base-rep-03_vs_current-rep-01/branch-equivalence-report.json` |
| `base-rep-03_vs_current-rep-02` | `base-current` | fail | 10 | 2 | 7 | 4 | 16 | 1 | 2.320e-01 | 9.924e-06 | 3.472e-03 | 0.000% | `pairs/base-rep-03_vs_current-rep-02/branch-equivalence-report.json` |
| `base-rep-03_vs_current-rep-03` | `base-current` | fail | 10 | 2 | 7 | 4 | 16 | 1 | 2.269e-01 | 9.939e-06 | 3.449e-03 | 0.000% | `pairs/base-rep-03_vs_current-rep-03/branch-equivalence-report.json` |

## Interpretation

Use same-branch pairs to estimate run-to-run scatter before accepting or tightening product-specific tolerances. Cross-branch differences that are consistently larger than same-branch scatter need a scientific explanation, a bug fix, or an explicit intentional-difference label.
