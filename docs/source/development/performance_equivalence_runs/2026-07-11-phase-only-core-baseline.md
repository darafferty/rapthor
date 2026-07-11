# 2026-07-11 Phase-Only Core Performance Baseline

Scenario: `phase-only-core-performance-baseline`

Base branch: `master` at `17448437b78583f1eaf38112a524b2dbe5f34bb8`

Current branch: `c42ae88db7a8d9b4c641563f123bb7da3b18a4c0`

Run root: `/tmp/rpbe0711`

## Outcome

All six pipeline runs completed successfully: three repetitions on `master` and
three repetitions on the current branch. The wrapper exited with code `1`
because every pairwise product comparison failed the current strict science
tolerances, including the same-branch repeatability pairs.

This run is therefore an advisory performance baseline, not a formal science
gate pass. It is still useful evidence because the same-branch comparisons show
that the current tolerance model is stricter than the observed run-to-run
scatter, especially on `master`.

## Runtime Result

The current branch was substantially faster in this scenario:

| Side | Runs | Min (s) | Median (s) | Max (s) |
| --- | ---: | ---: | ---: | ---: |
| `master` | 3 | 606.397 | 631.209 | 658.879 |
| current | 3 | 324.417 | 331.178 | 372.995 |

Current-vs-master median delta: **-47.5%**.

## Operation Timing

| Operation | Master Median (s) | Current Median (s) | Delta |
| --- | ---: | ---: | ---: |
| `calibrate_1` | 24.209 | 8.132 | -66.4% |
| `calibrate_2` | 24.692 | 8.104 | -67.2% |
| `calibrate_3` | 23.985 | 8.125 | -66.1% |
| `calibrate_4` | 26.080 | 11.276 | -56.8% |
| `image_1` | 125.558 | 70.711 | -43.7% |
| `image_2` | 118.947 | 65.315 | -45.1% |
| `image_3` | 117.723 | 66.329 | -43.7% |
| `image_4` | 124.293 | 67.297 | -45.9% |
| `mosaic_1` | 3.523 | 1.121 | -68.2% |
| `mosaic_2` | 3.609 | 1.122 | -68.9% |
| `mosaic_3` | 3.695 | 1.108 | -70.0% |
| `mosaic_4` | 3.553 | 1.143 | -67.8% |

The speedup is broad rather than isolated to one operation family.

## Science Comparison

The current strict comparator is not yet calibrated for repeatability:

| Pair Group | Pairs | Failed Pairs | Failure Count Range | Max Abs Delta Range | P99 Abs Delta Range | Residual RMS Range | Max Diagnostic Relative Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `master` vs `master` | 3 | 3 | 25-120 | 0.435586-0.437062 | 9.93e-06-7.90e-03 | 4.59e-03-4.67e-03 | 0.218% |
| current vs current | 3 | 3 | 17-18 | 1.46e-05-1.41e-03 | 9.92e-06-9.94e-06 | 4.16e-06-4.16e-06 | 0.014% |
| `master` vs current | 9 | 9 | 33-127 | 0.222021-0.222024 | 9.92e-06-7.90e-03 | 3.25e-03-3.29e-03 | 0.216% |

The cross-branch differences are within the `master` same-branch envelope for
the largest image/product metrics. The current branch is also materially more
self-repeatable than `master` in this run.

Notable strict failures:

- `field-MFS-model-pb.fits(.fz)` dominates the large absolute differences.
- Small FITS residual/image differences appear at approximately `1e-6` to
  `1e-5`, also present in same-branch pairs.
- A small h5parm numeric difference was reported in some cross-branch pairs
  (`fast_phase`, max absolute difference about `9.5e-6`).
- Output-record auxiliary artifact basename differences are warnings only.
- Image diagnostics were effectively stable; cross-branch diagnostic relative
  deltas stayed within the observed `master` repeatability envelope.

## Interpretation

This is encouraging performance evidence for the current branch, but it should
not be treated as the final performance-equivalence gate. The next gate
iteration should convert same-branch repeatability into explicit product
tolerances or pass/warn bands, then rerun the scenario.

## Follow-Up

- Keep short run roots for branch-equivalence runs that exercise the legacy
  `master` script path; a longer workspace run root triggered a PyBDSF
  multiprocessing `AF_UNIX path too long` failure before this successful run.
- Update the gate to classify cross-branch deltas relative to same-branch
  repeatability scatter.
- Re-run the phase-only core scenario after the tolerance/pass-band update.
- Treat this run as a performance baseline for later optimisation work, with
  the caveat that the science comparator is still advisory.
