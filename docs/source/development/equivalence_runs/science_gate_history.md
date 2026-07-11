# Rapthor CWL-to-Prefect Equivalence Report

Historical detailed run log. The current science-gate status summary now lives
in `EQUIVALENCE_REPORT.md`; the durable comparison method lives in
`docs/source/development/science_equivalence_contract.rst`.

Generated: 2026-06-11

Archived for post-cutover cleanup: 2026-06-12

Latest on-disk report scan: 2026-07-11

Method contract: `docs/source/development/science_equivalence_contract.rst`

## Summary

Rapthor's Prefect/Dask execution path was validated against the legacy CWL path
before the CWL runtime and workflow files were removed from the codebase.

The migration equivalence contract was:

- preserve the public parset and strategy contract
- preserve operation ordering and top-level process semantics
- preserve finalizer-visible `Field`, `Observation`, and `Sector` state
- preserve output filenames, output-record shapes, and product locations
- preserve restart/reset behaviour through `.done` and `.outputs.json`
- preserve local execution behaviour

That contract was satisfied for the required local scenario matrix. The old
active CWL equivalence harness, CWL workflow/parset files, and `cwltool`
validation tests have since been removed as part of post-cutover cleanup. This
file is now the historical parity record.

Current gate verdict as of 2026-07-11: the saved-reference matrix, focused
DD-plus-DI full-Jones branch-vs-master rerun, three-repeat normalized
branch-repeatability envelope, and full integration suite support accepting the
current branch for the covered scientific contract. Remaining cross-branch
warnings in the strongest repeatability envelope are auxiliary output-record
artifact names only; FITS, h5parm, text/region, catalog, and image-diagnostic
differences are within the same-branch repeatability envelope. The next
scientific checks should continue as risk-based option scenarios rather than
broader default-like reruns. The first option-matrix scenarios now pass for
provided sky-model flux-scale normalization, DP3 image-based predict, WSClean
predict, and BDA/averaging on the rich demo data. The latest 2026-07-09
saved-reference rerun also passes all active saved scenarios after correcting
the saved normalization fixture to use valid distinct reference frequencies.
The 2026-07-11 post-task-split rerun passes the same active saved-reference
matrix and the same active branch-vs-master option-matrix rows after the image,
mosaic, and calibration task-boundary work.
The multi-sector mosaic option-matrix row is documented as a current-branch
coverage/stored-reference target rather than a branch-vs-master gate because
legacy `master` fails before comparison in the CWL multi-sector image scatter.

## Science Equivalence Gate Decision

Status: accepted for the covered scientific contract. The current branch is
ready for the first low-risk scalability slice, provided product-affecting
changes continue to rerun the relevant focused equivalence checks.

The accepted contract is:

- keep operation order, operation presence, product presence, product basenames,
  FITS/HDF5/text product structure, h5parm solset/soltab names, axes, shapes,
  finite masks, source counts, and primary catalog values strict
- accept FITS image, source-catalog diagnostic, and image-diagnostic numeric
  differences only when they are inside the measured same-branch repeatability
  envelope
- treat legacy CWL output-record metadata, auxiliary diagnostic plot artifact
  names, and generated preview PNGs as non-scientific review aids
- keep raw FITS, h5parm, sky-model, catalog, region, and diagnostic JSON/report
  products as the scientific comparison surface

| Evidence | Latest tracked report | Result | Gate decision |
| --- | --- | --- | --- |
| Saved-reference final gate | `docs/source/development/equivalence_runs/2026-07-06-saved-reference-final-gate/` | pass | Current Prefect/Dask path preserves the saved legacy product contract for all non-stale references. |
| Core DD phase plus DI full-Jones repeatability | `docs/source/development/equivalence_runs/2026-07-06-dd-phase-plus-di-fulljones-normalized-repeatability-master-ref/` | pass | All 15 base-base, current-current, and base-current pairs pass; cross-branch warnings are auxiliary output-record artifact names only. |
| Core DD phase plus DI full-Jones focused smoke | `docs/source/development/equivalence_runs/2026-07-06-dd-phase-plus-di-fulljones-normalized-master-ref/` | strict fail, classified | All h5parm products pass; remaining strict failures are repeatability-bounded small image residuals, PyBDSF diagnostic catalog columns, DS9 formatting, and legacy metadata shape. |
| Risk-based option matrix | `docs/source/development/equivalence_runs/2026-07-06-option-matrix/` | pass for active rows | Provided normalization sky models, DP3 image-based predict, WSClean predict, and BDA/averaging pass against `master`; screens are skipped until target tool support is available. |
| Flexible carry-over matrix | `2026-07-05-*carryover*` and `2026-07-05-*mode-boundary*` reports | intentional strict differences | Current behavior follows the explicit strategy contract rather than copying unsafe implicit master state. |

Accepted current-vs-master intentional differences:

- Previous-cycle solutions may seed later matching solves, but they are optimizer
  seeds only. They are not silently applied during imaging after a new
  calibration step unless they are part of that cycle's calibration state.
- DD previous-cycle seeds require compatible directions. Fixed facets allow
  compatible fast and medium phase seeds; changed/regrouped facets block those
  seeds. Master can still pass stale DD seeds across direction changes.
- Previous full-Jones products may seed compatible later DI full-Jones solves,
  using `[amplitude000, phase000]`. They are not silently carried into a later
  DD-only imaging step after new DD calibration.
- Slow-gain/default-like current runs preserve the active amplitude solution in
  the final field-solutions h5parm. The current branch should not mimic the
  master run that logs an h5parm-combination broadcast error and finishes with
  phase-only active solutions.

Known target-environment or reference caveats:

- The `screens` option-matrix row remains skipped until reliable IDGCal/screen
  support is available in the target environment.
- MPI WSClean and Slurm/external-Dask checks remain deployment checks rather
  than local science-gate blockers.
- Legacy master repeatability runs must use short `/tmp` paths to avoid the
  PyBDSF/Toil `AF_UNIX path too long` failure.

Possible master bugs or legacy limitations to investigate separately:

- slow-gain h5parm combination logs a broadcasting error but the master run
  still returns success and leaves final field solutions without amplitude.
  This appears to be product-shape dependent rather than universal: the
  historical ICAL benchmark run in `runs/2026-07-02-ical-1node/` used legacy
  Rapthor `2.2.dev117+g01a81e1`, ran `combine_h5parms.py ... p1p2a2_diagonal`
  for cycles 3-7, and completed successfully because the slow-gain h5parms had
  a full frequency axis, for example `2 times, 26 freqs, 68 ants, 16 dirs, 2
  pols` in `calibrate_3`. The failing saved-reference case instead hit a
  singleton-axis broadcast from `(24,1,8,5,2)` into `(24,8,5,2)`. Reproduce and
  classify the exact trigger, likely tied to dataset/solve grid/frequency-step
  shape, before treating all master slow-gain runs as affected.
- previous DD fast-phase seeds are reused across changed facet/direction sets
  without a direction-compatibility guard
- previous DI full-Jones products can be applied in later DD-only imaging after
  a new calibration step, making correction state implicit rather than
  strategy-scoped

Rerun policy: documentation, report-only, and preview-artifact changes may use
the current evidence plus focused tests. Changes to calibration, prediction,
imaging, h5parm collection, FITS products, sky models, source catalogs, or
diagnostics should rerun the relevant saved-reference and branch-vs-master
scenario before new scalability work is judged.

## Evidence

Saved CWL reference artifacts were captured from legacy commit:

```text
4cfd2abe2fe815724e3f1c390d789eea249becef
```

The saved-reference comparison was run against the current Prefect/Dask path in
the development container. It compared backend-neutral summaries of:

- operation order
- operation `.done` markers
- operation output records from `.outputs.json` or legacy
  `pipeline_outputs.json`
- final products under `images`, `h5parms`, `skymodels`, and `regions`
- `logs/diagnostics.txt`, when scoped
- `field_state.json`, when present

Product summaries included FITS shape/dtype/statistics, h5parm
solset/soltab/dataset/axis structure, sky-model source and patch counts,
region-file content, and generic product basenames.

## 2026-07-09 Science Gate Rerun

The saved-reference gate was rerun on 2026-07-09 after the current normalization
contract required distinct reference frequencies. The first rerun exposed that
the historical saved normalization fixture still used duplicate reference
frequencies. The fixture-preparation helper now rewrites the saved
normalization scenario to use:

```text
[142000000.0, 142001000.0]
```

The final saved-reference rerun is:

```text
runs/science-gate-20260709-saved-reference-rerun/equivalence-report.json
runs/science-gate-20260709-saved-reference-rerun/equivalence-report.md
```

All active saved-reference scenarios passed. Remaining warnings are optional
output-record artifact basename differences only.

| Scenario | Result | Ops | Records | FITS | Image HDUs | Table HDUs | H5 | Text |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `dd_only_calibration` | pass | 4 | 4 | 6 | 5 | 1 | 3 | 10 |
| `di_only_calibration` | pass | 5 | 5 | 6 | 5 | 1 | 2 | 9 |
| `full_stokes_clean_disabled` | pass | 4 | 4 | 9 | 8 | 1 | 3 | 8 |
| `image_cube` | pass | 4 | 4 | 7 | 6 | 1 | 3 | 12 |
| `normalization` | pass | 5 | 5 | 7 | 6 | 1 | 4 | 12 |
| `peeling` | pass | 4 | 4 | 6 | 5 | 1 | 3 | 11 |
| `restart` | pass | 4 | 4 | 6 | 5 | 1 | 3 | 10 |

The risk-based option matrix was rerun from:

```text
runs/science-gate-20260709-option-matrix-final/option-matrix-summary.json
runs/science-gate-20260709-option-matrix-final/option-matrix-summary.md
```

The four active branch-vs-master option rows passed again. Each row has one
classified auxiliary output-record warning and no failures.

| Scenario | Result | Pairs | Passed Pairs | Failures | Warnings | FITS | H5 | Text | Diagnostics |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `normalization-rich-demo` | pass | 1 | 1 | 0 | 1 | 8 | 3 | 12 | 1 |
| `prediction-path-image-based` | pass | 1 | 1 | 0 | 1 | 7 | 2 | 10 | 1 |
| `prediction-path-wsclean` | pass | 1 | 1 | 0 | 1 | 7 | 2 | 10 | 1 |
| `bda-averaging` | pass | 1 | 1 | 0 | 1 | 7 | 2 | 10 | 1 |
| `multi-sector-mosaic` | skipped | 0 | 0 | 0 | 0 | - | - | - | - |
| `screens` | skipped | 0 | 0 | 0 | 0 | - | - | - | - |

The skipped multi-sector mosaic row was also run directly to establish the
failure mode:

```text
runs/science-gate-20260709-option-matrix-mosaic-recheck/
```

The current Prefect/Dask branch completed the multi-sector mosaic scenario, but
legacy `master` exited during `image_1` before products could be compared. The
failure occurs in the generated CWL image scatter for the 2x2 sector grid, where
one scatter input is not per-sector. This is a legacy orchestration limitation,
not evidence of a current-branch scientific mismatch. Keep multi-sector mosaic
covered by current-branch integration/benchmark runs for now, and promote it to
a stored-reference science gate once a stable reference run is captured.

## 2026-07-11 Post-Task-Split Science Gate Rerun

The saved-reference gate was rerun on 2026-07-11 after the image, mosaic, and
calibration task-boundary work. The archived report is:

```text
docs/source/development/equivalence_runs/2026-07-11-post-task-split-saved-reference/equivalence-report.json
docs/source/development/equivalence_runs/2026-07-11-post-task-split-saved-reference/equivalence-report.md
```

All active saved-reference scenarios passed. Remaining warnings are optional
output-record artifact basename differences only.

| Scenario | Result | Ops | Records | FITS | Image HDUs | Table HDUs | H5 | Text |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `dd_only_calibration` | pass | 4 | 4 | 6 | 5 | 1 | 3 | 10 |
| `di_only_calibration` | pass | 5 | 5 | 6 | 5 | 1 | 2 | 9 |
| `full_stokes_clean_disabled` | pass | 4 | 4 | 9 | 8 | 1 | 3 | 8 |
| `image_cube` | pass | 4 | 4 | 7 | 6 | 1 | 3 | 12 |
| `normalization` | pass | 5 | 5 | 7 | 6 | 1 | 4 | 12 |
| `peeling` | pass | 4 | 4 | 6 | 5 | 1 | 3 | 11 |
| `restart` | pass | 4 | 4 | 6 | 5 | 1 | 3 | 10 |

The risk-based branch-vs-master option matrix was rerun against `master` commit
`17448437`. The archived summary and per-scenario reports are:

```text
docs/source/development/equivalence_runs/2026-07-11-post-task-split-option-matrix/option-matrix-summary.json
docs/source/development/equivalence_runs/2026-07-11-post-task-split-option-matrix/option-matrix-summary.md
```

The four active branch-vs-master option rows passed. Each active row has one
classified auxiliary output-record warning and no failures.

| Scenario | Result | Pairs | Passed Pairs | Failures | Warnings | FITS | H5 | Text | Diagnostics |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `normalization-rich-demo` | pass | 1 | 1 | 0 | 1 | 8 | 3 | 12 | 1 |
| `prediction-path-image-based` | pass | 1 | 1 | 0 | 1 | 7 | 2 | 10 | 1 |
| `prediction-path-wsclean` | pass | 1 | 1 | 0 | 1 | 7 | 2 | 10 | 1 |
| `bda-averaging` | pass | 1 | 1 | 0 | 1 | 7 | 2 | 10 | 1 |
| `multi-sector-mosaic` | skipped | 0 | 0 | 0 | 0 | - | - | - | - |
| `screens` | skipped | 0 | 0 | 0 | 0 | - | - | - | - |

## Current Strengthened Saved-Reference Run

The current strengthened saved-reference run is:

```text
docs/source/development/equivalence_runs/2026-07-06-saved-reference-final-gate/equivalence-report.json
```

It was generated in the development container on 2026-07-06 and passed. The
matching Markdown report is:

```text
docs/source/development/equivalence_runs/2026-07-06-saved-reference-final-gate/equivalence-report.md
```

References that encode older scientific contracts are skipped by default, but
can still be run explicitly with `--include-stale-references`. The old
`di_full_jones_calibration` CWL fixture is now stale because it predates the
current same-cycle DI full-Jones application during image preparation; full-Jones
behavior is covered by fresh branch-vs-master scenarios instead.

| Scenario | Result | Ops | Records | FITS | Image HDUs | Table HDUs | H5 | Text |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `dd_only_calibration` | pass | 4 | 4 | 6 | 5 | 1 | 3 | 10 |
| `di_only_calibration` | pass | 5 | 5 | 6 | 5 | 1 | 2 | 9 |
| `full_stokes_clean_disabled` | pass | 4 | 4 | 9 | 8 | 1 | 3 | 8 |
| `image_cube` | pass | 4 | 4 | 7 | 6 | 1 | 3 | 12 |
| `normalization` | pass | 5 | 5 | 7 | 6 | 1 | 4 | 12 |
| `peeling` | pass | 4 | 4 | 6 | 5 | 1 | 3 | 11 |
| `restart` | pass | 4 | 4 | 6 | 5 | 1 | 3 | 10 |

All scenarios have only optional output-record warnings for newer
astrometry-corrected image products and local prepared-MS record names. Final
FITS, h5parm, sky-model, and region products pass.

## Product Statistic Checks

The JSON report stores pass/fail results, product counts, and FITS image/table
statistics used by the strengthened checks.

FITS image products compare:

- finite/NaN masks
- WCS/header keys
- finite pixel count plus `mean`, `std`, `rms`, `min`, and `max`, using
  `atol = 1e-6` and `rtol = 1e-3`
- pixel values using `np.allclose(..., atol=1e-6, rtol=1e-3, equal_nan=True)`
- a robust residual fallback for sparse float-level image outliers:
  `max_abs_delta <= 1e-5`, `p99_abs_delta <= 1e-6`, and
  `residual_rms <= 1e-6`
- a relative residual fallback for bright images where absolute float jitter is
  larger but scientifically tiny:
  `max_abs_delta <= 1e-4 * image_scale`,
  `p99_abs_delta <= 2e-5 * image_scale`, and
  `residual_rms <= 1e-5 * image_scale`, where `image_scale` is the larger of
  reference RMS, reference MAD-derived noise, and `atol`
- per-plane residual metrics for cubes and Stokes products

The table below shows the worst FITS image product in each scenario by maximum
absolute residual.

| Scenario | FITS products | Image HDUs | Table HDUs | Worst image product | Max abs delta | P99 abs delta | Residual RMS | RMS / Ref MAD |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| `dd_only_calibration` | 6 | 5 | 1 | `field-MFS-image-pb.fits` | 3.165e-10 | 2.910e-11 | 9.710e-12 | 2.294e-07 |
| `di_only_calibration` | 6 | 5 | 1 | `field-MFS-dirty.fits` | 2.292e-10 | 2.910e-11 | 1.107e-11 | 2.600e-07 |
| `full_stokes_clean_disabled` | 9 | 8 | 1 | `field-MFS-I-image-pb.fits` | 4.657e-09 | 4.366e-11 | 2.884e-11 | 6.812e-07 |
| `image_cube` | 7 | 6 | 1 | `sector_1_I_freq_cube.fits` | 4.547e-10 | 5.821e-11 | 1.972e-11 | 4.554e-07 |
| `normalization` | 7 | 6 | 1 | `field-MFS-residual.fits` | 2.833e-05 | 6.169e-06 | 2.373e-06 | 5.591e-06 |
| `peeling` | 6 | 5 | 1 | `field-MFS-image-pb.fits` | 2.176e-06 | 2.384e-07 | 9.129e-08 | 1.833e-07 |
| `restart` | 6 | 5 | 1 | `field-MFS-image-pb.fits` | 2.401e-10 | 2.910e-11 | 9.181e-12 | 2.169e-07 |

HDF5 products compare dataset names and shapes. Numeric datasets use
`np.allclose(..., atol=1e-6, rtol=1e-3, equal_nan=True)`, while non-numeric
datasets use exact array equality. All HDF5 checks passed in the current run.

Text-like products compare sky-model `lines` and `patches`, beam tables with
`atol = 1e-6` and `rtol = 1e-2`, and all other text and region files exactly.
All text-like product checks passed in the current run.

## Risk-Based Option Matrix

The first focused option matrix run is tracked under:

```text
docs/source/development/equivalence_runs/2026-07-06-option-matrix/
```

The first active rows compare `master` with the current branch on the generated
rich demo data:

- `normalization-rich-demo` enables flux-scale normalization with explicit
  two-frequency reference sky-model snapshots at 120 MHz and 160 MHz.
- `prediction-path-image-based` enables DD fast+medium phase calibration with
  DP3 image-based predict.
- `prediction-path-wsclean` enables DD fast+medium phase calibration with
  WSClean predict.
- `bda-averaging` enables calibration BDA plus imaging visibility averaging/BDA
  with an imaging-averaging cap that leaves four unique channels in the
  8-channel rich demo data.

All four rows passed with one non-blocking auxiliary output-record warning
for calibration plot artifact names. FITS image residuals, h5parm products,
text products, source-catalog tables, and image diagnostics pass the
strengthened branch-equivalence checks.

The run also caught and fixed three current-branch payload/compatibility bugs:
provided `normalization_skymodels` and
`normalization_reference_frequencies` were not passed through to
`normalize_flux_scale`; model-image coordinates from LSMTool were not converted
to payload lists; and WSClean predict called the installed LSMTool
`read_ds9_region_file` with an unsupported `extra_boundary` keyword. Focused
tests now cover these paths.

| Scenario | Result | Pairs | Passed Pairs | Failures | Warnings | FITS | H5 | Text | Diagnostics |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `normalization-rich-demo` | pass | 1 | 1 | 0 | 1 | 8 | 3 | 12 | 1 |
| `prediction-path-image-based` | pass | 1 | 1 | 0 | 1 | 7 | 2 | 10 | 1 |
| `prediction-path-wsclean` | pass | 1 | 1 | 0 | 1 | 7 | 2 | 10 | 1 |
| `bda-averaging` | pass | 1 | 1 | 0 | 1 | 7 | 2 | 10 | 1 |

The remaining matrix row is intentionally skipped until target tool support is
ready: screens.

## Branch-Vs-Master Default-Like Run

A multi-cycle default-like branch comparison was run on 2026-07-04 with
`master` as the reference. The base checkout resolved to:

```text
17448437b78583f1eaf38112a524b2dbe5f34bb8
```

The report paths are:

```text
runs/rbe-default-like-master-ref-codex/branch-equivalence-report.json
runs/rbe-default-like-master-ref-codex/branch-equivalence-report.md
```

Both branch executions completed successfully, but the product comparison
failed.

| Scenario | Base ref | Base RC | Current RC | Result | Ops | Records | FITS | Image HDUs | Table HDUs | H5 | Text | Warnings | Failures |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `benchmark-default-like` | `master` | 0 | 0 | fail | 12 | 12 | 28 | 20 | 3 | 14 | 37 | 8 | 142 |

Key findings:

- operation counts matched, so both branches ran the same broad four-cycle
  calibrate/image/mosaic shape
- all calibrate and image output-record summaries differed and need inspection
- current-branch outputs are missing the `field-MFS-image-pb-ast` products that
  `master` generated
- early image-cycle residuals are close but above the strengthened tolerance
  (`max_abs_delta` around `1.5e-05`, residual RMS around `4.2e-06`)
- later image cycles diverge materially, with worst image residuals reaching
  `max_abs_delta = 6.671e-01` and residual RMS up to `4.203e-02`
- the large cycle 3/4 divergence is explained by a legacy master slow-gain
  combination failure: master runs `combine_h5parms.py ... p1p2a2_diagonal`,
  logs `ValueError: could not broadcast input array from shape ...`, but the CWL
  step still completes successfully and leaves the active `field-solutions.h5`
  phase-only
- the current branch produces active cycle 3/4 `field-solutions.h5` products
  with both `phase000` and `amplitude000`, then WSClean applies
  `amplitude000,phase000`; master applies only `phase000`
- smaller early-cycle residuals are probably also affected by runtime-shape
  differences in the manual scenario, including WSClean `-parallel-gridding`
  and `-abs-mem` values

Treat this as a failed scientific equivalence gate. The next investigation
should decide the intended reference contract for slow-gain amplitudes before
any scalability changes are made. Either patch the master reference checkout so
slow-gain amplitudes are combined/applied as intended, or mark the master
behavior as a legacy bug and use an adapted phase-only current scenario when
strict branch-vs-master parity is required.

## Branch-Vs-Master Phase-Only Rerun

A four-cycle phase-only branch comparison was rerun on 2026-07-05 after the
master feature ports, with `master` commit
`17448437b78583f1eaf38112a524b2dbe5f34bb8` as the reference.

The tracked compact report bundle is:

```text
docs/source/development/equivalence_runs/2026-07-05-phase-only-master-ref/
```

Both branch executions completed successfully. The strict product comparison
still failed, but the report now includes image diagnostic deltas and compact
side-by-side PNG comparisons for selected image products and solution plots.

| Scenario | Base ref | Base RC | Current RC | Result | Ops | Records | FITS | Image HDUs | Table HDUs | H5 | Text | Diagnostics | Visuals | Warnings | Failures |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `benchmark-phase-only` | `master` | 0 | 0 | fail | 12 | 12 | 28 | 24 | 4 | 8 | 37 | 4 | 20 | 4 | 124 |

Key findings:

- `field-MFS-image-pb-ast` products are now present on both branches, so the
  earlier astrometry-corrected image product gap has been closed.
- Top-level image diagnostics are very close despite strict pixel failures:
  source counts match in all four image cycles, theoretical RMS is identical,
  and the largest relative diagnostic deltas are about `0.23%` in final-cycle
  dynamic range and minimum RMS noise.
- H5 phase solutions start equivalent and then diverge through the cycles:
  cycle 1 is exact, cycle 2 has maximum phase delta about `2.18e-07`, cycle 3
  about `1.60e-03`, and cycle 4 about `6.12` with small source-coordinate
  differences in `sol000/source`.
- Restored/residual image residuals are small in early cycles but grow by cycle:
  the cycle 3 restored-image RMS residual is about `1.96e-04`, while the final
  uncompressed image residual is about `4.64e-04`; the final dirty image differs
  more strongly with RMS residual about `2.04e-03`.
- The remaining failures are dominated by later-cycle FITS pixel/statistic
  differences, h5parm phase/source differences, sky-model summary differences,
  source-catalog columns, and text/region products.
- Current-branch output records are leaner path-only records, while master
  records CWL metadata such as checksums and file sizes; this accounts for the
  remaining output-record summary warnings.

Interpretation:

This is not yet a passing branch-vs-master scientific gate. It is, however, a
more useful diagnostic run than the 2026-07-04 phase-only report: product
presence is aligned, diagnostics suggest the high-level image quality is nearly
unchanged, and the residuals point to accumulated later-cycle source
selection/sky-model/h5parm drift rather than a basic execution failure.

The next equivalence task should investigate the cycle 3/4 h5parm and
sky-model/source-selection divergence, then decide which differences are
intentional product-contract changes and which should be fixed or tolerated by
product-specific comparison rules.

## Branch-Vs-Master Phase-Only Initial-Solutions Rerun

A follow-up four-cycle phase-only branch comparison was run on 2026-07-05 after
aligning DD previous-cycle solve initialization with the master phase-only
behavior. The tracked compact report bundle is:

```text
docs/source/development/equivalence_runs/2026-07-05-phase-only-initial-solutions-master-ref/
```

Both branch executions completed successfully. The strict product comparison
still failed, but the initial-solution inputs are now aligned: cycles 2-4 pass
only the previous cycle's fast-phase h5parm as the DD solve seed on both
branches, while medium/slow solve seeds remain unset for this phase-only
scenario.

| Scenario | Base ref | Base RC | Current RC | Result | Ops | Records | FITS | Image HDUs | Table HDUs | H5 | Text | Diagnostics | Visuals | Warnings | Failures |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `benchmark-phase-only-initial-solutions-fast-only` | `master` | 0 | 0 | fail | 12 | 12 | 28 | 24 | 4 | 8 | 37 | 4 | 20 | 4 | 31 |

Key findings:

- The previous stale-solution warnings are gone from the current command log,
  and the obsolete `parallel_gridding_threads` parset warning remains fixed.
- H5 phase divergence is now tiny: cycle 1 is exact, cycles 2 and 3 have maximum
  phase deltas below `6e-07`, and cycle 4 has maximum phase delta
  `1.470e-05` with RMS `3.592e-06`. This replaces the earlier cycle-4
  multi-radian phase divergence.
- Image diagnostics are very close: source counts and theoretical RMS match in
  all four image cycles, and the largest relative diagnostic delta is about
  `0.005%` in final-cycle true-sky noise/dynamic-range metrics.
- The remaining strict failures are dominated by float-level FITS residuals,
  sparse `field-MFS-model-pb` component differences, exact text/region
  differences, and calibrate output-record summary differences.
- Current and master still differ in output-record metadata shape: master
  records CWL file metadata such as checksums and sizes, while the current path
  records leaner path-oriented products.

Interpretation:

The initial-solution behavior is now aligned for the master-compatible
phase-only scenario. The remaining failures look like comparison-contract and
product-contract work rather than a calibration initialization bug. The next
equivalence work should add product-specific tolerances/semantic comparison for
phase h5parms, sparse model images, and deterministic text/region products,
while keeping operation ordering, product presence, shapes, and finite values
strict.

## Branch-Vs-Master DD Phase Plus DI Full-Jones Run

The first essential branch-vs-master scenario was run on 2026-07-05. It uses a
single cycle that runs DD fast+medium phase-only calibration first, then DI
full-Jones calibration, matching the legacy master operation order. The tracked
compact report bundle is:

```text
docs/source/development/equivalence_runs/2026-07-05-dd-phase-plus-di-fulljones-master-ref/
```

Both branch executions completed successfully. The strict product comparison
failed, but operation counts, product presence, source counts, and top-level
image diagnostics are close.

| Scenario | Base ref | Base RC | Current RC | Result | Ops | Records | FITS | Image HDUs | Table HDUs | H5 | Text | Diagnostics | Visuals | Warnings | Failures |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `dd-phase-plus-di-fulljones` | `master` | 0 | 0 | fail | 5 | 5 | 7 | 6 | 1 | 3 | 10 | 1 | 5 | 2 | 47 |

Key findings:

- The mixed-mode execution shape is aligned: both branches run
  `calibrate_1`, `predict_di_1`, `calibrate_di_1`, `image_1`, and `mosaic_1`.
- Source counts match (`11` sources), theoretical RMS is identical, and the
  largest image-diagnostic relative deltas are about `0.24%`.
- Restored-image residuals are small but above the current strict tolerance:
  `field-MFS-image-pb.fits` has max absolute delta about `1.025e-02`, p99
  absolute delta about `1.066e-04`, and residual RMS about `1.673e-04`.
- The DI full-Jones h5parm differs in `sol000/amplitude000/val` with maximum
  absolute delta about `1.14e-03`.
- Remaining failures include source-catalog flux/error columns, exact
  `sector_1_facets_ds9.reg` text comparison, and output-record summary shape
  differences for `calibrate_1` and `calibrate_di_1`.

Interpretation:

This scenario confirms that the current branch can represent and execute the
legacy DD-then-DI full-Jones mixed calibration order. It is not a passing
scientific equivalence gate yet. Follow-up inspection found that the active DP3
solve commands, image-preparation applycal payloads, WSClean imaging commands,
and full-Jones applycal soltabs match between branches. The systematic
full-Jones amplitude offset is explained by a missing current-branch
post-processing step: legacy master normalizes the collected full-Jones gain
amplitudes with flagging and smoothing disabled before plotting and imaging.
The current branch now ports that full-Jones normalization behavior. Rerun this
focused scenario before interpreting the pre-fix residuals as scientific
differences or tolerance work.

## DD Phase Plus DI Full-Jones Repeatability Envelope

A three-repeat branch repeatability run was completed for the DD phase plus DI
full-Jones scenario. The run used short paths (`/tmp/rfjr` and `/tmp/rfjw`) and
reused the short-path master checkout/venv so the legacy master CWL imaging
path did not hit the PyBDSF/Toil `AF_UNIX path too long` failure.

The tracked compact report bundle is:

```text
docs/source/development/equivalence_runs/2026-07-05-dd-phase-plus-di-fulljones-repeatability-master-ref/
```

All six branch executions completed successfully: three `master` repeats and
three current-branch repeats. Master was stable within the current strict
comparison rules. The current branch had one outlying repetition: pairs
involving `current` `rep-01` failed, while `current` `rep-02` versus `rep-03`
passed. All branch-vs-master pairs failed and were systematically larger than
same-branch master scatter.

| Pair Group | Pairs | Passed | Max Failures | Max Warnings | Max Abs Delta | Max P99 Abs Delta | Max Residual RMS | Max Diagnostic Rel Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `base-base` | 3 | 3 | 0 | 0 | 1.431e-06 | 8.941e-08 | 3.340e-08 | 1.909e-04 |
| `current-current` | 3 | 1 | 29 | 0 | 2.745e-03 | 1.061e-05 | 4.204e-06 | 1.351e-04 |
| `base-current` | 9 | 0 | 61 | 2 | 1.025e-02 | 9.260e-04 | 3.872e-04 | 2.390e-03 |

Key findings:

- Master same-branch pairs all pass, so this scenario is stable on the legacy
  reference branch under the current strict rules.
- The current branch's `rep-01` differs from `rep-02` and `rep-03`, but the
  current-current deltas remain much smaller than the branch-vs-master deltas.
- Cross-branch differences are concentrated in restored/dirty/residual FITS
  metrics, the `fulljones-solutions.h5:sol000/amplitude000/val` dataset,
  PyBDSF source-catalog flux/error/rms columns, exact DS9 region text, and
  output-record metadata summaries.
- Unlike the fixed-facet DD carry-over repeatability envelope below, the
  DD-plus-DI full-Jones branch-vs-master residuals are not explained by
  ordinary same-branch scatter alone.

Interpretation:

This envelope was useful because it showed the DD-plus-DI full-Jones
branch-vs-master split was systematic rather than ordinary master scatter.
Subsequent product inspection traced the split to the missing current-branch
full-Jones gain normalization step described above. Treat this envelope as
pre-fix evidence: rerun the focused DD-plus-DI full-Jones comparison after the
normalization port, then refresh the three-repeat envelope only if the focused
rerun still shows residuals larger than same-branch scatter.

## DD Phase Plus DI Full-Jones Normalized Rerun

A focused branch-vs-master rerun was completed on 2026-07-06 after porting the
legacy full-Jones gain normalization step into the current Prefect/Dask
calibration collection path. The run used short paths (`/tmp/rfjn` and
`/tmp/rfjnw`) and the tracked input snapshots copied into:

```text
docs/source/development/equivalence_runs/2026-07-06-dd-phase-plus-di-fulljones-normalized-master-ref/
```

Both branches returned `0`. Strict comparison still failed, but the full-Jones
h5parm difference that dominated the earlier run is gone: all three h5parm
products pass the strengthened comparison. The remaining failures are now
limited to small image residuals, PyBDSF/source-catalog uncertainty columns, the
known DS9 facet-region text difference, and output-record metadata shape.
The compact report now classifies every remaining warning/failure without
changing strict pass/fail status.

| Scenario | Base RC | Current RC | Result | Ops | Records | FITS | Image HDUs | Table HDUs | H5 | Text | Diagnostics | Warnings | Failures |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `dd-phase-plus-di-fulljones-normalized-smoke` | 0 | 0 | fail | 5 | 5 | 7 | 6 | 1 | 3 | 10 | 1 | 2 | 29 |

Key deltas after the fix:

- `fulljones-solutions.h5` no longer appears in the failure list.
- The restored image max absolute residual dropped from about `1.025e-02` in
  the pre-fix report to about `2.486e-05`.
- The residual-image RMS is about `4.194e-06`, and the restored-image residual
  RMS is about `3.978e-06`.
- Image diagnostics are essentially aligned: source counts and theoretical RMS
  match exactly, and the largest relative diagnostic delta is about `0.021%`.
- Difference classification: 4 small image residuals, 1 sparse model-image
  residual, 23 PyBDSF diagnostic catalog columns, 1 DS9 region text-formatting
  difference, and 2 legacy output-record metadata warnings. No h5parm
  differences, primary source-flux differences, operation-order differences, or
  product-presence differences remain in this focused run.

Interpretation:

The systematic DD-plus-DI full-Jones h5parm/amplitude split was a
current-branch regression in gain post-processing, not an intentional
flexible-strategy behavior. The remaining differences are now in the same
family as comparison-contract work: small WSClean/PyBDSF image/catalog
residuals, text formatting, and legacy CWL output-record metadata. The next
comparison-contract work should keep h5parm structure, product presence,
operation order, source count, and primary catalog values strict. Semantic DS9
region comparison has now been added to the shared product comparator, so
future reruns will ignore harmless label-placement differences while keeping
coordinate systems, geometry, and label sets strict. Output-record comparison
now also separates metadata-shape differences from strict product-basename
drift, so future reports can keep legacy CWL metadata noise non-blocking while
still failing if an operation record points at different products. Any
image/PyBDSF numeric tolerances should still be derived from same-branch
repeatability before being accepted.

## DD Phase Plus DI Full-Jones Normalized Repeatability Envelope

A refreshed three-repeat branch repeatability run was completed on 2026-07-06
after the full-Jones normalization fix and comparison-rule cleanup. The run
used short paths (`/tmp/rfjnr`, `/tmp/rfjnw`, and `/tmp/rfjnv`) to avoid the
legacy master PyBDSF/Toil path-length failure.

The tracked compact report bundle is:

```text
docs/source/development/equivalence_runs/2026-07-06-dd-phase-plus-di-fulljones-normalized-repeatability-master-ref/
```

All six branch executions completed successfully: three `master` repeats and
three current-branch repeats. All 15 pair comparisons passed under the refined
comparison contract. Same-branch pairs had no warnings; all nine cross-branch
pairs had only two non-blocking auxiliary output-record artifact warnings for
diagnostic plot-name vocabulary and the local full-Jones h5 alias.

| Pair Group | Pairs | Passed | Max Failures | Max Warnings | Max Abs Delta | Max P99 Abs Delta | Max Residual RMS | Max Diagnostic Rel Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `base-base` | 3 | 3 | 0 | 0 | 9.537e-07 | 8.941e-08 | 3.157e-08 | 2.142e-04 |
| `current-current` | 3 | 3 | 0 | 0 | 1.431e-06 | 8.941e-08 | 2.973e-08 | 1.000e-04 |
| `base-current` | 9 | 9 | 0 | 2 | 1.431e-06 | 8.941e-08 | 3.079e-08 | 1.944e-04 |

Interpretation:

This is the strongest DD plus DI full-Jones evidence so far. After
normalization and semantic comparison cleanup, cross-branch FITS, h5parm,
text/region, catalog, and image-diagnostic differences are inside the
same-branch repeatability envelope. The remaining cross-branch warnings are
execution-record artifact naming only and do not represent scientific product
drift.

## Branch-Vs-Master Fixed-Facet Carry-Over Run

A two-cycle fixed-`facet_layout` DD phase-only branch comparison was run on
2026-07-05. Cycle 1 calibrates and images using a fixed rich-demo facet layout;
cycle 2 is calibration-only so that previous-cycle initial-solution carry-over
is isolated. The tracked compact report bundle is:

```text
docs/source/development/equivalence_runs/2026-07-05-fixed-facet-carryover-master-ref/
```

Both branch executions completed successfully and both ran `calibrate_2`.
Strict comparison failed, but the run confirmed the intended current-branch
flexible carry-over behavior.

| Scenario | Base ref | Base RC | Current RC | Result | Ops | Records | FITS | Image HDUs | Table HDUs | H5 | Text | Diagnostics | Visuals | Warnings | Failures |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `fixed-facet-carryover` | `master` | 0 | 0 | fail | 4 | 4 | 7 | 6 | 1 | 4 | 16 | 1 | 7 | 2 | 9 |

Key findings:

- Both branches read the same 5-patch fixed facet layout and run
  `calibrate_1`, `image_1`, `mosaic_1`, then calibration-only `calibrate_2`.
- In `calibrate_2`, master passes only
  `fast_initialsolutions_h5parm` from cycle 1. The current branch passes both
  `solve1_initialsolutions_h5parm` and `solve2_initialsolutions_h5parm`, using
  the previous fast and medium phase products because fixed facets prove DD
  direction compatibility.
- Image diagnostics from cycle 1 are effectively identical: source counts and
  theoretical RMS match, and displayed diagnostic deltas round to `0.000%`.
- The remaining strict image failures are small except for sparse
  `field-MFS-model-pb.fits.fz` component differences. Restored image residual
  RMS is about `1.19e-06`, close to the current robust tolerance boundary.
- The text failure is the known exact DS9 region formatting difference: master
  writes patch labels on point rows, while the current branch writes labels on
  polygon rows.

Interpretation:

This is expected to fail strict branch-vs-master comparison because the current
branch intentionally carries a compatible medium-phase seed that master does
not persist for phase-only DD cycles. Scientifically, this is the desired
flexible-strategy behavior when the facet layout is fixed. The next DD
carry-over scenario should use changing/regrouped facets and should verify that
the current branch refuses unsafe previous-cycle DD seeds.

## Fixed-Facet Repeatability Envelope

A three-repeat branch repeatability run was completed for the fixed-facet DD
carry-over scenario. The first attempt used long run/work paths and reproduced
a legacy master imaging failure in PyBDSF multiprocessing:
`OSError: AF_UNIX path too long`. The successful run used short paths
(`/tmp/rffr` and `/tmp/rffw`). Keep repeatability `--run-root`,
`--repeatability-work-root`, and base checkout/venv paths short when master
executes the CWL imaging path.

The tracked compact report bundle is:

```text
docs/source/development/equivalence_runs/2026-07-05-fixed-facet-repeatability-master-ref/
```

All six branch executions completed successfully: three `master` repeats and
three current-branch repeats. Strict pair comparison still failed for all
same-branch and cross-branch pairs, which shows that the current strict
comparison rules are tighter than normal DP3/WSClean/PyBDSF run-to-run scatter.

| Pair Group | Pairs | Passed | Max Failures | Max Warnings | Max Abs Delta | Max P99 Abs Delta | Max Residual RMS | Max Diagnostic Rel Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `base-base` | 3 | 0 | 9 | 0 | 2.330e-01 | 9.939e-06 | 3.480e-03 | 2.906e-06 |
| `current-current` | 3 | 0 | 9 | 0 | 2.347e-01 | 9.939e-06 | 3.491e-03 | 4.218e-06 |
| `base-current` | 9 | 0 | 10 | 2 | 2.348e-01 | 9.939e-06 | 3.478e-03 | 3.565e-06 |

Key findings:

- The branch-vs-master FITS residual envelope is not larger than same-branch
  scatter for the aggregate image metrics shown above.
- Same-branch failures are dominated by low-level FITS pixel drift and sparse
  `field-MFS-model-pb.fits.fz` component differences. These need
  repeatability-bounded comparison rules rather than strict allclose checks.
- Cross-branch-only differences remain for metadata/output-record summaries
  and exact DS9 region text formatting. Those should be classified separately
  from image numeric scatter.
- Image diagnostics are effectively stable within and across branches for this
  scenario; the maximum relative diagnostic delta is below `5e-06`.

Interpretation:

This repeatability envelope supports treating the fixed-facet branch-vs-master
image differences as repeatability-bounded rather than as a scientific
regression. The next comparison-rule work should derive product-specific
tolerances from same-branch scatter, keep structural contracts strict, and
separate exact text/metadata differences from numeric product differences.

## Branch-Vs-Master Changing-Facet Carry-Over Run

A two-cycle DD phase-only branch comparison with no fixed `facet_layout` was
run on 2026-07-05. Cycle 1 uses five DD calibration directions. Cycle 2 changes
to three directions and is calibration-only, isolating whether previous-cycle
DD initial solutions are reused safely. The tracked compact report bundle is:

```text
docs/source/development/equivalence_runs/2026-07-05-changing-facet-carryover-master-ref/
```

Both branch executions completed successfully. Strict comparison failed, but
the run confirmed the intended current-branch guard for regrouped/changing DD
directions.

| Scenario | Base ref | Base RC | Current RC | Result | Ops | Records | FITS | Image HDUs | Table HDUs | H5 | Text | Diagnostics | Visuals | Warnings | Failures |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `changing-facet-carryover` | `master` | 0 | 0 | fail | 4 | 4 | 7 | 6 | 1 | 4 | 16 | 1 | 7 | 2 | 10 |

Key findings:

- Both branches run `calibrate_1`, `image_1`, `mosaic_1`, then
  calibration-only `calibrate_2`.
- Cycle 2 uses three directions: `Patch_2`, `Patch_3`, and `Patch_4`.
- In `calibrate_2`, master still passes the cycle-1
  `fast_initialsolutions_h5parm` even though the cycle-1 h5parm contains extra
  directions `Patch_0` and `Patch_1`.
- In `calibrate_2`, the current branch sets all DD initial-solution slots to
  `None` and logs that the previous fast and medium h5parms were skipped
  because their directions do not match the current calibration patches.
- Cycle-1 image diagnostics are effectively identical: source counts and
  theoretical RMS match, and displayed diagnostic deltas round to `0.000%`.

Interpretation:

This run supports the current branch's flexible calibration contract. Previous
DD solutions can seed later DD solves only when direction compatibility is
proven. Fixed layouts allow compatible carry-over; regrouped/changing
directions correctly block previous-cycle DD seeds. Master-compatible
equivalence reports should label this as an intentional current-branch safety
improvement rather than a product regression.

## Branch-Vs-Master Slow-Gain Default-Like Run

A one-cycle calibration-only DD slow-gain/default-like branch comparison was
run on 2026-07-05. The master input uses the legacy
`do_slowgain_solve = True` flag, while the current-branch input uses the
explicit flexible strategy:

```python
{"dd": ["fast_phase", "medium_phase", "slow_gains", "medium_phase"], "di": []}
```

The tracked compact report bundle is:

```text
docs/source/development/equivalence_runs/2026-07-05-slow-gain-default-like-master-ref/
```

Both branch executions completed successfully. Strict comparison failed because
the final combined h5parm differs.

| Scenario | Base ref | Base RC | Current RC | Result | Ops | Records | FITS | Image HDUs | Table HDUs | H5 | Text | Diagnostics | Visuals | Warnings | Failures |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `slow-gain-default-like` | `master` | 0 | 0 | fail | 1 | 1 | 0 | 0 | 0 | 5 | 7 | 0 | 2 | 1 | 1 |

Key findings:

- Both branches execute a DD solve order equivalent to fast phase, medium
  phase, slow diagonal gains, and post-slow medium phase.
- Both branches produce standalone `slow_gains.h5parm` and
  `medium2_phases.h5parm` products.
- Master logs `CRITICAL - rapthor:combine_h5parms - ValueError: could not
  broadcast input array from shape (24,1,8,5,2) into shape (24,8,5,2)` during
  the `combine_fast_and_full_slow_h5parms` step, but the overall run still
  returns `0`.
- Master final `field-solutions.h5` contains only `sol000/phase000`, with no
  `sol000/amplitude000` soltab. The current branch final `field-solutions.h5`
  contains both `phase000` and `amplitude000`, preserving the active slow-gain
  amplitude product.

Interpretation:

This should be treated as a documented master bug or legacy limitation, not a
current-branch regression. The current branch appears to preserve the intended
slow-gain/default-like active solution state. Future master-reference reporting
should either label this difference explicitly or compare against a patched
reference if a true intended-amplitude master run is required.

## Branch-Vs-Master DI Multi-Cycle Carry-Over Run

A two-selfcal-cycle branch comparison was run on 2026-07-05 to exercise
previous-cycle DI full-Jones carry-over. Because the legacy master strategy
cannot express pure DI-only cycles, both branches use a master-compatible
sequence in each cycle: DD fast+medium phase-only calibration, DI full-Jones
calibration, and imaging. A third strategy entry acts only as the final-cycle
template. The tracked compact report bundle is:

```text
docs/source/development/equivalence_runs/2026-07-05-di-multicycle-carryover-master-ref/
```

Both branch executions completed successfully after fixing the current branch's
full-Jones initial-solution soltab mapping.

| Scenario | Base ref | Base RC | Current RC | Result | Ops | Records | FITS | Image HDUs | Table HDUs | H5 | Text | Diagnostics | Visuals | Warnings | Failures |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `di-multicycle-carryover` | `master` | 0 | 0 | fail | 10 | 10 | 14 | 12 | 2 | 6 | 19 | 2 | 10 | 4 | 123 |

Key findings:

- An initial `v3` run exposed a current-branch bug: cycle-2 `calibrate_di_2`
  passed the previous `fulljones-solutions.h5` as
  `solve1.initialsolutions.h5parm` with
  `solve1.initialsolutions.soltab=[phase000]`, and DP3 failed.
- The current branch now uses
  `solve1.initialsolutions.soltab=[amplitude000,phase000]` for full-Jones
  initial solutions. Focused command-builder and operation tests cover this.
- The `v4` rerun completed on both branches and reached `calibrate_2`,
  `predict_di_2`, `calibrate_di_2`, `image_2`, and `mosaic_2`.
- Current `calibrate_di_2` seeds the full-Jones solve from cycle 1 with
  `[amplitude000,phase000]`. Current `image_2` applies the cycle-2
  `fulljones-solutions.h5` via `applycal.steps=[fulljones]`.
- Cycle-1 diagnostics remain close: source counts and theoretical RMS match,
  and RMS diagnostics differ by about `0.224%`.
- Cycle-2 source counts still match, but RMS diagnostics differ by about
  `9-10%`. The largest h5parm differences are in cycle-2 DD/full-Jones
  products, including `field-solutions-fast-phase.h5` and
  `fulljones-solutions.h5`.

Interpretation:

The current branch now handles DI full-Jones carry-over with the correct h5parm
soltab contract and can apply the resulting cycle-2 full-Jones solutions during
imaging. The remaining cycle-2 divergence is expected to be dominated by the
same changing-facet DD seed behavior observed in the dedicated changing-facet
scenario: master carries a previous DD fast-phase seed across changed
directions, while the current branch skips unsafe DD seeds. This scenario should
therefore be treated as a mixed result: the DI full-Jones carry-over bug is
fixed, but branch-vs-master product tolerances cannot be tuned from this run
without the planned repeatability envelope.

## Branch-Vs-Master DI/DD Mode-Boundary Runs

Two fixed-facet two-cycle branch comparisons were run on 2026-07-05 to exercise
the boundaries between DD phase-only calibration and DI full-Jones calibration.
The tracked compact report bundles are:

```text
docs/source/development/equivalence_runs/2026-07-05-di-then-dd-mode-boundary-master-ref/
docs/source/development/equivalence_runs/2026-07-05-dd-then-di-mode-boundary-master-ref/
```

Both branches completed both scenarios successfully.

| Scenario | Base ref | Base RC | Current RC | Result | Ops | Records | FITS | Image HDUs | Table HDUs | H5 | Text | Diagnostics | Visuals | Warnings | Failures |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `di-then-dd-mode-boundary` | `master` | 0 | 0 | fail | 8 | 8 | 14 | 12 | 2 | 5 | 19 | 2 | 10 | 3 | 117 |
| `dd-then-di-mode-boundary` | `master` | 0 | 0 | fail | 8 | 8 | 14 | 12 | 2 | 5 | 19 | 2 | 10 | 3 | 64 |

Key findings:

- In the DI-to-DD scenario, cycle 1 runs DD plus DI full-Jones and cycle 2
  returns to DD-only calibration and imaging. Master passes the cycle-1
  `fulljones-solutions.h5` into `image_2`, with
  `prepare_data_applycal_steps = [fulljones]`. The current branch intentionally
  leaves `fulljones_h5parm` unset for `image_2` and logs that the cycle-1
  full-Jones h5parm is ignored for cycle-2 calibration because it was produced
  in cycle 1.
- In the DD-to-DI scenario, cycle 1 is DD-only and cycle 2 adds DI full-Jones
  after DD. Both branches pass the cycle-2 `fulljones-solutions.h5` into
  `image_2`, with `prepare_data_applycal_steps = [fulljones]`.
- In both fixed-facet scenarios, master passes only the previous fast-phase DD
  seed into `calibrate_2`, while the current branch passes compatible fast and
  medium phase seeds. That difference is intentional under the flexible
  strategy contract but remains a systematic branch difference for strict
  product comparisons.
- The DI-to-DD cycle-2 diagnostics diverge in the expected boundary-sensitive
  place: source counts match, but min-RMS diagnostics differ by about
  `10-12%`. The DD-to-DI run shows cycle-1 diagnostics essentially identical,
  then cycle-2 RMS diagnostics differing by about `11%`.

Interpretation:

The mode-boundary matrix is now complete enough to treat this as a policy
decision rather than add more ad hoc branch-vs-master runs. The adopted
flexible-strategy contract is to avoid silent carry-over after a new
calibration step: a previous full-Jones product may not be applied during later
imaging unless it was part of that cycle's calibration state. Master's lower
cycle-2 RMS in the DI-to-DD scenario is therefore interpreted as a useful but
implicit stale correction, not as behavior to copy silently. Do not tune product
tolerances from these two runs until the planned repeatability envelope shows
same-branch scatter for DP3/WSClean/PyBDSF outputs.

## Historical Passing Scenarios

The required local saved-reference gate passed for:

| Scenario | Result | Notes |
| --- | --- | --- |
| `di_only_calibration` | pass | Existing DI fast-phase reference |
| `dd_only_calibration` | pass | Exposed and fixed diagnostic plot-output bookkeeping |
| `di_then_dd_calibration` | pass | Exposed and fixed DI medium-phase model handoff |
| `dd_then_di_calibration` | pass | Reverse DI/DD handoff matched |
| `di_full_jones_calibration` | pass | DI full-Jones plotting and product scopes matched |
| `dd_slow_gain_calibration` | pass | Exposed and fixed slow-only legacy output aliasing |
| `normalization` | pass | Used normalization-specific Measurement Set |
| `peeling` | pass | Bright-source and outlier peeling scopes matched |
| `full_stokes_clean_disabled` | pass | Full-Stokes clean-disabled imaging matched |
| `image_cube` | pass | Image cube products and metadata matched |
| `restart` | pass | Restart and persisted output-record reuse matched |

The live CWL-vs-Prefect smoke gate also passed for the existing DI fast-phase
integration scenario against the preserved legacy checkout.

## Deferred Target-Environment Checks

These were intentionally deferred until after the migration cutover:

- `mpi_wsclean`: run with the intended MPI/WSClean deployment stack.
- Slurm/external-Dask: run inside a representative Slurm allocation.

These optional paths were deferred from the required local gate:

- `hybrid_screens`: not used by the current target workflow and dependent on
  DP3 IDGCal Python bindings.
- `shared_facet_rw`: too flaky in the available WSClean environments; revisit
  when the intended environment is reliable.

## Outcome

The public `rapthor.process.run()` route now executes the Prefect process flow.
The in-tree CWL runner, workflow/parset files, active CWL comparison tests, and
`cwltool` dependency have been removed. The remaining regression strategy is
the Prefect/Dask operation, flow, process, demo, and focused integration test
suite, with this report retained as the migration parity record.
