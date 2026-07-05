# Rapthor CWL-to-Prefect Equivalence Report

Generated: 2026-06-11

Archived for post-cutover cleanup: 2026-06-12

Latest on-disk report scan: 2026-07-05

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

## Current Strengthened Saved-Reference Run

The current strengthened saved-reference run is:

```text
runs/equivalence-strengthened-20260704-codex-green/equivalence-report.json
```

It was generated in the development container on 2026-07-04 14:55:09 BST and
passed. The matching Markdown report is:

```text
runs/equivalence-strengthened-20260704-codex-green/equivalence-report.md
```

References that encode older scientific contracts are skipped by default, but
can still be run explicitly with `--include-stale-references`.

| Scenario | Result | Ops | Records | FITS | Image HDUs | Table HDUs | H5 | Text |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `dd_only_calibration` | pass | 4 | 4 | 6 | 5 | 1 | 3 | 10 |
| `di_full_jones_calibration` | pass | 5 | 5 | 6 | 5 | 1 | 1 | 9 |
| `di_only_calibration` | pass | 5 | 5 | 6 | 5 | 1 | 2 | 9 |
| `full_stokes_clean_disabled` | pass | 4 | 4 | 9 | 8 | 1 | 3 | 8 |
| `image_cube` | pass | 4 | 4 | 7 | 6 | 1 | 3 | 12 |
| `normalization` | pass | 5 | 5 | 7 | 6 | 1 | 4 | 12 |
| `peeling` | pass | 4 | 4 | 6 | 5 | 1 | 3 | 11 |
| `restart` | pass | 4 | 4 | 6 | 5 | 1 | 3 | 10 |

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
- per-plane residual metrics for cubes and Stokes products

The table below shows the worst FITS image product in each scenario by maximum
absolute residual.

| Scenario | FITS products | Image HDUs | Table HDUs | Worst image product | Max abs delta | P99 abs delta | Residual RMS | RMS / Ref MAD |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| `dd_only_calibration` | 6 | 5 | 1 | `field-MFS-image-pb.fits` | 2.128e-10 | 2.365e-11 | 8.345e-12 | 1.972e-07 |
| `di_full_jones_calibration` | 6 | 5 | 1 | `field-MFS-dirty.fits` | 2.086e-06 | 2.980e-07 | 1.036e-07 | 1.962e-07 |
| `di_only_calibration` | 6 | 5 | 1 | `field-MFS-dirty.fits` | 1.510e-10 | 2.910e-11 | 9.418e-12 | 2.211e-07 |
| `full_stokes_clean_disabled` | 9 | 8 | 1 | `field-MFS-I-image-pb.fits` | 2.219e-10 | 2.410e-11 | 8.460e-12 | 1.999e-07 |
| `image_cube` | 7 | 6 | 1 | `sector_1_I_freq_cube.fits` | 5.093e-10 | 4.547e-11 | 1.618e-11 | 3.736e-07 |
| `normalization` | 7 | 6 | 1 | `sector_1_I_freq_cube.fits` | 6.676e-06 | 6.706e-07 | 2.002e-07 | 3.921e-07 |
| `peeling` | 6 | 5 | 1 | `field-MFS-image-pb.fits` | 2.187e-06 | 2.384e-07 | 8.266e-08 | 1.660e-07 |
| `restart` | 6 | 5 | 1 | `field-MFS-image-pb.fits` | 2.219e-10 | 2.910e-11 | 1.083e-11 | 2.558e-07 |

HDF5 products compare dataset names and shapes. Numeric datasets use
`np.allclose(..., atol=1e-6, rtol=1e-3, equal_nan=True)`, while non-numeric
datasets use exact array equality. All HDF5 checks passed in the current run.

Text-like products compare sky-model `lines` and `patches`, beam tables with
`atol = 1e-6` and `rtol = 1e-2`, and all other text and region files exactly.
All text-like product checks passed in the current run.

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
