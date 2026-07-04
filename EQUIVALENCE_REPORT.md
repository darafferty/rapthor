# Rapthor CWL-to-Prefect Equivalence Report

Generated: 2026-06-11

Archived for post-cutover cleanup: 2026-06-12

Latest on-disk report scan: 2026-07-04

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
