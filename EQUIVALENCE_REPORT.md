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

## Current On-Disk Saved-Reference Run

The current on-disk default saved-reference run is:

```text
/tmp/rapthor-equivalence-codex-3/equivalence-report.json
```

It was generated in the development container on 2026-07-04 06:32:31 and
passed. References that encode older scientific contracts are skipped by
default, but can still be run explicitly with `--include-stale-references`.

| Scenario | Result | Ops | Records | FITS | H5 | Text |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `dd_only_calibration` | pass | 4 | 4 | 6 | 3 | 10 |
| `di_full_jones_calibration` | pass | 5 | 5 | 6 | 1 | 9 |
| `di_only_calibration` | pass | 5 | 5 | 6 | 2 | 9 |
| `full_stokes_clean_disabled` | pass | 4 | 4 | 9 | 3 | 8 |
| `image_cube` | pass | 4 | 4 | 7 | 3 | 12 |
| `normalization` | pass | 5 | 5 | 7 | 4 | 12 |
| `peeling` | pass | 4 | 4 | 6 | 3 | 11 |
| `restart` | pass | 4 | 4 | 6 | 3 | 10 |

## Product Statistic Checks

The JSON report stores pass/fail results and product counts. The tables below
were derived from the same saved reference artifacts and current run products
on disk to show the data-product statistics used by the checks.

FITS image products compare finite pixel count plus `mean`, `std`, `rms`,
`min`, and `max`, using `atol = 1e-6` and `rtol = 1e-3`. A max tolerance ratio
below 1.0 means the worst observed statistic was inside tolerance.

| Scenario | FITS products | Image HDUs | Table HDUs | Worst product | Worst stat | Max abs delta | Max tolerance ratio |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: |
| `dd_only_calibration` | 6 | 5 | 1 | `field-MFS-image.fits` | `max` | 2.910e-11 | 2.432e-05 |
| `di_full_jones_calibration` | 6 | 5 | 1 | `field-MFS-image.fits` | `min` | 2.384e-07 | 2.510e-04 |
| `di_only_calibration` | 6 | 5 | 1 | `field-MFS-image.fits` | `max` | 2.910e-11 | 2.414e-05 |
| `full_stokes_clean_disabled` | 9 | 8 | 1 | `field-MFS-I-image.fits` | `max` | 1.455e-11 | 1.216e-05 |
| `image_cube` | 7 | 6 | 1 | `sector_1_I_freq_cube.fits` | `max` | 8.731e-11 | 7.276e-05 |
| `normalization` | 7 | 6 | 1 | `field-MFS-dirty.fits` | `mean` | 1.318e-09 | 7.208e-04 |
| `peeling` | 6 | 5 | 1 | `field-MFS-dirty.fits` | `max` | 1.788e-07 | 1.797e-04 |
| `restart` | 6 | 5 | 1 | `field-MFS-dirty.fits` | `min` | 1.455e-11 | 1.335e-05 |

HDF5 products compare dataset names and shapes. Numeric datasets use
`np.allclose(..., atol=1e-6, rtol=1e-3, equal_nan=True)`, while non-numeric
datasets use exact array equality.

| Scenario | H5 files | Numeric datasets | Non-numeric datasets | Worst file | Worst dataset | Max abs delta | Max tolerance ratio |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: |
| `dd_only_calibration` | 3 | 0 | 3 | n/a | n/a | 0.000e+00 | 0.000e+00 |
| `di_full_jones_calibration` | 1 | 0 | 1 | n/a | n/a | 0.000e+00 | 0.000e+00 |
| `di_only_calibration` | 2 | 0 | 2 | n/a | n/a | 0.000e+00 | 0.000e+00 |
| `full_stokes_clean_disabled` | 3 | 0 | 3 | n/a | n/a | 0.000e+00 | 0.000e+00 |
| `image_cube` | 3 | 0 | 3 | n/a | n/a | 0.000e+00 | 0.000e+00 |
| `normalization` | 4 | 1 | 3 | n/a | n/a | 0.000e+00 | 0.000e+00 |
| `peeling` | 3 | 0 | 3 | n/a | n/a | 0.000e+00 | 0.000e+00 |
| `restart` | 3 | 0 | 3 | n/a | n/a | 0.000e+00 | 0.000e+00 |

Text-like products compare sky-model `lines` and `patches`, beam tables with
`atol = 1e-6` and `rtol = 1e-2`, and all other text and region files exactly.

| Scenario | Sky-model summaries | Beam tables | Exact text files | Region files |
| --- | ---: | ---: | ---: | ---: |
| `dd_only_calibration` | 8 | 0 | 0 | 2 |
| `di_full_jones_calibration` | 8 | 0 | 0 | 1 |
| `di_only_calibration` | 8 | 0 | 0 | 1 |
| `full_stokes_clean_disabled` | 6 | 0 | 0 | 2 |
| `image_cube` | 8 | 1 | 1 | 2 |
| `normalization` | 8 | 1 | 1 | 2 |
| `peeling` | 9 | 0 | 0 | 2 |
| `restart` | 8 | 0 | 0 | 2 |

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
