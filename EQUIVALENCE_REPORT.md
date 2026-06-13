# Rapthor CWL-to-Prefect Equivalence Report

Generated: 2026-06-11

Archived for post-cutover cleanup: 2026-06-12

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

## Passing Scenarios

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
