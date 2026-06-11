# Rapthor Prefect/Dask Migration Plan

Status snapshot: 2026-06-11.

## Goal

Make Prefect/Dask the production execution path for Rapthor while preserving
the existing parset/strategy contract, operation ordering, restart state,
output records, product filenames/locations, finalizer-visible field state, and
Slurm/MPI safety. CWL should remain only as a reference mechanism until
equivalence is proven, then as static fixtures or compatibility helpers.

## Current Position

Most implementation work is complete.

- `rapthor/execution/` contains the new execution layer: config, Dask task
  runner construction, resource/capability checks, command builders, shell
  execution, command logs, output records, artifacts, work-directory helpers,
  Slurm helpers, and equivalence utilities.
- Prefect flows exist for concatenate, mosaic, predict, image, calibration, and
  top-level process orchestration.
- Operation adapters for concatenate, mosaic, predict, image, image-initial,
  image-normalize, and calibration execute Prefect flows and produce
  finalizer-compatible output records.
- `rapthor.process.run()` is still the legacy CWL baseline. The current
  side-by-side Prefect entry point is
  `rapthor.execution.flows.process.process_flow()`.
- Saved-reference equivalence is in place. Eleven CWL reference scenarios from
  legacy commit `4cfd2abe2fe815724e3f1c390d789eea249becef` pass comparison
  against the current Prefect path.
- Demo and observability support is in place: persistent Prefect dashboard
  support, unique run directories, local/external Dask dashboard support, Dask
  performance reports, streamed external-command logs, Python logs in Prefect,
  plot/FITS artifacts, command timing artifacts, and richer synthetic demo data.
  Several manual demo runs have been completed successfully and look good for
  the current migration stage.
- Recent correctness fixes include cycle-aware h5parm provenance/filtering so
  stale solutions from previous cycles are not applied accidentally.
- CI has been adjusted and verified passing: tests that start a Prefect test
  server are marked and run serially, while the remaining non-integration tests
  still use `pytest-xdist -n auto`.
- CI now pins DP3 to the dev-container-tested commit `18e793a4`.
- CWL runtime code, package data, Toil, StreamFlow, and cwltool are still
  present and must stay until the final equivalence gate and public route
  cutover are complete.

## Evidence Already Available

- Execution-layer unit and flow tests cover command construction, payloads,
  output records, restart/reuse, failure handling, artifacts, logging, resource
  validation, task-runner selection, work directories, and equivalence helpers.
- Mocked process-flow tests compare the Prefect lifecycle with the legacy
  process lifecycle for final-only, selfcal, convergence/divergence/failure,
  repeated final cycles, normalization, full-Stokes/cube flags, concatenation,
  calibration strategy hand-offs, validation failures, and artifact publication.
- `tests/execution/fixtures/equivalence_gate_scenarios.json` defines the
  saved-reference scenario matrix.
- `scripts/capture_cwl_reference_artifacts.py` can populate CWL reference
  artifacts from a preserved pre-cutover checkout.
- `tests/integration/test_saved_cwl_equivalence.py` compares fresh Prefect runs
  against saved CWL artifacts when explicitly enabled.
- `tests/integration/test_live_cwl_equivalence.py` can run a live CWL checkout
  and the current Prefect flow side by side for a smoke scenario.
- `EQUIVALENCE_REPORT.md` records the current equivalence method, passing
  scenarios, known blockers, and commands.

Passing saved-reference scenarios:

- `di_only_calibration`
- `dd_only_calibration`
- `di_then_dd_calibration`
- `dd_then_di_calibration`
- `di_full_jones_calibration`
- `dd_slow_gain_calibration`
- `normalization`
- `peeling`
- `full_stokes_clean_disabled`
- `image_cube`
- `restart`

Still missing target-environment proof:

- `mpi_wsclean`: should run in the intended MPI/WSClean deployment
  environment.
- Slurm/external-Dask: should run inside a representative Slurm allocation.

Deferred optional scenarios:

- `hybrid_screens` is excluded from the required gate for now because it is not
  used by the current target workflow. Capturing it would require IDG Python
  bindings for DP3 IDGCal, so leave the known-good DP3-pinned image alone unless
  this path becomes relevant again.
- `shared_facet_rw` is excluded from the required gate for now because WSClean's
  shared-facet read/write flags have been too flaky in available environments.

## Remaining Work

### 1. Finish Target-Environment Equivalence

This is the main blocker before public cutover.

- Run `tests/integration/test_saved_cwl_equivalence.py` with
  `RAPTHOR_RUN_SAVED_CWL_EQUIVALENCE=1` and `RAPTHOR_CWL_REFERENCE_ROOT`
  pointing at the directory that contains the saved CWL output subdirectories
  for every required, non-deferred scenario. This reruns the current Prefect
  candidates and compares them against those saved CWL references.
- Run the live CWL-vs-Prefect smoke gate against the preserved legacy checkout.
- Run `mpi_wsclean` with the intended MPI/WSClean stack.
- Run the Slurm/external-Dask hook inside a real Slurm allocation.
- Record the passing source data, artifact root, strategy files, commit SHAs,
  tool versions, commands, and test output.
- Fix real differences. Document only intentional, user-invisible differences.

### 2. Refresh Real External-Tool Coverage

The unit and mocked flow tests are broad, but the production confidence still
depends on real radio-astronomy tools and representative data.

- Exercise real DP3 prediction, DDECal, LoSoTo, WSClean, EveryBeam, PyBDSF,
  cfitsio/fpack, and mosaic paths in the dev container or staging environment.
- Leave IDG/screen-generation coverage deferred unless `hybrid_screens` becomes
  a supported target path again.

### 3. Cut Over The Public Route

Only start this once target-environment equivalence has passed.

- Route the CLI-compatible `rapthor.process.run()` path through
  `process_flow()`.
- Keep no public `execution_backend` selector.
- Update operation/process tests so Prefect/Dask is the expected production
  route.
- Keep saved CWL artifacts and CWL-derived command/output fixtures only as
  parity evidence.
- Update user-facing docs that still describe Toil/CWL as the normal execution
  mechanism.

### 4. Remove CWL Production Runtime

Only after public route cutover.

- Remove Toil, StreamFlow, and cwltool dependencies if no longer needed.
- Remove `rapthor/lib/cwlrunner.py`, CWL-only operation plumbing, and obsolete
  CWL runner tests.
- Remove CWL workflow templates from production package data once needed static
  fixtures are preserved elsewhere.
- Update package-data, lint, format, tox, CI, and release jobs that currently
  know about CWL files.
- Keep or rename CWL-shaped output helpers only if finalizers still depend on
  those record shapes during the first Prefect/Dask release.

### 5. Final Documentation And Release Notes

- Update README and Sphinx docs so Prefect/Dask is documented as the supported
  execution path.
- Document local, external-Dask, Slurm, MPI WSClean, dashboard, artifact, and
  restart behaviour.
- Record the equivalence evidence and known limitations in release notes.
- Ensure generated demo data and large reference artifacts stay out of version
  control.

### 6. Post-Cutover Refactor And Deduplication

Only after equivalence, public route cutover, and CWL runtime removal.

- Review the execution flows, operation adapters, command builders, output
  handling, artifact publication, and tests for duplication introduced during
  the side-by-side migration.
- Consolidate repeated patterns where a small shared helper improves clarity
  without hiding operation-specific behaviour.
- Remove temporary migration scaffolding that is no longer needed once CWL is no
  longer a production backend.
- Keep this as a cleanup pass, not a behaviour-changing prerequisite for the
  equivalence gate.

## Deferred Performance Follow-Up

Dask optimization is explicitly deferred until after equivalence and public
route cutover. Later work can use the Dask dashboard and performance reports to
review task granularity, chunk sizing, worker/resource tuning, and whether any
operation should expose finer-grained parallelism. These are not merge blockers
unless a performance issue prevents production use.

## Immediate Next Actions

1. Run the saved-CWL equivalence test with `RAPTHOR_CWL_REFERENCE_ROOT` pointing
   at the populated reference directory for the required, non-deferred
   scenarios.
2. Run the live CWL-vs-Prefect smoke gate.
3. Run `mpi_wsclean` and Slurm/external-Dask hooks in the deployment-like
   environment.
4. If all gates pass, switch `rapthor.process.run()` to `process_flow()`.
5. Remove CWL production runtime and update docs, packaging, and CI.
6. Do a post-cutover refactor pass to clean up migration scaffolding and reduce
   duplication.

## Useful Commands

Serial Prefect-server lane:

```bash
python3 -m pytest -m "not integration and prefect" -k "not test_field.py" tests -q --tb=short
```

Parallel non-Prefect lane:

```bash
python3 -m pytest -m "not integration and not prefect" -n auto -k "not test_field.py" tests -q --tb=short
```

Saved-reference gate:

```bash
RAPTHOR_RUN_SAVED_CWL_EQUIVALENCE=1 \
RAPTHOR_CWL_REFERENCE_ROOT=/path/to/references \
python3 -m pytest tests/integration/test_saved_cwl_equivalence.py -q --tb=short
```

Live CWL-vs-Prefect smoke gate:

```bash
RAPTHOR_RUN_LIVE_CWL_EQUIVALENCE=1 \
RAPTHOR_LEGACY_CWL_REPO=/path/to/pre-cutover-checkout \
python3 -m pytest tests/integration/test_live_cwl_equivalence.py -q --tb=short
```

Slurm hook:

```bash
RAPTHOR_RUN_SLURM_INTEGRATION=1 \
python3 -m pytest tests/integration/test_slurm_execution.py -q --tb=short
```

Rich local demo:

```bash
scripts/dev/generate-prefect-demo-data.py --force
scripts/dev/run-rapthor-prefect-demo.py \
  --task-runner local_dask \
  --dask-dashboard-address :8787 \
  --dask-performance-report \
  examples/generated/prefect_demo_rich/prefect_demo_rich.parset
```

## Merge Criteria

The migration is ready to merge when:

- all supported public operation and process paths run through Prefect/Dask
- the target-environment equivalence gate passes for the supported scenario
  matrix
- restart/reset behaviour works with Prefect-produced operation state
- runtime, resource, filesystem, command-log, artifact, Dask, MPI, and Slurm
  checks pass
- CI is stable with Prefect tests isolated from high xdist fan-out
- docs describe Prefect/Dask as the supported production execution path
- CWL artifacts that remain in the tree are static reference fixtures or
  compatibility helpers, not production execution machinery
