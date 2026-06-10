# Rapthor Prefect and Dask Migration Plan

This is the active status snapshot for migrating Rapthor from Toil/CWL
execution to Python orchestration with Prefect and Dask.

## Goal

Replace the CWL workflow runner path with Prefect/Dask while preserving:

- the existing parset and strategy contract
- operation ordering and top-level process semantics
- finalizer-visible `Field`, `Observation`, and `Sector` state
- output filenames, output-record shapes, and product locations
- restart/reset behaviour through `.done` and `.outputs.json`
- local execution, Slurm/external-Dask execution, and MPI WSClean safety

The migration branch should merge with one production execution path:
Prefect/Dask. CWL remains only as a branch-local reference until equivalence is
proven.

## Current State

Most of the migration implementation is already in place.

- `rapthor/execution/` now contains the execution layer: configuration,
  capability preflight, command builders, serializable payload checks, output
  helpers, runtime environment handling, shell execution, command logging,
  resource validation, task-runner construction, task-local work directories,
  Slurm helpers, and CWL-to-Prefect equivalence helpers.
- Operation-level Prefect flows exist for concatenate, mosaic, predict, image,
  calibration, and process orchestration under `rapthor/execution/flows/`.
- The operation adapters for `Concatenate`, `Mosaic`, `Predict`, `Image`,
  `ImageInitial`, `ImageNormalize`, and `Calibrate` now execute Prefect flows and
  return finalizer-compatible output records.
- `rapthor.process.run()` is intentionally still the legacy top-level baseline.
  The side-by-side Prefect entry point is `rapthor.execution.flows.process.process_flow()`.
  This keeps a comparison route available until the final equivalence gate has
  passed in the target environment.
- Eleven local saved CWL references have been captured from legacy commit
  `4cfd2abe2fe815724e3f1c390d789eea249becef` and pass saved-reference
  comparison against the current Prefect execution path.
- CWL runner code, CWL package data, Toil, StreamFlow, and cwltool dependencies
  are still present. Do not remove them until the equivalence gate and public
  route cutover are complete.

## Already Done

### Execution Foundation

- Added `ExecutionConfig` with `local_dask`, `external_dask`, and `sync` task
  runner modes.
- Added Dask scheduler discovery from parset settings and `DASK_SCHEDULER`.
- Added Prefect task-runner wiring so flows actually use the configured runner.
- Added shell execution helpers with streamed output and backend-neutral command
  logs at `logs/commands.jsonl`.
- Added finalizer-compatible file and directory output helpers.
- Added serializable payload boundaries so worker tasks do not receive live
  Rapthor domain objects.
- Added task-local working-directory helpers and cleanup semantics.
- Added centralized resource validation for threads, memory, local Dask worker
  capacity, Slurm allocation limits, and MPI exclusivity.
- Added preflight checks for unsupported runtime, container, feature, and
  resource combinations.

### Operation Migration

- Concatenate now runs through the Prefect flow and has command, output,
  restart, failure, and operation-adapter coverage.
- Mosaic now runs through the Prefect flow and has operation coverage for
  output selection, finalizer state, compressed outputs, skip-processing, and
  helper scripts using small FITS fixtures.
- Predict now runs DI and DD paths through the Prefect flow, including sector
  model add/subtract scripts, operation restart, failure handling, and
  finalizer-visible field updates.
- Image now covers no-DDE, facet, screen, compressed, filtered-model,
  clean-disabled, full-Stokes, image-cube, normalization, bright-source peeling,
  MPI WSClean, previous-mask, restart, failure, and operation-adapter paths.
- Calibration now covers DI full-Jones, DI scalar phase, DD fast/medium phase,
  DD slow/source-adjusted products, DI pre-apply before DD, image-based
  prediction, IDG/screen generation, restart, failure, and operation-adapter
  paths.

### Top-Level Process Work

- Added side-by-side Prefect process orchestration with `process_flow()`.
- Added mocked equivalence coverage against the legacy process lifecycle for
  final-only imaging, selfcal convergence/divergence/failure, repeated final
  cycles, concatenation before strategy selection, calibration strategy
  hand-offs, normalization, final full-Stokes image-cube flags, and validation
  failures.
- Kept `process.run()` on the legacy route as the current comparison baseline.

### Runtime And Slurm

- Added production and development Slurm launch templates under `scripts/`.
- Added external-Dask scheduler validation and Slurm environment mapping.
- Added local-Dask capacity checks and MPI WSClean resource validation.
- Added a target-environment Slurm integration hook in
  `tests/integration/test_slurm_execution.py`, skipped unless
  `RAPTHOR_RUN_SLURM_INTEGRATION=1` is set inside a Slurm allocation.
- Documented the selected Slurm/external-Dask mode in `docs/source/running.rst`.

### Tests And Fixtures

- Added execution-layer unit and flow tests under `tests/execution/`.
- Added CWL-derived command and output fixtures for parity checks.
- Added a machine-readable supported merge feature matrix at
  `tests/execution/fixtures/supported_merge_feature_matrix.json`.
- Added a CWL-to-Prefect equivalence harness in `rapthor/execution/equivalence.py`.
- Added an equivalence-gate scenario manifest at
  `tests/execution/fixtures/equivalence_gate_scenarios.json`.
- Added a saved-CWL reference artifact contract: each equivalence scenario now
  declares `cwl_reference_artifact_dir`, and staging can validate
  `$RAPTHOR_CWL_REFERENCE_ROOT/<scenario-id>/` before running comparisons.
- Added saved-reference equivalence runner helpers so staging can compare the
  full scenario manifest against fresh Prefect candidate runs with
  `compare_saved_reference_equivalence_manifest()`.
- Added a scenario parset materializer for the saved-reference gate. It sets the
  candidate `dir_working`, scratch directories, common template inputs from
  `RAPTHOR_EQUIVALENCE_INPUT_MS`, `RAPTHOR_EQUIVALENCE_INPUT_SKYMODEL`,
  `RAPTHOR_EQUIVALENCE_APPARENT_SKYMODEL`, and optional per-scenario
  `parset_overrides`.
- Added scenario-specific strategy fixtures under
  `tests/execution/fixtures/equivalence_strategies/` and wired them into the
  equivalence scenario manifest.
- Added an opt-in saved-CWL regression integration test at
  `tests/integration/test_saved_cwl_equivalence.py`. It validates saved CWL
  artifacts, runs the current Prefect candidate, and compares normalized outputs.
- Added an opt-in live CWL-vs-Prefect integration test at
  `tests/integration/test_live_cwl_equivalence.py` that reuses the existing DI
  fast-phase integration fixture, runs a legacy checkout and the current Prefect
  flow, and compares their working-directory outputs.
- Added `scripts/capture_cwl_reference_artifacts.py` to populate saved CWL
  artifacts by running scenarios through a separate pre-cutover checkout.
- Captured and verified local saved references for `di_only_calibration`,
  `dd_only_calibration`, `di_then_dd_calibration`, `dd_then_di_calibration`,
  `di_full_jones_calibration`, `dd_slow_gain_calibration`, `normalization`,
  `peeling`, `full_stokes_clean_disabled`, `image_cube`, and `restart`.
- Fixed the bare `do_slowgain_solve` CWL `when:` expressions in
  `calibrate_pipeline.cwl`; the cached legacy reference checkout was patched the
  same way for `hybrid_screens` capture attempts.
- Modernized integration command-log helpers so assertions do not depend on Toil
  log filenames.
- Replaced several placeholder script tests with small real Measurement Set or
  FITS-backed coverage.

## Remaining Work

### 1. Finish The Target-Environment Equivalence Gate

This is the main blocker before public cutover. Local saved-reference coverage
now proves eleven scenarios; the remaining work is to close the blocked local
cases and run the deployment-specific cases.

- Resolve the `hybrid_screens` reference blocker in an environment where DP3 can
  import the Python `idg` module. The earlier CWL `do_slowgain_solve`
  expression issue is fixed.
- Resolve the `shared_facet_rw` reference blocker in a WSClean environment that
  supports `-shared-facet-reads` and `-shared-facet-writes`.
- Run the full available saved-reference scenario manifest after those
  references are captured.
- Run `mpi_wsclean` and the selected Slurm/external-Dask cases in the target
  environment with representative data.
- Compare operation order, `.done` markers, `.outputs.json` shape, product
  basenames, FITS dimensions/statistics, h5parm solset/soltab structure, sky
  model counts, region files, reports, logs, and finalizer-visible state.
- Fix real differences or document intentional user-invisible differences with
  tests.
- Record the passing local and staging baselines before starting the route
  cutover.

Important caveat: because operation adapters on this branch already execute
Prefect flows, the CWL side of the final proof must come from a preserved
pre-cutover CWL checkout and then be saved as reference artifacts.

### 2. Refresh Real External-Tool Integration Coverage

Some gaps are intentionally waiting for a container or staging environment with
the radio astronomy toolchain available.

- Concatenate: multi-input DP3/TAQL integration.
- Predict: DP3-produced DI/DD model-data products.
- Image: small real runs through WSClean, EveryBeam/IDG where applicable,
  PyBDSF, full-Stokes, cubes, normalization, facets, screens, and MPI.
- Calibrate: real DP3/LoSoTo/IDG/h5parm paths for DI, DD, image-based
  prediction, source adjustment, and screen generation.
- Mosaic: a real image-to-mosaic path from Image-flow products.
- Restart/failure: at least one injected external-tool failure path for the
  supported matrix, if practical in staging.

### 3. Cut Over The Public Route

After the equivalence gate passes:

- Route the CLI-compatible `rapthor.process.run()` path through the Prefect
  top-level process flow.
- Remove branch-internal mixed-backend production paths.
- Keep no public `execution_backend` selector.
- Update operation and process tests so Prefect/Dask is the expected production
  route.
- Keep useful CWL-derived artifacts only as static fixtures.

### 4. Remove CWL Production Code

Only after public cutover:

- Remove Toil, StreamFlow, and cwltool dependencies if no longer needed.
- Remove `rapthor/lib/cwlrunner.py`, CWL-only operation plumbing, and unused CWL
  runner tests.
- Remove `rapthor/pipeline/parsets/**` and `rapthor/pipeline/steps/**` from
  packaged production data once static parity fixtures are preserved elsewhere.
- Update package data and lint/test target lists.

Keep `rapthor/lib/cwl.py` or specific CWL-shaped record helpers only if
finalizers still need those structures during the first Prefect/Dask release.
If retained, rename or document them as generic output-record helpers rather
than a CWL runtime dependency.

### 5. Final Documentation, Packaging, And CI

- Update README, Sphinx docs, parset documentation, operation docs, and runtime
  instructions so Prefect/Dask is described as the supported execution path.
- Update installation, tox, CI, and packaging metadata after removing CWL
  dependencies and package data.
- Document the target-environment Slurm validation process and expected skips or
  xfails.
- Run the final non-integration suite, focused integration suite, equivalence
  gate, lint checks, and packaging checks before merging.

## Immediate Next Actions

1. Capture `hybrid_screens` in an IDG-capable DP3 environment.
2. Capture `shared_facet_rw` in an environment where WSClean supports the
   shared-facet read/write flags.
3. Run the saved-reference regression over the full artifact root:
   `RAPTHOR_RUN_SAVED_CWL_EQUIVALENCE=1 RAPTHOR_CWL_REFERENCE_ROOT=/path/to/references python3 -m pytest tests/integration/test_saved_cwl_equivalence.py -q --tb=short`.
4. Run `mpi_wsclean` in the intended MPI/WSClean deployment environment.
5. Run the Slurm hook inside a real Slurm allocation:
   `RAPTHOR_RUN_SLURM_INTEGRATION=1 python3 -m pytest tests/integration/test_slurm_execution.py`.
6. If equivalence passes, switch `rapthor.process.run()` to `process_flow()`.
7. Remove CWL production code and dependencies, then update docs, packaging, and
   CI.

## Suggested Verification Commands

Use focused commands while iterating, then broaden before cutover:

```bash
python3 -m pytest tests/execution -q --tb=short
python3 -m pytest -m "not integration" tests -q --tb=short
python3 -m pytest tests/integration -q --tb=short
python3 -m ruff check
python3 -m ruff format --check .
```

The integration suite depends on external tools and data. The Slurm test is
skipped unless explicitly enabled inside a Slurm allocation.

## Merge Criteria

The migration is ready to merge when:

- all supported operation and process paths run through Prefect/Dask
- the target-environment equivalence gate passes for the supported feature
  matrix
- restart/reset behaviour works with Prefect-produced operation state
- runtime, resource, filesystem, command-log, and Slurm checks pass
- CI and packaging no longer require CWL runtime dependencies
- docs describe Prefect/Dask as the only supported production execution path
- CWL artifacts that remain in the tree are static reference fixtures or
  compatibility helpers, not production execution machinery
