# Rapthor Prefect/Dask Migration Plan

Status snapshot: 2026-06-11.

## Goal

Make Prefect/Dask the production execution path for Rapthor while preserving
the existing parset/strategy contract, operation ordering, restart state,
output records, product filenames/locations, finalizer-visible field state, and
the local execution contract. CWL should remain only as a reference mechanism
until equivalence is proven, then as static fixtures or compatibility helpers.
Slurm/external-Dask and MPI WSClean validation are deferred until after the
migration cutover.

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
- `rapthor.process.run()` now routes through
  `rapthor.execution.flows.process.process_flow()`, making Prefect/Dask the
  public processing path.
- Saved-reference equivalence is in place. The local saved-reference gate runs
  eleven CWL reference scenarios from legacy commit
  `4cfd2abe2fe815724e3f1c390d789eea249becef` and passes comparison against the
  current Prefect path.
- The live CWL-vs-Prefect smoke gate also passes for the existing DI fast-phase
  integration scenario against the preserved legacy checkout.
- Demo and observability support is in place: persistent Prefect dashboard
  support, unique run directories, local/external Dask dashboard support, Dask
  performance reports, streamed external-command logs, Python logs in Prefect,
  plot/FITS artifacts, command timing artifacts, and richer synthetic demo data.
  Several manual demo runs have been completed successfully and look good for
  the current migration stage.
- Recent correctness fixes include cycle-aware h5parm provenance/filtering so
  stale solutions from previous cycles are not applied accidentally.
- Real external-tool coverage has been refreshed in the dev container for
  selected DI/DD calibration scenarios. The refreshed subset exercises DP3
  prediction/DDECal, DI full-Jones plotting, WSClean imaging, EveryBeam beam
  application, PyBDSF source extraction, diagnostics, and mosaic hand-off
  through the public `rapthor` route.
- CI has been adjusted and verified passing: tests that start a Prefect test
  server are marked and run serially, while the remaining non-integration tests
  still use `pytest-xdist -n auto`.
- CI now pins DP3 to the dev-container-tested commit `18e793a4`.
- User-facing docs and release notes have been refreshed for the current
  migration stage: Prefect/Dask is documented as the production path,
  obsolete Toil/CWL production wording has been removed from the main Sphinx
  pages, dashboard/artifact/restart behaviour is documented, and deferred
  Slurm/external-Dask plus MPI WSClean validation is called out.
- CWL production runtime has been removed. Toil and StreamFlow are no longer
  production dependencies, `cwltool` is test/reference-only, the in-tree CWL
  runner and Toil batch-system helper have been deleted, and the base
  `Operation` no longer falls back to CWL execution. Preserved CWL templates
  and CWL-shaped record helpers remain as static parity/reference material.

## Evidence Already Available

- Execution-layer unit and flow tests cover command construction, payloads,
  output records, restart/reuse, failure handling, artifacts, logging, resource
  validation, task-runner selection, work directories, and equivalence helpers.
- Mocked process-flow tests cover final-only, selfcal,
  convergence/divergence/failure, repeated final cycles, normalization,
  full-Stokes/cube flags, concatenation, calibration strategy hand-offs,
  validation failures, public route delegation, and artifact publication.
- `tests/execution/fixtures/equivalence_gate_scenarios.json` defines the
  saved-reference scenario matrix.
- `scripts/capture_cwl_reference_artifacts.py` can populate CWL reference
  artifacts from a preserved pre-cutover checkout.
- `tests/integration/test_saved_cwl_equivalence.py` compares fresh Prefect runs
  against saved CWL artifacts when explicitly enabled.
- `tests/integration/test_live_cwl_equivalence.py` can run a live CWL checkout
  and the current Prefect flow side by side for a smoke scenario; this gate now
  passes in the dev container.
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

Deferred post-migration target-environment proof:

- `mpi_wsclean`: should run in the intended MPI/WSClean deployment
  environment after the migration cutover.
- Slurm/external-Dask: should run inside a representative Slurm allocation
  after the migration cutover.

Deferred optional tasks:

- Revisit `hybrid_screens` only if it becomes a supported target workflow. At
  that point, use an image where DP3's IDGCal can import Python `idg`, capture
  CWL references, and add the scenario back to the required gate. Do not install
  or pin to a newer DP3 for this without separate validation.
- Revisit `shared_facet_rw` after WSClean shared-facet read/write behaviour is
  reliable in the intended environment, then capture references and add it back
  to the required gate.

## Remaining Work

### 1. Keep Local Equivalence Evidence Green

Local equivalence is the cutover gate. Slurm/external-Dask and MPI WSClean are
deferred until after the migration is complete.

- Keep the saved-CWL local gate as the release regression for local scenarios.
  Rerun it whenever references, product publishing, or equivalence helpers
  change.
- Keep the live CWL-vs-Prefect smoke gate as a regression for the preserved
  legacy checkout.
- Record the passing source data, artifact root, strategy files, commit SHAs,
  tool versions, commands, and test output.
- Fix real differences. Document only intentional, user-invisible differences.

### 2. Refresh Real External-Tool Coverage

The unit and mocked flow tests are broad, but the production confidence still
depends on real radio-astronomy tools and representative data.

- Done for the current local migration stage: a focused dev-container subset
  passed on 2026-06-11 after refreshing the installed `plotrapthor` script from
  the checkout.
- Covered paths include DP3 prediction/DDECal, DI fast/medium/slow solves, DI
  full-Jones solves, DD fast/medium/slow solves, WSClean imaging, EveryBeam beam
  application, PyBDSF source extraction, image diagnostics, and mosaic hand-off.
- Broader staging or release-candidate runs can still be done if desired, but
  they are not a separate cutover blocker unless they expose a real regression.
- Leave IDG/screen-generation coverage deferred unless `hybrid_screens` becomes
  a supported target path again.

### 3. Cut Over The Public Route

Done for the current migration stage.

- `rapthor.process.run()` routes through `process_flow()`.
- Keep no public `execution_backend` selector.
- Operation/process tests treat Prefect/Dask as the expected production route.
- Keep saved CWL artifacts and CWL-derived command/output fixtures only as
  parity evidence.
- User-facing docs no longer describe Toil/CWL as the normal execution
  mechanism.

### 4. Remove CWL Production Runtime

Done for the production runtime.

- Removed Toil and StreamFlow from production dependencies.
- Moved pinned `cwltool` to test/reference-only dependency sets for static CWL
  fixture checks.
- Removed `rapthor/lib/cwlrunner.py`, the Toil batch-system helper, and obsolete
  CWL runner tests.
- Removed CWL execution fallbacks from operation adapters; the base `Operation`
  now raises if a subclass does not provide a Prefect/Dask implementation.
- Updated lint/format targets, CI release jobs, defaults, README, and key Sphinx
  pages so Toil/CWL is no longer documented as the production route.
- Kept preserved CWL templates and CWL-shaped output helpers as static
  parity/reference material for now. Moving or renaming those can be handled in
  the post-cutover cleanup pass once no tests/docs need their current location.

### 5. Final Documentation And Release Notes

Done for the current migration stage.

- README and Sphinx docs document Prefect/Dask as the supported execution path.
- Running docs cover the local demo helper, Prefect dashboard, Dask dashboard,
  streamed logs, command timing artifacts, plot/FITS artifacts, generated demo
  data, unique working directories, and restart/debug state.
- The changelog records the Prefect/Dask cutover, equivalence evidence, demo
  and observability support, and deferred target-environment checks.
- Generated demo data and large reference artifacts remain ignored by VCS.
- Slurm/external-Dask and MPI WSClean validation are documented as deferred
  post-migration checks.

### 6. Post-Migration Target-Environment Validation

Deferred by project decision until after the migration cutover.

- Run `mpi_wsclean` with the intended MPI/WSClean stack.
- Run the Slurm/external-Dask hook inside a representative Slurm allocation.
- Fix any target-environment issues that appear there.
- Record the commands, environment, tool versions, and outcomes in the
  equivalence report or release follow-up notes.

### 7. Post-Cutover Refactor And Deduplication

Only after equivalence, public route cutover, and CWL runtime removal.

- Started with a low-risk cleanup of operation-adapter and mocked integration
  test docstrings so they describe the Prefect/Dask runtime rather than the
  removed CWL production runtime.
- Continued by consolidating repeated operation-adapter flow execution through
  the base `Operation.run_prefect_flow()` helper.
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

1. Keep the saved-reference and live-smoke equivalence gates green until merge.
2. Do a post-cutover refactor pass to clean up migration scaffolding and reduce
   duplication.
3. Run Slurm/external-Dask and MPI WSClean target-environment checks after the
   migration cutover.

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
- the local saved-reference and live smoke equivalence gates pass for the
  supported non-deferred scenario matrix
- restart/reset behaviour works with Prefect-produced operation state
- runtime, resource, filesystem, command-log, artifact, and local Dask checks
  pass
- CI is stable with Prefect tests isolated from high xdist fan-out
- docs describe Prefect/Dask as the supported production execution path
- docs record Slurm/external-Dask and MPI WSClean checks as deferred
  post-migration validation
- CWL artifacts that remain in the tree are static reference fixtures or
  compatibility helpers, not production execution machinery
