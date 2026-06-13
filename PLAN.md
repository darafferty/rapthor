# Rapthor Prefect/Dask Cutover Plan

Status snapshot: 2026-06-13.

## Goal

Keep Prefect/Dask as the production execution path for Rapthor while preserving
the public parset/strategy contract, operation ordering, restart state, output
records, product filenames/locations, finalizer-visible field state, and local
execution behaviour.

## Current Position

The migration is in the post-cutover cleanup stage.

- `rapthor.process.run()` routes through
  `rapthor.execution.flows.process.process_flow()`.
- Prefect flows exist for concatenate, mosaic, predict, image, calibration, and
  top-level process orchestration.
- Operation adapters for concatenate, mosaic, predict, image, image-initial,
  image-normalize, and calibration execute Prefect flows and produce
  finalizer-compatible output records.
- Demo and observability support is in place: persistent Prefect dashboard
  support, unique run directories, local/external Dask dashboard support, Dask
  performance reports, streamed external-command logs, Python logs in Prefect,
  plot/FITS artifacts, command timing artifacts, and richer synthetic demo data.
- CI has been adjusted and verified passing: tests that start a Prefect test
  server are marked and run serially, while the remaining non-integration tests
  use all available xdist workers.
- CI pins DP3 to the dev-container-tested commit `18e793a4`.
- User-facing docs and release notes describe Prefect/Dask as the production
  execution path.
- CWL/Toil production runtime has been removed. Toil, StreamFlow, `cwltool`,
  the in-tree CWL runner, old CWL workflow/parset files, static CWL tests, and
  active CWL equivalence gates have been removed from the codebase.
- CWL-to-Prefect parity is documented in `EQUIVALENCE_REPORT.md` as historical
  migration evidence. The report records the comparison method, passing
  scenario matrix, legacy commit, and deferred target-environment checks.

## Completed Refactor Work

- Consolidated repeated operation-adapter flow execution through the base
  `Operation.run_prefect_flow()` helper.
- Removed redundant `uses_python_flow()` overrides from Prefect-backed operation
  adapters.
- Consolidated repeated base flow parset keys through
  `Operation.flow_parset_parameters()`.
- Removed the static CWL template rendering path from the base `Operation`.
- Removed unused legacy runtime attributes from the base `Operation`.
- Removed operation-level script/MPI path attributes that were only needed by
  the old CWL runtime.
- Inlined the base operation's `debug_workflow` state into
  `keep_temporary_files`; the parset option remains supported.
- Replaced operation-level input coverage tests that rendered CWL templates with
  explicit Prefect flow payload contracts for calibration and prediction.
- Cleaned operation tests by renaming image-flow mocks, removing retired cluster
  settings from local fixtures, and replacing placeholder tests with concrete
  adapter assertions.
- Deduplicated the public `rapthor.process.run_steps()` compatibility helper by
  delegating to the Prefect process-step scheduler while preserving the
  monkeypatchable operation constructors used by tests.
- Removed the legacy `cwl_runner` setting from the Prefect demo parset and demo
  data generator.
- Removed old CWL workflow/parset files, `cwltool`, CWL parser tests, the CWL
  reference capture script, and active CWL equivalence test gates after the
  parity evidence was documented.
- Renamed the remaining generic file/directory record helper from
  `rapthor.lib.cwl` to `rapthor.lib.records` and replaced `CWLFile`/`CWLDir`
  with neutral `FileRecord`/`DirectoryRecord` helpers.
- Renamed migration-era execution golden fixtures from `legacy_*_reference.json`
  to neutral command/output reference fixtures and removed stale
  CWL-equivalence wording from active execution-flow tests.

## Remaining Work

### 1. Keep Focused Runtime Coverage Green

- Continue using operation and execution-flow tests as the main regression
  suite for command construction, payloads, output records, restart/reuse,
  failure handling, artifacts, logging, resource validation, task-runner
  selection, work directories, and process orchestration.
- Keep representative external-tool integration tests for DP3, WSClean,
  EveryBeam, PyBDSF, diagnostics, and mosaic hand-off.
- Fix real differences found by tests or demo runs.

### 2. Final Polish

- Run the formatter/linter after the refactor settles.
- Run the narrowest meaningful tests after each cleanup slice, then the broader
  non-integration suite before merge.
- Keep generated demo data, large reference artifacts, and run outputs ignored
  by VCS.

## Deferred Follow-Up

- Slurm/external-Dask validation remains deferred until after the migration
  cutover and should be run inside a representative Slurm allocation.
- MPI WSClean validation remains deferred until it can be run with the intended
  MPI/WSClean deployment stack.
- Dask task-granularity/resource optimization is explicitly deferred. Use the
  Dask dashboard and performance reports later to review worker/resource tuning
  and possible finer-grained parallelism.
- Revisit `hybrid_screens` only if it becomes a supported target workflow.
- Revisit `shared_facet_rw` after WSClean shared-facet read/write behaviour is
  reliable in the intended environment.

## Useful Commands

Serial Prefect-server lane:

```bash
python3 -m pytest -m "not integration and prefect" -k "not test_field.py" tests -q --tb=short
```

Parallel non-Prefect lane:

```bash
python3 -m pytest -m "not integration and not prefect" -n auto -k "not test_field.py" tests -q --tb=short
```

Focused process-flow checks:

```bash
python3 -m pytest tests/execution/test_process_flow.py tests/test_process.py -q --tb=short
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

- All supported public operation and process paths run through Prefect/Dask.
- Restart/reset behaviour works with Prefect-produced operation state.
- Runtime, resource, filesystem, command-log, artifact, and local Dask checks
  pass.
- CI is stable with Prefect tests isolated from high xdist fan-out.
- Docs describe Prefect/Dask as the supported production execution path.
- Docs record Slurm/external-Dask and MPI WSClean checks as deferred
  post-migration validation.
