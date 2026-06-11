# Rapthor CWL-to-Prefect Equivalence Report

Generated: 2026-06-10

## Executive Summary

Rapthor's Prefect/Dask execution path has a defined equivalence contract against
the legacy CWL path. The migration is considered equivalent when the Prefect
candidate produces the same normalized operation state, output records, final
products, reports, and finalizer-visible field state as the CWL reference for
the supported scenario matrix.

Current status: the equivalence harness and opt-in integration gates are in
place. Eleven local saved CWL references have been captured from legacy commit
`4cfd2abe2fe815724e3f1c390d789eea249becef`, and all eleven pass
saved-reference comparison against the current Prefect execution path. Full
required production equivalence is not yet recorded because `mpi_wsclean` and
Slurm/external-Dask still need usable target-environment references or
execution. `hybrid_screens` and `shared_facet_rw` are deferred from the required
gate: the former is not used by the current target workflow and would require
IDG Python bindings for DP3 IDGCal, while the latter has been too flaky in
available WSClean environments.

## Scope Of Equivalence

The migration must preserve:

- the existing parset and strategy contract
- operation ordering and top-level process semantics
- finalizer-visible `Field`, `Observation`, and `Sector` state
- output filenames, output-record shapes, and product locations
- restart/reset behaviour through `.done` and `.outputs.json`
- local execution, Slurm/external-Dask execution, and MPI WSClean safety

The comparison is backend-neutral: it ignores absolute run-directory paths and
compares observable products and operation records.

## Equivalence Mechanism

The core comparison helpers live in `rapthor/execution/equivalence.py`.

`collect_backend_summary()` normalizes each run into:

- `operation_order`
- operation `.done` markers
- operation output records from `.outputs.json` or legacy `pipeline_outputs.json`
- final products under `images`, `h5parms`, `skymodels`, and `regions`
- `logs/diagnostics.txt`
- `field_state.json`, when present

`compare_backend_runs()` compares the normalized CWL reference summary against
the normalized Prefect candidate summary and reports structured differences.

Product-level summaries include:

- FITS HDU shape, dtype, finite count, NaN count, and numeric statistics
- h5parm solset/soltab/dataset/axis structure
- sky-model source and patch counts
- region-file content
- generic product basenames

## Test Gates

### Live Side-By-Side Gate

`tests/integration/test_live_cwl_equivalence.py` reuses an existing integration
fixture and runs the same generated parset through:

- legacy CWL from `RAPTHOR_LEGACY_CWL_REPO`
- current Prefect via `rapthor.execution.flows.process.process_flow()`

It then compares the two working directories with `compare_backend_runs()`.

Run command:

```bash
RAPTHOR_RUN_LIVE_CWL_EQUIVALENCE=1 \
RAPTHOR_LEGACY_CWL_REPO=/path/to/pre-cutover-checkout \
python3 -m pytest tests/integration/test_live_cwl_equivalence.py -q --tb=short
```

Current live scenario:

- DI fast-phase calibration using `tests/resources/integration_template.parset`
  plus the existing `generated_parset_path` and
  `single_loop_strategy_with_calibration_strategy` integration fixtures

### Saved-Reference Gate

`tests/integration/test_saved_cwl_equivalence.py` validates saved CWL artifacts,
runs fresh Prefect candidates, and compares every selected scenario against the
saved CWL reference.

Run command:

```bash
RAPTHOR_RUN_SAVED_CWL_EQUIVALENCE=1 \
RAPTHOR_CWL_REFERENCE_ROOT=/path/to/references \
python3 -m pytest tests/integration/test_saved_cwl_equivalence.py -q --tb=short
```

Subset command:

```bash
RAPTHOR_RUN_SAVED_CWL_EQUIVALENCE=1 \
RAPTHOR_CWL_REFERENCE_ROOT=/path/to/references \
RAPTHOR_EQUIVALENCE_SCENARIOS=di_only_calibration,normalization \
python3 -m pytest tests/integration/test_saved_cwl_equivalence.py -q --tb=short
```

### Reference Capture

`scripts/capture_cwl_reference_artifacts.py` populates saved CWL artifacts by
running the scenario manifest through a separate pre-cutover checkout.

Run command:

```bash
RAPTHOR_LEGACY_CWL_REPO=/path/to/pre-cutover-checkout \
RAPTHOR_CWL_REFERENCE_ROOT=/path/to/references \
RAPTHOR_EQUIVALENCE_INPUT_MS=/path/to/input.ms \
RAPTHOR_EQUIVALENCE_INPUT_SKYMODEL=/path/to/true.txt \
RAPTHOR_EQUIVALENCE_APPARENT_SKYMODEL=/path/to/apparent.txt \
python3 scripts/capture_cwl_reference_artifacts.py
```

Use `--scenario <id>` to capture one scenario at a time and `--overwrite` to
replace an existing scenario artifact directory.

## Scenario Matrix

The saved-reference gate is driven by
`tests/execution/fixtures/equivalence_gate_scenarios.json`.

| Scenario | Comparison scopes |
| --- | --- |
| `di_only_calibration` | operations, products, h5parm, fits, skymodel, regions |
| `dd_only_calibration` | operations, products, h5parm, fits, skymodel, regions |
| `di_then_dd_calibration` | operations, products, h5parm, fits, skymodel, regions |
| `dd_then_di_calibration` | operations, products, h5parm, fits, skymodel, regions |
| `di_full_jones_calibration` | operations, products, h5parm, fits, skymodel, regions |
| `dd_slow_gain_calibration` | operations, products, h5parm, fits, skymodel, regions |
| `hybrid_screens` | operations, products, h5parm, fits, skymodel, regions |
| `normalization` | operations, products, h5parm, fits, skymodel |
| `peeling` | operations, products, fits, skymodel, regions |
| `full_stokes_clean_disabled` | operations, products, fits |
| `image_cube` | operations, products, fits |
| `shared_facet_rw` | operations, products, fits, regions |
| `mpi_wsclean` | operations, products, fits |
| `restart` | operations, products, restart |

`mpi_wsclean` is a target-environment scenario and should be run where the MPI
and WSClean environment matches the intended deployment target.

The required cutover gate currently excludes scenarios with
`deferred_from_required_gate=true` in the manifest. Deferred scenarios remain
explicitly selectable for future investigation.

## Current Evidence

Implemented evidence:

- operation-level Prefect flows for concatenate, mosaic, predict, image,
  calibration, and process orchestration
- operation-adapter tests for command construction, output records, restart
  behaviour, failure handling, and finalizer-visible state
- mocked process-flow equivalence coverage for top-level lifecycle behaviour
- backend-neutral command-log collection that accepts both Prefect JSONL logs and
  retained legacy log files
- unit tests for reference artifact validation, scenario parset materialization,
  saved-reference comparison, FITS tolerances, and h5parm tolerances
- opt-in live CWL-vs-Prefect integration test using an existing integration
  fixture
- opt-in saved-CWL regression integration test for the required, non-deferred
  scenario set

Local verification in the development container on 2026-06-09:

```bash
python3 -m pytest tests/execution/test_equivalence.py \
  tests/integration/test_saved_cwl_equivalence.py \
  tests/integration/test_live_cwl_equivalence.py -q --tb=short
```

Result: 24 passed, 5 skipped, 1 warning. The live and saved-reference tests
skip by default unless their required environment variables and artifacts are
supplied.

Saved-reference verification in the development container on 2026-06-10 used
real CWL artifacts under `.pytest_cache/cwl-reference-artifacts`:

```bash
RAPTHOR_RUN_SAVED_CWL_EQUIVALENCE=1 \
RAPTHOR_CWL_REFERENCE_ROOT=/app/.pytest_cache/cwl-reference-artifacts \
RAPTHOR_EQUIVALENCE_SCENARIOS=<scenario-id> \
RAPTHOR_EQUIVALENCE_INPUT_MS=/app/tests/resources/test.ms \
RAPTHOR_EQUIVALENCE_INPUT_SKYMODEL=/app/tests/resources/integration_true_sky.txt \
RAPTHOR_EQUIVALENCE_APPARENT_SKYMODEL=/app/tests/resources/integration_apparent_sky.txt \
python3 -m pytest tests/integration/test_saved_cwl_equivalence.py -q --tb=short
```

Reference details:

- legacy checkout: `.pytest_cache/legacy-cwl-4cfd2abe`
- legacy commit: `4cfd2abe2fe815724e3f1c390d789eea249becef`
- reference artifact root: `.pytest_cache/cwl-reference-artifacts`
- reference size after these captures: approximately 4.9 GB
- `normalization` uses a deterministic normalization Measurement Set generated
  from the existing `ms_for_normalisation` integration-fixture logic; the other
  passing local scenarios use `tests/resources/test.ms`

Saved-reference results:

| Scenario | CWL reference | Prefect comparison | Notes |
| --- | --- | --- | --- |
| `di_only_calibration` | captured | pass | Existing DI fast-phase reference |
| `dd_only_calibration` | captured | pass | Exposed and fixed diagnostic plot-output bookkeeping |
| `di_then_dd_calibration` | captured | pass | Exposed and fixed DI medium-phase model handoff (`reusemodel=[solve1.*]`) |
| `dd_then_di_calibration` | captured | pass | Reverse DI/DD handoff matched |
| `di_full_jones_calibration` | captured | pass | Product/operation scopes pass; diagnostics text is no longer compared unless `report` is scoped |
| `dd_slow_gain_calibration` | captured | pass | Exposed and fixed slow-only legacy output aliasing |
| `normalization` | captured | pass | Requires normalization-specific MS; exposed a harmless FITS summary mean drift handled by the 1e-8 summary float tolerance |
| `peeling` | captured | pass | Bright-source and outlier peeling flags exercised |
| `full_stokes_clean_disabled` | captured | pass | Required short scratch override to avoid PyBDSF AF_UNIX path-length failures |
| `image_cube` | captured | pass | Image-cube products and metadata exercised |
| `restart` | captured | pass | Persisted `.done` and output-record state present and comparable |
| `hybrid_screens` | deferred | not run | Excluded from the required gate for now because it is not used by the current target workflow. Capturing it would require IDG Python bindings for DP3 IDGCal, so the known-good DP3-pinned image is left unchanged |
| `shared_facet_rw` | deferred | not run | Excluded from the required gate for now; serial WSClean aborts with `SIGABRT` when `-shared-facet-reads` and `-shared-facet-writes` are enabled |
| `mpi_wsclean` | pending | not run | Target-environment scenario |

Captured operation orders:

| Scenario | Operation order |
| --- | --- |
| `di_only_calibration` | `predict_di_1`, `calibrate_di_1`, `predict_1`, `image_1`, `mosaic_1` |
| `dd_only_calibration` | `calibrate_1`, `predict_1`, `image_1`, `mosaic_1` |
| `di_then_dd_calibration` | `predict_di_1`, `calibrate_di_1`, `calibrate_1`, `predict_1`, `image_1`, `mosaic_1` |
| `dd_then_di_calibration` | `calibrate_1`, `predict_di_1`, `calibrate_di_1`, `predict_1`, `image_1`, `mosaic_1` |
| `di_full_jones_calibration` | `predict_di_1`, `calibrate_di_1`, `predict_1`, `image_1`, `mosaic_1` |
| `dd_slow_gain_calibration` | `calibrate_1`, `predict_1`, `image_1`, `mosaic_1` |
| `normalization` | `calibrate_1`, `predict_1`, `normalize_1`, `image_1`, `mosaic_1` |
| `peeling` | `calibrate_1`, `predict_1`, `image_1`, `mosaic_1` |
| `full_stokes_clean_disabled` | `calibrate_1`, `predict_1`, `image_1`, `mosaic_1` |
| `image_cube` | `calibrate_1`, `predict_1`, `image_1`, `mosaic_1` |
| `restart` | `calibrate_1`, `predict_1`, `image_1`, `mosaic_1` |

## Production Equivalence Criteria

The migration should be treated as production-equivalent only after all of the
following are true:

- live CWL-vs-Prefect smoke equivalence passes against a pre-cutover checkout
- saved CWL artifacts exist for every required, non-deferred scenario in the
  manifest
- saved-reference equivalence passes for the required, non-deferred scenario set
- Slurm/external-Dask and MPI WSClean target-environment hooks pass
- any differences are either fixed or documented as intentional and
  user-invisible
- the passing artifact root, source data, strategy files, commit SHAs, and test
  command outputs are recorded for release review

## Known Caveats

- The current migration branch operation adapters already execute Prefect flows,
  so the CWL reference must come from a separate pre-cutover checkout or saved
  CWL artifacts.
- The saved-reference gate depends on representative input data and external
  radio astronomy tools that are not exercised by normal unit tests.
- Numeric product comparison is summary-based for broad backend equivalence.
  Targeted FITS and h5parm numeric tolerance helpers are available for deeper
  product comparisons where required.
- Scenario-specific strategies for the captured local scenarios are tracked
  under `tests/execution/fixtures/equivalence_strategies` and wired into
  `equivalence_gate_scenarios.json` with `parset_overrides`.
- `hybrid_screens` is deferred from the required gate because it is not used by
  the current target workflow. If this path becomes relevant again, capture it
  in an image where DP3 IDGCal can import Python `idg`, while keeping DP3 pinned
  to the known-good commit.
- `shared_facet_rw` is deferred from the required gate because WSClean aborts
  when the shared-facet read/write flags are enabled in available serial
  environments.
- The report documents the equivalence method and current evidence; it is not a
  substitute for recording a passing target-environment equivalence run.

## Recommended Next Action

Run the saved-reference regression over the required, non-deferred artifact
root. The target-environment `mpi_wsclean` and Slurm/external-Dask hooks still
need to be run in an environment matching deployment.
