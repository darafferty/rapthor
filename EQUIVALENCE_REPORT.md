# Rapthor CWL-to-Prefect Equivalence Report

Generated: 2026-06-09

## Executive Summary

Rapthor's Prefect/Dask execution path has a defined equivalence contract against
the legacy CWL path. The migration is considered equivalent when the Prefect
candidate produces the same normalized operation state, output records, final
products, reports, and finalizer-visible field state as the CWL reference for
the supported scenario matrix.

Current status: the equivalence harness and opt-in integration gates are in
place. Final production equivalence is not yet recorded in this checkout because
the target-environment CWL reference runs still need to be executed from a
pre-cutover checkout, or supplied as saved CWL reference artifacts.

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
RAPTHOR_EQUIVALENCE_STRATEGY=/path/to/strategy.py \
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
- opt-in saved-CWL regression integration test for the full scenario manifest

Local verification in the development container on 2026-06-09:

```bash
python3 -m pytest tests/execution/test_equivalence.py \
  tests/integration/test_saved_cwl_equivalence.py \
  tests/integration/test_live_cwl_equivalence.py -q --tb=short
```

Result: 24 passed, 5 skipped, 1 warning. The live and saved-reference tests
skip by default unless their required environment variables and artifacts are
supplied.

## Production Equivalence Criteria

The migration should be treated as production-equivalent only after all of the
following are true:

- live CWL-vs-Prefect smoke equivalence passes against a pre-cutover checkout
- saved CWL artifacts exist for every required scenario in the manifest
- saved-reference equivalence passes for the full scenario manifest
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
- The report documents the equivalence method and current evidence; it is not a
  substitute for recording a passing target-environment equivalence run.

## Recommended Next Action

Run the live side-by-side smoke gate first. If it passes, capture saved CWL
references scenario by scenario, then run the full saved-reference regression.
Record the legacy checkout SHA, current Prefect checkout SHA, input data paths,
strategy paths, environment, and pytest output alongside the saved artifacts.
