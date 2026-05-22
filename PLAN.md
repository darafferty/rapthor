# Calibration Strategy Implementation Plan

This plan tracks the remaining work needed to make the calibration strategy in
`CALIBRATION_STRATEGY.md` match the code and keep the unit, CWL, and integration
tests green.

## Current State

- The scheduler preserves the order of `calibration_strategy` keys and runs DI
  prediction before DI calibration.
- Focused `tests/operations/test_calibrate.py` coverage now passes for DD and DI
  operation initialization, parameter setup, finalization, BDA/IBP cases, and
  solution-combine mode.
- DI calibration operation naming now has focused unit coverage for
  `dd -> calibrate` and `di -> calibrate_di`, but process-level and integration
  coverage still need to prove the directory, log, solution, and `.done` marker
  contract across mixed DI/DD cycles.
- DD-only calibration is still mostly driven by legacy flags such as
  `do_slowgain_solve`, not directly by the DD solve list.
- DI calibration is not yet a general solve-list implementation. It is mostly
  hard-coded around full-Jones-style inputs and outputs.
- DD prediction can apply existing DD solutions when preparing DI inputs, but DD
  calibration cannot yet apply DI solutions before solving.
- Imaging builds applycal steps from the strategy order, but solution file
  selection does not cleanly distinguish DD scalar/gain, DI scalar/gain, and
  DI full-Jones products.
- CWL template validation is the current focused blocker. The latest completed
  focused run has all operation tests passing but calibrate CWL generation still
  failing.
- An older broader CWL log also shows image workflow validation failures that are
  not covered by the calibration-focused run and should be fixed before broad
  test closure.

## Latest Pytest Snapshot

Reviewed `/tmp/rapthor_focused_calibrate.log` from May 22 2026:

- `56` tests collected: `38` passed, `18` failed.
- All `tests/operations/test_calibrate.py` cases passed.
- The remaining focused failures are in `tests/lib/test_cwl.py`:
  - all `16` DD `test_calibrate_workflow` parameter combinations;
  - both DI `test_calibrate_di_workflow` cases for `max_cores=None` and
    `max_cores=8`.
- DD calibrate failures split into two patterns:
  - with `generate_screens=False`, the generated workflow defines
    `combined_solutions` more than once and emits an output item without a
    required `type` field;
  - with `generate_screens=True`, generated steps reference missing IDs such as
    `solint_solve1_timestep`, `solve/output_h5parm1`,
    `collected_solve1_h5parm`, `combined_solve1_solve2_h5parm`,
    `combined_solve1_solve2_solve4_h5parm`, `dp3_steps`,
    `calibrator_patch_names`, `calibrator_fluxes`,
    `solution_combine_mode`, `max_normalization_delta`,
    `scale_normalization_delta`, `phase_center_ra`, and `phase_center_dec`.
- DI calibrate failures are validation mismatches in `calibrate_di_pipeline.cwl`:
  unsupported `correctfreqsmearing` and `correcttimesmearing` inputs are passed
  to `ddecal_solve.cwl`, `solve*_solutions_per_direction` has the wrong nested
  array shape, and several optional array outputs/inputs are typed too loosely
  for the downstream sinks.
- The active VS Code terminal reported another `pytest` still running during this
  review. Reconcile this snapshot if that run finishes with newer output.

Reviewed `ci.log` as broader context:

- Repeated image workflow validation failures are present in
  `tests/lib/test_cwl.py::TestImageWorkflow`.
- The recurring image workflow causes are an unsupported
  `save_filtered_model_image` input passed to `filter_skymodel.cwl` and an
  incompatible `output_image` source wired into the `skymodel_image_fits` sink
  through `merge_nested` plus `pickValue: all_non_null`.

## Target Behaviour

Implement the four branches from `CALIBRATION_STRATEGY.md`:

1. DI only:
   - `Predict("di")`
   - `Calibrate("di")`
   - apply DI solutions during imaging

2. DD only:
   - `Calibrate("dd")`
   - pass DD solutions and facet regions to imaging

3. DI then DD:
   - `Predict("di")`
   - `Calibrate("di")`
   - apply DI solutions before DD calibration
   - `Calibrate("dd")`
   - apply DD plus DI solutions during imaging

4. DD then DI:
   - `Calibrate("dd")`
   - use DD solutions and facet regions during DI prediction
   - `Predict("di")`
   - `Calibrate("di")`
   - apply DD plus DI solutions during imaging

## Implementation Tasks

1. Close the operation naming contract.
   - Make DI calibration use a distinct `calibrate_di_<cycle>` operation name,
     working directory, log directory, solutions directory, and `.done` marker.
   - Keep DD calibration as `calibrate_<cycle>`.
   - Verify process-level scheduling and integration log expectations use the
     same naming contract that the focused `Calibrate` unit tests now expect.
   - Update tests to use the chosen naming consistently.

2. Add a calibration solve planner.
   - Convert each mode's solve list into DP3 solve slots, modes, output names,
     collection names, solution intervals, and post-processing steps.
   - Support `fast_phase`, `medium_phase`, `slow_gains`, and `full_jones`.
   - Remove behavioural dependence on `do_slowgain_solve` where an explicit
     `calibration_strategy` is present.
   - Preserve user-specified mode order and solve order.

3. Generalize DI calibration.
   - Support DI fast phase, medium phase, slow gains, and full Jones from the
     strategy solve list.
   - Generate DI-specific output filenames such as `fast_phase_di_*.h5parm`,
     `medium1_phase_di_*.h5parm`, and `slow_gains_di_*.h5parm`.
   - Finalize DI products into stable solution paths and scan them into field
     state.

4. Generalize DD calibration.
   - Support custom DD solve lists instead of always assuming fast plus medium
     and optional slow gain.
   - Ensure DD-only `slow_gains` maps to the expected single solve slot and
     output names.
   - Preserve existing legacy behaviour when no explicit `calibration_strategy`
     is provided.

5. Wire solution application between branches.
   - For DI then DD, pass DI applycal steps and DI solution files into the DD
     calibration workflow before DD solves.
   - For DD then DI, keep applying DD solutions during `Predict("di")`.
   - For imaging, build applycal steps from actual available solution products,
     not just strategy text.

6. Fix DD calibrate CWL validation failures.
   - Align Python input keys with `calibrate_pipeline.cwl` input IDs.
   - Ensure every generated workflow output is unique and has a `type`; repair
     the duplicate/dangling `combined_solutions` output produced when
     `generate_screens=False`.
   - For `generate_screens=True`, make all screen-generation solve, collect,
     combine, process, and adjust-source steps reference declared inputs or real
     step outputs. In particular, repair references to solve-slot solints,
     collected and combined h5parm names, `dp3_steps`, calibrator names/fluxes,
     normalization limits, phase center coordinates, and `solution_combine_mode`.
   - Reconcile shared `solve/output_h5parm*` references with the split solve step
     IDs such as `solve_fast_phases_only` and `solve_fast_phases_slow_gains`.
   - Fix duplicate `adjust_h5parm_sources` step IDs.
   - Fix the `step2`/`solve2` typo in the medium-phase combine condition.
   - Keep all combinations of `generate_screens`, `use_image_based_predict`,
     `do_slowgain_solve`, and `max_cores` valid under `cwltool --validate`.

7. Fix DI calibrate CWL validation failures.
   - Decide whether DI uses a separate `calibrate_di_pipeline.cwl` or the shared
     `calibrate_pipeline.cwl`, then make tests and log expectations match.
   - Stop passing unsupported `correctfreqsmearing` and `correcttimesmearing`
     inputs to `ddecal_solve.cwl`, or add those inputs to the step definition if
     they are truly required.
   - Match `solve*_solutions_per_direction` to the nested array shape expected by
     `ddecal_solve.cwl`.
   - Tighten optional array and nullable types for DI solve outputs, BDA inputs,
     smoothness reference frequencies, and `max_threads` so generated workflows
     validate cleanly for both `max_cores=None` and `max_cores=8`.

8. Fix image workflow CWL validation failures from the broader log.
   - Align the image subpipeline with `filter_skymodel.cwl`; either remove the
     generated `save_filtered_model_image` input or add it to the step contract
     and implementation.
   - Repair the `output_image` to `skymodel_image_fits` wiring so the source type,
     `linkMerge`, and `pickValue` match the sink type.
   - Review nearby image workflow warnings for Q/U/V channel arrays, cube output
     names, optional masks, offsets, and diagnostic plots after the hard errors
     are fixed.

9. Update tests.
   - Add focused unit tests for the solve planner.
   - Update `test_calibrate.py` for DI/DD operation names and CWL input IDs.
   - Keep the existing `test_calibrate_workflow` matrix as regression coverage
     for both screen and non-screen DD workflow generation.
   - Keep `test_calibrate_di_workflow` validating both `max_cores` branches.
   - Add or update image workflow CWL regression cases for the
     `filter_skymodel.cwl` input contract and `skymodel_image_fits` wiring.
   - Add process-level tests for DI-only, DD-only, DI-then-DD, and DD-then-DI
     operation ordering.
   - Update integration tests to inspect the correct log directories and CWL
     step names.
   - Add explicit mixed-order integration coverage for:
     - `{"di": [...], "dd": [...]}`
     - `{"dd": [...], "di": [...]}`

10. Keep the test environment reproducible.
   - Local `pytest` is now available and produced the focused results above.
   - Existing `.tox` environments may still contain stale absolute paths; recreate
     tox before relying on full-suite or lint results.
   - Note the local focused log used `cwltool 3.1.20260108082145`, while the
     broader CI-style log used `cwltool 3.2.20260413085819`; validate with the CI
     version if validator behavior differs.

## Verification Sequence

Run the checks in increasing scope:

```bash
python -m pytest tests/operations/test_calibrate.py tests/lib/test_cwl.py -k "calibrate"
python -m pytest tests/lib/test_cwl.py -k "image_workflow"
python -m pytest tests/test_process.py tests/lib/test_field.py tests/lib/test_strategy.py
python -m pytest tests/operations/test_calibrate.py tests/operations/test_predict.py tests/operations/test_image.py
python -m pytest tests/lib/test_cwl.py
python -m pytest -m "integration" tests/integration
tox -e lint
tox
```

All integration tests should be green before the implementation is considered
complete.
