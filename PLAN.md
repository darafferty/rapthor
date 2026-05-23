# Calibration Strategy Implementation Plan

This plan tracks the remaining work needed to make the calibration strategy in
`CALIBRATION_STRATEGY.md` match the code and keep the unit, CWL, and integration
tests green.

## Current State

- The scheduler preserves the order of `calibration_strategy` keys and runs DI
  prediction before DI calibration. Process-level unit coverage now locks this
  in for DI-only, DD-only, DI-then-DD, and DD-then-DI strategy orderings.
- Focused `tests/operations/test_calibrate.py` coverage now passes for DD and DI
  operation initialization, parameter setup, planned finalization, BDA/IBP cases,
  and solution-combine mode.
- DI calibration operation naming now has focused unit coverage for
  `dd -> calibrate` and `di -> calibrate_di`, and process-level ordering tests
  now use the same DI/DD operation contract. DI integration expectations now use
  `calibrate_di_<cycle>` log directories and the shared `ddecal_solve.cwl` step.
- Calibrate operation input generation now has a solve planner that maps DD and
  DI strategy solve lists to DP3 solve slots, modes, output names, collection
  names, and solution intervals while preserving legacy defaults when no
  explicit `calibration_strategy` is present.
- DI calibration input generation can now represent fast phase, medium phase,
  slow gains, and full-Jones solve lists with DI-specific output filenames. DI
  finalize handling now copies planned scalar products and full-Jones products
  into stable solution paths and scans the active DI solution through field
  state.
- DD calibration input generation and final product handling can now represent
  custom DD solve lists, including the single-slot `slow_gains` case. Legacy DD
  default finalization remains compatible with the existing fast-only and
  slow-gain product layout.
- DD calibration can now pre-apply available DI scalar and full-Jones solutions
  before DD solves. DD prediction now prefers the stable DD scalar product when
  preparing DI inputs after a DD branch.
- Imaging now builds applycal steps only from solution products that are actually
  present in field state, so initial imaging and normalization imaging can run
  without a calibration h5parm. Mixed DI/DD scalar products are selected by
  source, with DD scalar products preferred for the shared imaging h5parm input.
- DD prediction no longer treats DI-only scalar products as DD sector h5parms;
  this prevents DI scalar solutions from being applied with DD facet directions
  that are not present in the DI h5parm.
- DI solve workflows now leave DDECal `solve.directions` unset, rather than
  passing an empty direction list, and the shared calibrate workflow has a
  no-solve4 slow-gain combine branch for explicit DI fast+medium+slow plans.
- Full-Jones-only and DI-only imaging now avoid WSClean facet mode unless a DD
  scalar h5parm is available. DI phase plus slow-gain plans apply the stable DI
  phase component during imaging and keep the DI slow-gain product finalized
  separately, avoiding the blank-image failure seen in the focused integration
  run.
- Calibrate CWL template validation is no longer the focused blocker. The DD and
  DI calibrate workflow matrix passes locally with `cwltool --validate`.
- The older broader image-workflow CWL failures have not reproduced locally. The
  full image-workflow matrix now passes with `cwltool --validate`.

## Latest Pytest Snapshot

Completed local checks after the CWL, operation, solve-planner, DI/DD
solution-flow, and focused DI integration fixes:

- `python3 -m py_compile rapthor/operations/calibrate.py rapthor/operations/image.py rapthor/operations/predict.py tests/operations/test_calibrate.py tests/operations/test_image.py tests/operations/test_predict.py tests/integration/test_di_calibration.py`:
  passed.
- `python3 -m pytest tests/operations/test_calibrate.py tests/operations/test_image.py tests/operations/test_predict.py -q`:
  `178 passed, 1 warning`.
- `python3 -m pytest tests/lib/test_cwl.py -k "calibrate" -q`:
  `18 passed, 431 deselected`.
- `python3 -m pytest tests/test_process.py tests/lib/test_field.py tests/lib/test_strategy.py -q`:
  `96 passed, 1 warning`.
- `python3 -m pytest tests/integration/test_di_calibration.py -q`:
  `7 passed`.
- `python3 -m pytest tests/lib/test_cwl.py -k "image_workflow" -q`:
  `385 passed, 64 deselected`.

Resolved focused failures:

- Added the missing screen workflow `solint_solve1_timestep` input and moved the
  Jinja branch boundary that created duplicate step keys.
- Removed the stale `model_data_column` input and aligned `Calibrate` input keys
  with the shared `calibrate_pipeline.cwl` input IDs.
- Populated the shared-template DI placeholders needed for validation while DI
  still uses the shared calibrate workflow.
- Removed duplicate non-screen `combined_solutions` outputs, fixed the duplicate
  `adjust_h5parm_sources` step IDs, and corrected the `step2`/`solve2` condition
  typo.
- Split DI/DD scalar solution selection so DD prediction ignores DI-only h5parms
  and imaging only enables facet mode when a DD scalar h5parm is available.
- Added a nullable `solve_directions` workflow input so DI DDECal runs do not
  receive `solve1.directions=[]`.
- Added a no-solve4 slow-gain combination path for explicit fast+medium+slow
  solve plans, and kept the legacy solve4 branch intact for default DD slow-gain
  post-processing.
- Adjusted DI phase plus slow-gain imaging/DD preapply to skip the destructive
  DI slow-gain amplitude application while preserving the finalized slow-gain
  product for inspection and future application paths.

Reviewed `ci.log` as broader context; the hard image-workflow errors from that
older log no longer reproduce in the local focused matrix:

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

1. Close the operation naming contract. **Complete for unit/process and updated
  integration expectations.**
   - Make DI calibration use a distinct `calibrate_di_<cycle>` operation name,
     working directory, log directory, solutions directory, and `.done` marker.
   - Keep DD calibration as `calibrate_<cycle>`.
   - Process-level scheduling now has focused coverage for DI-only, DD-only,
     DI-then-DD, and DD-then-DI ordering.
   - Integration log expectations now use the same naming contract that the
     focused `Calibrate` unit tests expect.

2. Add a calibration solve planner. **Complete for operation input planning.**
   - Convert each mode's solve list into DP3 solve slots, modes, output names,
     collection names, solution intervals, and post-processing steps.
   - Support `fast_phase`, `medium_phase`, `slow_gains`, and `full_jones`.
   - Remove behavioural dependence on `do_slowgain_solve` where an explicit
     `calibration_strategy` is present.
   - Preserve user-specified mode order and solve order.
   - Planner coverage now locks legacy DD defaults, explicit DD ordering, DI
     output suffixes, and explicit DD/DI input parameter generation.
   - Remaining post-processing and final product behavior is tracked under the
     DI/DD generalization tasks below.

3. Generalize DI calibration. **Complete for input generation and focused
  finalize coverage.**
   - Support DI fast phase, medium phase, slow gains, and full Jones from the
     strategy solve list.
   - Generate DI-specific output filenames such as `fast_phase_di_*.h5parm`,
     `medium1_phase_di_*.h5parm`, and `slow_gains_di_*.h5parm`.
   - Finalize DI products into stable solution paths and scan them into field
     state.
   - Scalar DI finalize coverage now verifies `di-solutions.h5`,
     `di-solutions-fast-phase.h5`, `di-solutions-medium1-phase.h5`, and
     `di-solutions-slow-gain.h5` copying.
   - Mixed scalar plus full-Jones DI finalize coverage now verifies
     `fulljones-solutions.h5` alongside the scalar products.

4. Generalize DD calibration. **Complete for focused input generation and
   finalization coverage.**
   - Support custom DD solve lists instead of always assuming fast plus medium
     and optional slow gain.
   - Ensure DD-only `slow_gains` maps to the expected single solve slot and
     output names.
   - Finalize DD products from the solve plan, including explicit single-slot
     `slow_gains`, while preserving legacy default product copying.
   - Preserve existing legacy behaviour when no explicit `calibration_strategy`
     is provided.

5. Wire solution application between branches. **Complete for focused unit and
   integration-log expectations.**
   - For DI then DD, pass DI applycal steps and DI solution files into the DD
     calibration workflow before DD solves.
   - For DD then DI, keep applying DD solutions during `Predict("di")`.
   - Imaging now builds applycal steps from actual available solution products,
     not just strategy text.
   - DI phase plus slow-gain plans apply the stable scalar phase component during
     imaging/DD preapply and retain the DI slow-gain product separately; this
     avoids the real-workflow blank-image failure from applying the DI amplitude
     component with the current shared prepare-imaging applycal path.
   - Mixed-order integration coverage now asserts DI-before-DD operation order,
     DD-before-DI operation order, DD pre-application of DI full-Jones solutions,
     and DI prediction pre-application of DD scalar solutions.

6. Fix DD calibrate CWL validation failures. **Complete for the focused matrix.**
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

7. Fix DI calibrate CWL validation failures. **Complete for the shared-template
  validation matrix; solve-list generalization remains in tasks 2 and 3.**
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

8. Fix image workflow CWL validation failures from the broader log. **Verified
  locally.**
   - Align the image subpipeline with `filter_skymodel.cwl`; either remove the
     generated `save_filtered_model_image` input or add it to the step contract
     and implementation.
   - Repair the `output_image` to `skymodel_image_fits` wiring so the source type,
     `linkMerge`, and `pickValue` match the sink type.
   - The local `image_workflow` matrix passes with `385 passed, 64 deselected`.
   - Review nearby image workflow warnings for Q/U/V channel arrays, cube output
     names, optional masks, offsets, and diagnostic plots after the hard errors
     are fixed.

9. Update tests. **Complete for focused and updated integration expectations.**
    - Focused unit tests for the solve planner and DI scalar finalize handling
      are now present.
    - Update `test_calibrate.py` for DI/DD operation names and CWL input IDs.
    - Keep the existing `test_calibrate_workflow` matrix as regression coverage
      for both screen and non-screen DD workflow generation.
    - Keep `test_calibrate_di_workflow` validating both `max_cores` branches.
    - Add or update image workflow CWL regression cases for the
      `filter_skymodel.cwl` input contract and `skymodel_image_fits` wiring.
    - Process-level tests for DI-only, DD-only, DI-then-DD, and DD-then-DI
      operation ordering are now present.
    - Integration tests now inspect the correct log directories and CWL step
      names.
    - Focused DI integration now passes for fast phase, slow-only strategy
      metadata, fast+medium, fast+medium+slow, full-Jones-only, DI-then-DD, and
      DD-then-DI cases.
    - Explicit mixed-order integration coverage is now present for:
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
python3 -m pytest tests/operations/test_calibrate.py tests/lib/test_cwl.py -k "calibrate"
python3 -m pytest tests/lib/test_cwl.py -k "image_workflow"
python3 -m pytest tests/test_process.py tests/lib/test_field.py tests/lib/test_strategy.py
python3 -m pytest tests/operations/test_calibrate.py tests/operations/test_predict.py tests/operations/test_image.py
python3 -m pytest tests/lib/test_cwl.py
python3 -m pytest -m "integration" tests/integration
tox -e lint
tox
```

Focused DI integration is green. Run the broader integration marker before a
release cut, since this plan intentionally used the focused DI file plus unit and
CWL matrices as the completion gate for this implementation pass.
