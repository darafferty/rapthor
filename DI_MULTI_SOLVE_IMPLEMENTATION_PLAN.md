# DI Multi-Solve Implementation Plan

## Goal

Implement direction-independent (DI) calibration for the solve types listed in
`calibration_strategy["di"]`:

- `full_jones`
- `fast_phase`
- `medium_phase`
- `slow_gains`

The implementation must be explicit about which combinations are supported in
this pass, must keep the existing full-Jones imaging contract intact, and must
produce CWL inputs that match the real `ddecal_solve.cwl` and
`combine_h5parms.cwl` interfaces.


## Current Status

Completed:

- Step 1 strategy validation guardrails are implemented in
  `rapthor/lib/strategy.py::_validate_calibrate_strategy()`.
- Strategy tests now cover accepted and rejected DI/DD combinations.
- The misplaced `_do_calibrate_mode()` mixed-full-Jones xfail was removed;
  `_do_calibrate_mode()` remains a simple mode gate.
- Remaining plan-specified test scaffolding has been added:
  - DI dynamic calibrate-input tests are present as strict xfails until the DI
    mapping implementation lands.
  - DI combine-mode tests are present as strict xfails until combine mapping
    lands.
  - DI workflow rendering tests for every supported solve set are present as
    strict xfails until `calibrate_di_pipeline.cwl` is rewritten.
  - DI medium-phase integration coverage is present as a strict xfail until the
    multi-solve DI workflow is implemented.
  - CLI-level integration guards for unsupported DI combinations are passing.
  - `combine_h5parms.py` log parsing is implemented for integration assertions.
  - Image applycal missing-file and generic-DI-h5parm tests are present.

Verified so far:

- `pytest tests/lib/test_strategy.py tests/test_process.py -q` -> `72 passed`
- `pytest tests/integration/test_utils.py -q` -> `4 passed`
- Targeted remaining-test scaffold run -> `21 passed, 12 xfailed`
- `pytest tests/integration/test_di_calibration.py::test_rapthor_rejects_unsupported_di_strategy_combinations -q` -> `2 passed`

Known current blocker:

- `tests/operations/test_image.py::TestImage::test_build_applycal_steps` still
  has two failures because `Image._build_applycal_steps()` currently forces DD
  solves before DI solves. Step 2 must decide and implement the plan's preferred
  user-provided strategy order preservation.

## Current Code Facts

- `process.py` already gates calibration execution by mode. If a user-provided
  `calibration_strategy` contains `"di": []`, `Predict("di", ...)` and
  `Calibrate("di", ...)` are skipped.
- If the whole `calibration_strategy` key is omitted or set to `None`,
  `Field.set_calibration_strategy()` falls back to the legacy defaults. In that
  legacy path, `do_fulljones_solve=True` still requests DI `full_jones`.
- `calibrate_di_pipeline.cwl` is currently full-Jones-only.
- `prepare_imaging_data.cwl` has two solution-file inputs:
  - `h5parm` for generic phase/amplitude corrections such as `fastphase`,
    `mediumphase`, and `slowgain`
  - `fulljones_h5parm` for `fulljones`
- `Predict` and `Image` currently expose only one generic non-full-Jones h5parm
  slot through `field.h5parm_filename`.

## Supported Scope For This Pass

Supported:

- DI-only `fast_phase`
- DI-only `medium_phase`
- DI-only `slow_gains`
- DI-only `fast_phase + medium_phase`
- DI-only `fast_phase + medium_phase + slow_gains`
- DI `full_jones`, including after DD calibration, because full-Jones uses the
  separate `field.fulljones_h5parm_filename` imaging input.
- No DI when a user-provided strategy has `"di": []` or omits the `di` key.

Rejected for this pass:

- Any DI list that mixes `full_jones` with non-full-Jones DI solves.
- Any cycle that combines DD solves with non-full-Jones DI solves.

Rationale for the second rejection: DD and non-full-Jones DI both need the
single generic `h5parm` input during prediction/imaging. Supporting them
together requires either merging DD and DI solution streams into one h5parm with
a well-defined applycal contract, or extending the prediction/imaging CWL with
separate applycal stages. That is a larger contract change and should not be
hidden inside this implementation.

## Solve Mapping

Use one mapping in `rapthor/operations/calibrate.py`.

```python
DI_SOLVE_TYPE_ORDER = ["fast_phase", "medium_phase", "slow_gains", "full_jones"]

DI_SOLVE_TYPE_CONFIG = {
    "fast_phase": {
        "mode": "scalarphase",
        "solint_key": "solint_fast_timestep",
        "nchan_key": "solint_fast_freqstep",
        "chunk_output_stem": "fast_phase_di",
        "collected_h5parm": "fast_phase_di.h5parm",
        "field_attr": "fast_phases_h5parm_filename",
        "final_filename": "field-solutions-fast-phase.h5",
        "initial_h5_key": "fast_phases_h5parm_filename",
        "initial_soltab": "[phase000]",
        "plot_soltypes": ["phase"],
        "needs_process_gains": False,
    },
    "medium_phase": {
        "mode": "scalarphase",
        "solint_key": "solint_medium_timestep",
        "nchan_key": "solint_medium_freqstep",
        "chunk_output_stem": "medium1_phase_di",
        "collected_h5parm": "medium1_phase_di.h5parm",
        "field_attr": "medium1_phases_h5parm_filename",
        "final_filename": "field-solutions-medium1-phase.h5",
        "initial_h5_key": "medium1_phases_h5parm_filename",
        "initial_soltab": "[phase000]",
        "plot_soltypes": ["phase"],
        "needs_process_gains": False,
    },
    "slow_gains": {
        "mode": "diagonal",
        "solint_key": "solint_slow_timestep",
        "nchan_key": "solint_slow_freqstep",
        "chunk_output_stem": "slow_gains_di",
        "collected_h5parm": "slow_gains_di.h5parm",
        "processed_h5parm": "slow_gains_di_processed.h5parm",
        "field_attr": "slow_gains_h5parm_filename",
        "final_filename": "field-solutions-slow-gain.h5",
        "initial_h5_key": "slow_gains_h5parm_filename",
        "initial_soltab": "[phase000,amplitude000]",
        "plot_soltypes": ["phase", "amplitude"],
        "needs_process_gains": True,
    },
    "full_jones": {
        "mode": "fulljones",
        "solint_key": "solint_fulljones_timestep",
        "nchan_key": "solint_fulljones_freqstep",
        "chunk_output_stem": "fulljones_gain",
        "collected_h5parm": "fulljones_gains.h5",
        "processed_h5parm": "fulljones_gains_processed.h5",
        "field_attr": "fulljones_h5parm_filename",
        "final_filename": "fulljones-solutions.h5",
        "initial_h5_key": None,
        "initial_soltab": "[phase000,amplitude000]",
        "plot_soltypes": ["phase", "amplitude"],
        "needs_process_gains": True,
    },
}
```

## Validation Changes

Add validation in `rapthor/lib/strategy.py::_validate_calibrate_strategy()`:

- Keep the existing unknown-mode and unknown-solve checks.
- Reject DI `full_jones` mixed with any of `fast_phase`, `medium_phase`, or
  `slow_gains`.
- Reject a cycle with `calibration_strategy["dd"]` non-empty and
  `calibration_strategy["di"]` containing any non-full-Jones solve.

Do not put this responsibility in `_do_calibrate_mode()`. That helper should
only return booleans for modes that should run.

## Calibrate Operation Changes

Location: `rapthor/operations/calibrate.py`

### Add Helpers

- `_get_di_solves()`:
  - Reads `field.calibration_strategy.get("di", [])`.
  - Filters solves into `DI_SOLVE_TYPE_ORDER`.
  - Returns an empty list when no DI solve is requested.
  - Should be used by `set_parset_parameters()`, `set_input_parameters()`, and
    `finalize()` so direct unit-test calls do not depend on method call order.

- `_get_di_final_h5parm()`:
  - `["full_jones"]` -> processed full-Jones h5parm.
  - Single non-full-Jones solve -> collected or processed h5parm for that solve.
  - `["fast_phase", "medium_phase"]` -> `di_fast_medium.h5parm`.
  - `["fast_phase", "medium_phase", "slow_gains"]` -> `di_solutions.h5`.

- `_get_di_combine_plan()`:
  - `["fast_phase"]`, `["medium_phase"]`, `["slow_gains"]`: no combine.
  - `["fast_phase", "medium_phase"]`: one combine with mode `p1p2_scalar`.
  - `["fast_phase", "medium_phase", "slow_gains"]`: first combine
    `p1p2_scalar`, then second combine using
    `p1p2a2_diagonal` when `field.apply_diagonal_solutions` is true, otherwise
    `p1p2a2_scalar`.
  - `["full_jones"]`: no generic DI combine.

### `set_parset_parameters()`

For DI mode:

- Set `self.di_solves`.
- Set template parameters needed to render the DI workflow:
  - `di_solves`
  - `nr_di_solves`
  - `has_slow_gains`
  - `is_full_jones`
  - `needs_combine_fast_medium`
  - `needs_combine_slow`
- Keep `max_cores` and `rapthor_pipeline_dir`.

### `set_input_parameters()`

For DI mode:

- Return early only if `self.di_solves` is empty. In normal execution this path
  is skipped by `process.py`, but the guard keeps direct tests safe.
- Build generic workflow inputs:
  - `timechunk_filename`
  - `data_colname = "DATA"`
  - `starttime`
  - `ntimes`
  - `steps = "[solve1]"`, `"[solve1,solve2]"`, or `"[solve1,solve2,solve3]"`
  - solver controls: `maxiter`, `llssolver`, `propagatesolutions`,
    `solveralgorithm`, `stepsize`, `stepsigma`, `tolerance`, `uvlambdamin`,
    `solverlbfgs_dof`, `solverlbfgs_iter`, `solverlbfgs_minibatches`,
    `correctfreqsmearing`, `correcttimesmearing`, `max_threads`
  - processing controls for `process_gains.cwl`: `max_normalization_delta`,
    `scale_normalization_delta`, `phase_center_ra`, `phase_center_dec`
- For each active DI solve, assign consecutive `solveN_*` inputs:
  - `solveN_mode`
  - `solveN_h5parm` as an array, e.g. `fast_phase_di_0.h5parm`
  - `solveN_solint`
  - `solveN_nchan`
  - `solveN_initialsolutions_h5parm`
  - `solveN_initialsolutions_soltab`
  - `solveN_llssolver`, `solveN_maxiter`, `solveN_propagatesolutions`,
    `solveN_solveralgorithm`, `solveN_solverlbfgs_*`, `solveN_stepsize`,
    `solveN_stepsigma`, `solveN_tolerance`, `solveN_uvlambdamin`
- For multi-solve DI, wire model reuse like the DD workflow does:
  - `solve1_keepmodel = "True"` when there is more than one active solve.
  - `solveN_reusemodel = "[solve1.*]"` for `N > 1`.
- Add collected/processed/final h5parm names and combine mode inputs generated
  by `_get_di_combine_plan()`.

### `finalize()`

For DI mode:

- If `self.di_solves == ["full_jones"]`:
  - Copy the processed full-Jones h5parm to
    `solutions/calibrate_di_N/fulljones-solutions.h5`.
  - Set `field.fulljones_h5parm_filename`.
  - Do not overwrite `field.h5parm_filename`, because DD solutions may already
    occupy the generic h5parm slot.
- For non-full-Jones DI:
  - Copy the final generic DI h5parm to
    `solutions/calibrate_di_N/field-solutions-di.h5`.
  - Set `field.h5parm_filename`.
  - Copy per-solve collected/processed h5parms to their solve-specific field
    attributes for reuse as initial solutions in later cycles.
- Call `field.scan_h5parms()` after copying so `apply_amplitudes` and
  `apply_fulljones` reflect the files that are now present.
- Compute flagged fraction from `field.fulljones_h5parm_filename` for
  full-Jones and from `field.h5parm_filename` for non-full-Jones DI.

## DI CWL Workflow Changes

Location: `rapthor/pipeline/parsets/calibrate_di_pipeline.cwl`

Replace the current full-Jones-only workflow with a Jinja-rendered workflow that
uses the real `ddecal_solve.cwl` contract.

### Required Design

- Render inputs for the active solve count (`solve1`, `solve2`, `solve3`).
- Scatter all active per-chunk arrays together:
  - always: `msin`, `starttime`, `ntimes`, `solve1_h5parm`,
    `solve1_solint`, `solve1_nchan`
  - if `nr_di_solves >= 2`: also scatter `solve2_h5parm`,
    `solve2_solint`, `solve2_nchan`
  - if `nr_di_solves >= 3`: also scatter `solve3_h5parm`,
    `solve3_solint`, `solve3_nchan`
- Collect each active `ddecal_solve.cwl` output separately:
  - `output_h5parm1` -> collected solve1 h5parm
  - `output_h5parm2` -> collected solve2 h5parm
  - `output_h5parm3` -> collected solve3 h5parm
- Process amplitude-producing solves:
  - `slow_gains`: `process_gains.cwl` with flag/smooth settings matching the
    DD slow-gain path.
  - `full_jones`: `process_gains.cwl` with the current full-Jones normalization
    behavior.
- Plot every requested solve's expected soltabs.
- Use `combine_h5parms.cwl` only when more than one non-full-Jones DI solve
  needs combining.
- Do not reference undefined steps. The old sketch names `collect_fast`,
  `collect_medium`, and `process_slow`; the actual workflow must use concrete
  step IDs rendered from the active solve list.

### Combine Cases

- `fast_phase + medium_phase`:
  - collect fast
  - collect medium
  - combine fast + medium with `p1p2_scalar`
  - final generic DI h5parm is the combine output.
- `fast_phase + medium_phase + slow_gains`:
  - collect fast
  - collect medium
  - collect and process slow
  - combine fast + medium with `p1p2_scalar`
  - combine that result with processed slow using `solution_combine_mode`
  - final generic DI h5parm is the second combine output.
- Single non-full-Jones solve:
  - no combine
  - final generic DI h5parm is the collected/processed solve output.
- `full_jones`:
  - no generic DI combine
  - final full-Jones h5parm is the processed full-Jones output.

### `combine_h5parms.cwl` Contract

Every combine step must pass all required inputs:

- `inh5parm1`
- `inh5parm2`
- `outh5parm`
- `mode`
- `reweight`
- `calibrator_names`
- `calibrator_fluxes`

Use only modes supported by `combine_h5parms.py`:

- `p1p2_scalar`
- `p1p2a2_scalar`
- `p1p2a2_diagonal`

## Image And Predict Changes

Location: `rapthor/operations/image.py`

- Preserve the order of the user-provided strategy by iterating
  `field.calibration_strategy.items()` instead of forcing `("dd", "di")`.
- Keep the missing-file robustness:
  - skip generic applycal steps when `field.h5parm_filename` is absent or does
    not exist.
  - skip `fulljones` when `field.fulljones_h5parm_filename` is absent or does
    not exist.
- Keep `fulljones_h5parm` separate from generic `h5parm`.

Location: `rapthor/operations/predict.py`

- For this pass, no new full-Jones applycal is required in `Predict("di", ...)`
  because supported DD + DI overlap is limited to DD + full-Jones and the
  full-Jones solve happens after DI prediction.
- If future work supports non-full-Jones DI together with DD, update `Predict`
  and `Image` together so both operation types apply the same solution streams.

## Test Plan

### Unit Tests

`tests/lib/test_strategy.py`

Status: done.

- Unknown mode still raises.
- Unknown solve still raises.
- DI `["full_jones", "fast_phase"]` raises.
- DI `["full_jones", "slow_gains"]` raises.
- DD non-empty plus DI non-full-Jones raises.
- DD non-empty plus DI `["full_jones"]` remains valid.
- DI-only non-full-Jones combinations remain valid.

`tests/test_process.py`

Status: done.

- `_do_calibrate_mode()` boolean tests remain.
- The mixed-full-Jones validation xfail was removed from this file; validation
  now lives in `tests/lib/test_strategy.py`.

`tests/operations/test_calibrate.py`

Status: present as strict xfail scaffolding.

- Convert the DI dynamic mapping xfails into passing tests after implementation.
- Covered solve sets:
  - `fast_phase`
  - `medium_phase`
  - `slow_gains`
  - `fast_phase + medium_phase`
  - `fast_phase + medium_phase + slow_gains`
  - `full_jones`
- Asserted contract:
  - `steps`
  - `solveN_mode`
  - `solveN_h5parm`
  - `solveN_solint`
  - `solveN_nchan`
  - `solveN_initialsolutions_*`
  - `solveN_keepmodel` / `solveN_reusemodel` for multi-solve cases
  - combine mode names where applicable.
- Still to add when implementation exposes it: final h5parm name assertions.

`tests/operations/test_image.py`

Status: partially done; step 2 remains.

- Missing-file tests are present.
- A DI non-full-Jones generic `h5parm` test is present.
- Still to do in step 2: fix applycal order expectations and implementation to
  match the chosen strategy order behavior. The preferred behavior for this pass
  remains user-provided order preservation.

`tests/scripts/test_combine_h5parms.py`

Status: done.

- Supported-mode dispatch tests are present.
- Unknown-mode rejection is present.

### CWL Rendering Tests

Status: present as strict xfail scaffolding.

`tests/lib/test_cwl.py` now includes rendering coverage for:

- `fast_phase`
- `medium_phase`
- `slow_gains`
- `fast_phase + medium_phase`
- `fast_phase + medium_phase + slow_gains`
- `full_jones`

These tests currently xfail because `calibrate_di_pipeline.cwl` is still
full-Jones-only. They should catch missing workflow inputs, bad scatter lists,
and undefined step references when the workflow rewrite lands.

### Integration Tests

Status: mostly present, with unsupported-strategy guards passing and
not-yet-implemented solve paths marked strict xfail where needed.

Coverage now includes:

- DI `fast_phase`
- DI `medium_phase` as strict xfail until implementation lands
- DI `slow_gains`
- DI `fast_phase + medium_phase`
- DI `fast_phase + medium_phase + slow_gains`
- DI `full_jones`
- No calibration with `{"di": [], "dd": []}`
- Rejection of unsupported mixed DI full-Jones/non-full-Jones strategies
- Rejection of unsupported DD + non-full-Jones DI strategies

For combine-chain assertions:

Status: parser and strict xfail tests are present.

- `parse_combine_h5parms_args_from_log()` parses `$ combine_h5parms.py ...`
  command lines and reads the mode from the fourth positional argument.
- Combine-chain tests use this parser and assert:
  - `fast_phase + medium_phase` includes `p1p2_scalar`.
  - `fast_phase + medium_phase + slow_gains` includes `p1p2_scalar` followed by
    `p1p2a2_scalar` or `p1p2a2_diagonal`.
- Remove strict xfails after DI combine steps exist in the workflow.

For imaging assertions:

- Parse `prepare_imaging_data.cwl` DP3 logs.
- DI `fast_phase` should include `applycal.steps=[fastphase]`.
- DI `medium_phase` should include `applycal.steps=[mediumphase]`.
- DI `slow_gains` should include `applycal.steps=[slowgain]` when amplitudes
  are present.
- DI `full_jones` should include `applycal.steps=[fulljones]` and pass
  `applycal.fulljones.parmdb`.

## Implementation Order

1. Done: Add strategy validation guardrails and tests.
2. Next: Fix `Image._build_applycal_steps()` ordering and keep the
   missing-file behavior tests passing.
3. Add DI solve mapping helpers in `calibrate.py`.
4. Rewrite `calibrate_di_pipeline.cwl` with active-solve Jinja rendering,
   correct scatter lists, per-solve collection, processing, plotting, and
   combine steps.
5. Update `Calibrate.set_input_parameters()` and `finalize()` for DI
   non-full-Jones and full-Jones paths.
6. Done as strict xfail scaffolding: Add CWL rendering tests for every
   supported DI combination. Later, remove xfails when the workflow rewrite
   lands.
7. Partially done as strict xfail scaffolding: DI unit tests exist. Later,
   turn DI unit xfails into normal passing tests when implementation lands.
8. Partially done: combine-log parsing exists and combine-chain tests use it.
   Later, remove combine-chain xfails when DI combine steps are implemented.
9. Run targeted tests, then the DI integration suite after each implementation
   step.

## Success Criteria

- Empty user-provided DI lists do not run DI prediction/calibration.
- Legacy defaults still work when no `calibration_strategy` is provided.
- Full-Jones DI keeps using `field.fulljones_h5parm_filename`.
- Non-full-Jones DI writes the generic DI h5parm to `field.h5parm_filename`.
- Unsupported mixed strategies fail during strategy validation with clear
  messages.
- Every supported DI solve combination renders valid CWL.
- Every supported DI solve combination passes unit and integration tests.
- DI multi-solve outputs are combined through `combine_h5parms.cwl` before
  imaging when combination is required.
