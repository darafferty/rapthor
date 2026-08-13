# Feature Plan

## Current Status

The optional failure policy for high DP3 calibration out-of-memory risk is
implemented and verified. No feature in this plan remains in progress.

## Completed Feature: DP3 Calibration Memory Checks

Status: Complete

Rapthor now performs advisory DP3 calibration memory checks before processing
and again after each calibration cycle's facet count and solve intervals have
been resolved.

### Delivered

- [x] Added a pure current-DP3 peak-memory estimator in
  `rapthor/lib/cluster.py`, using decimal GB and the 80-byte-per-sample model.
- [x] Added input validation and structured component results for the estimator.
- [x] Centralized calibration solve metadata and DI/DD solve resolution in
  `rapthor/lib/calibration.py`, shared by calibration and memory assessment.
- [x] Added pre-flight checks using each strategy step's `max_directions` upper
  bound.
- [x] Added resolved checks using the actual calibration patch count and rounded
  observation solve intervals.
- [x] Evaluated every enabled solve and observation while reporting only the
  largest independent workflow-task estimate.
- [x] Compared estimates with configured `mem_per_node_gb`, falling back to
  memory available on the current machine when it is zero.
- [x] Kept all checks advisory: estimation, metadata, or memory-probe failures
  are logged without interrupting processing or changing calibration settings.
- [x] Added INFO/WARNING summaries and DEBUG calculation terms, including the
  cycle, solve, observation, effective direction count, `max_directions`, peak
  estimate, capacity source, and headroom or overage.
- [x] Documented the behavior and assumptions in `docs/source/parset.rst`.
- [x] Added generic Rapthor unit and orchestration coverage without SKA-specific
  fixtures or internet requirements.

### Verified

- Focused estimator, solve-resolution, process, and calibration tests:
  `114 passed`.
- Ruff checks for all files touched by the feature: passed.
- Full non-integration suite: not rerun during this plan update; the sequential
  container run was stopped after making slow progress with no failures.
- Integration coverage was not required because the feature does not alter CWL
  inputs or calibration execution order.

## Completed Feature: Optional Failure on High Calibration OOM Risk

Status: Complete

### Objective

Add a boolean parset option under `[cluster]`, provisionally named
`fail_on_calibration_oom_risk`, with a default value of `False`. When enabled,
Rapthor will raise an error and stop before running an affected operation if a
DP3 calibration memory estimate is strictly greater than the applicable memory
limit. The default will preserve the current advisory-only behavior.

The existing command-line entry point converts an uncaught exception into exit
code 1, so the domain and orchestration code should raise an informative
exception rather than call `sys.exit()` directly.

### Behavior Contract

- Apply the policy to both existing check stages:
  - the pre-flight estimate based on each strategy step's `max_directions`;
  - the per-cycle estimate based on the resolved facet count and solution
    intervals.
- Treat `estimate > limit` as high OOM risk.
- Treat `estimate == limit` as fitting, consistent with current behavior.
- With the option disabled, log the existing warning and continue.
- With the option enabled, raise before starting the affected pipeline
  operation.
- Continue using a positive `mem_per_node_gb` as the limit. If it is zero, use
  memory available on the machine running Rapthor.
- If capacity cannot be determined, continue with an advisory warning because
  high OOM risk has not been established.
- If estimation or metadata resolution fails, retain the current advisory
  failure behavior. The strict flag applies only to a successfully calculated
  estimate that exceeds a known limit.
- Include the stage, cycle, solve, observation, estimate, limit, and overage in
  the failure message, together with the option name so the response is
  actionable.
- Document that Slurm users should set `mem_per_node_gb`, because memory on the
  machine running Rapthor may not represent memory on a compute node.

### Scope and Non-goals

- No CWL changes are expected; this is a Python orchestration and configuration
  policy.
- Do not alter the DP3 memory formula, automatic calibration settings, or
  resource requests.
- Do not make unknown capacity or failed estimation fatal.
- Do not introduce process termination into library code.
- Preserve operation ordering and the current default behavior.

### TDD Implementation Record

Implemented in small red-green-refactor increments, running the narrowest
relevant tests after each increment before broadening coverage.

1. **[x] Specify parset behavior with failing tests**

   - Add coverage in `tests/lib/test_parset.py` proving that the new option
     defaults to `False` and an explicit `True` is parsed as a boolean.
   - Update the minimal and complete reference parset dictionaries so their
     exact-content tests describe the new public configuration contract.

2. **[x] Specify memory-policy behavior with failing unit tests**

   - Move checker-focused tests from `tests/test_process.py` into a dedicated
     `tests/lib/test_calibration_memory.py`, keeping process tests focused on
     orchestration.
   - Cover over-limit behavior with the flag disabled and enabled.
   - Cover below-limit and exact-limit behavior in strict mode.
   - Assert the contents of the raised error and existing log messages.
   - Cover unknown capacity and estimation failures in strict mode to confirm
     that they remain advisory.
   - Exercise both pre-flight and resolved inputs.

3. **[x] Specify orchestration behavior with failing tests**

   - Verify that a pre-flight risk failure prevents initial imaging and all
     processing cycles.
   - Verify that a resolved-cycle risk failure occurs after `field.update()`
     and observation-parameter resolution but before `Predict` or `Calibrate`.
   - Verify that disabled mode retains the existing operation order.

4. **[x] Add the public configuration option**

   - Add `fail_on_calibration_oom_risk = False` and an explanatory comment to
     `rapthor/settings/defaults.parset`.
   - Mirror the option in `rapthor/settings/defaults_skalow.parset` and
     `rapthor/settings/defaults.json`.
   - Update complete parset examples and expected dictionaries used by tests.
   - Keep the option in the `[cluster]` section because it controls behavior
     relative to the per-node memory limit.

5. **[x] Separate assessment, presentation, and enforcement**

   - Add a dedicated `CalibrationMemoryRiskError`, derived from `RuntimeError`.
   - Represent the capacity comparison once, either with a small immutable
     assessment object or a focused pure helper, so logging and enforcement
     cannot disagree about whether the estimate exceeds the limit.
   - Keep estimation free of policy decisions.
   - Keep message construction centralized and reuse it for logging and the
     exception.
   - Ensure the intentional risk exception is raised outside the broad
     advisory exception boundary; otherwise `except Exception` would swallow
     it and allow processing to continue.

6. **[x] Move observation setup to orchestration**

   - In `run_steps()`, call `field.set_obs_parameters()` after cycle settings
     and `generate_screens` have been resolved, and before the per-cycle memory
     check.
   - Remove this state mutation from `check_calibration_memory()` so the checker
     consumes resolved state and its advisory exception handling cannot hide a
     required setup failure.
   - Retain the later call in `Calibrate.set_input_parameters()`. A preceding DI
     prediction can change observation filenames, so calibration inputs still
     need to be refreshed.

7. **[x] Add lightweight CLI integration coverage**

   - Complete `tests/integration/test_oom_risk_logic.py` using an existing small
     Measurement Set, strict mode, and an intentionally tiny positive
     `mem_per_node_gb`.
   - Assert a non-zero `rapthor` exit code and an actionable OOM-risk message.
   - Assert that no calibration operation starts. The pre-flight failure should
     make this test independent of DP3 or WSClean execution.

8. **[x] Document the feature**

   - Extend `docs/source/parset.rst` with the new option, its default, both
     enforcement stages, the strict comparison boundary, exit behavior,
     unknown-capacity behavior, and Slurm guidance.
   - Update any checked-in parset examples intended to enumerate all cluster
     options.

### Verification

The implementation was verified in the project's development container:

- Parset tests: `37 passed`.
- Focused estimator, policy, process, and calibration tests: `106 passed`.
- Focused CLI integration test: `1 passed`; it stopped during pre-flight without
  invoking a radio-astronomy workflow.
- Direct sequential field tests: `31 passed`.
- Direct remaining non-integration suite: `1620 passed, 1 skipped, 2 xfailed`.
- Packaged `tox -e py312` run: `31 passed` sequentially, followed by
  `1620 passed, 1 skipped, 2 xfailed` in parallel.
- `tox -e lint`: passed.
- JSON validation and `git diff --check`: passed.

The plain `tox` command completed its lint and format environments but the
current tox version interpreted the existing `py3{9,10,11,12,13}` env-list
entry literally, causing that brace expression to leak into the package install
path. Running the equivalent concrete `py312` environment avoided this
pre-existing configuration compatibility issue and passed as reported above.

### Completion Criteria

- [x] The new option is accepted from all supported default parsets and defaults to
  advisory behavior.
- [x] Strict mode fails at either check stage only when a known limit is exceeded.
- [x] The failure is raised before any affected operation begins and produces CLI
  exit code 1.
- [x] Disabled mode, exact-limit behavior, unknown capacity, and estimation failure
  retain their documented behavior.
- [x] Unit, orchestration, parset, CLI integration, and lint checks pass.
- [x] User documentation and configuration examples describe the final option name
  and semantics.

### Worktree Reconciliation

The existing integration-test work was reviewed before implementation. The
malformed duplicate fixture in `tests/integration/conftest.py` was removed while
preserving its valid existing counterpart, leaving no diff in that file. The
untracked `tests/integration/test_oom_risk_logic.py` stub was completed as the
focused CLI integration test described above.
