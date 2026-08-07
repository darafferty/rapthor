# Feature Plan

## Current Status

No feature is currently planned. The DP3 calibration memory-check feature is
complete, and this file is ready to be populated for the next feature.

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

## Next Feature

Status: Not defined

Replace this section when the next feature is selected. Capture:

1. Objective and user-visible behavior.
2. Scope, assumptions, and explicit non-goals.
3. Implementation tasks and affected layers (Python, CWL, settings, or docs).
4. Focused and broader test tasks.
5. Verification commands and environment requirements.
6. Completion criteria.
