# Post-Merge Development Plan

Updated: 2026-10-14.

## Scope and Review Baseline

This plan covers the work needed to release the first Prefect/Dask version of
Rapthor from `main`, and the improvements that follow it. Implement each task
on `main` in a focused branch and merge request. The normal command remains
`rapthor input.parset`; temporary `rapthor3` comparison overlays do not belong
in production branches.

- **Part A: missing master functionality and bug fixes.** These are identified
  gaps, verified review findings, missing regression coverage, and explicit
  compatibility decisions. Part A is organised into workstreams that touch
  disjoint code so that different people or agents can work in parallel.
- **Part B: future improvements.** These extend or improve the resulting
  Prefect/Dask pipeline; they are not missing ports from master.

### Branch state at the 2026-10-14 recheck

| Branch | Revision | Notes |
| --- | --- | --- |
| `main` | `d0e7e148` (review baseline); `aff26023` since 2026-10-15 | Squashed migration `c6196cb4` (GEC-468) plus `d0e7e148` (reduced frequency-BDA default, closes review task P06). `aff26023` adds the `use_array_beam` option (see P2-PORT-01); it touches no open P0 task. |
| `master` | `76a7ff2f` | Tagged `v2.2rc1` at `b9c98086`. 69 commits since the common ancestor `2e21be62`. |

Master gained five commits after the previous plan baseline `ff1f29d9`. Three
are CWL-only (Docker tag churn in CWL steps, `76a7ff2f` Toil tmpdir option) and
do not apply. `b9c98086` is release preparation. `9fc97dc3` (GEC-656, sky model
download error handling) is already present on `main` in
`rapthor/execution/image/diagnostic_calculation.py:safe_load_skymodel`, which is
the same plain function returning `None` on `OSError`; only master's new tests
have no counterpart. No further master changes need porting beyond the tasks
below.

Because master is at `v2.2rc1`, the first Prefect release is a major version.
The [upgrade guide](docs/source/upgrading.rst) is the migration document and
must record every behaviour change accepted in Part A.

### Verification status

Every Part A finding was rechecked by source inspection on `main` at `d0e7e148`
on 2026-10-14 (see the verification record at the end). All remain present
except review task P06 (BDA default), which has moved to the "already covered" table. One correction: the
P0-BUG-02 payload gap is narrower than previously described. The stand-alone
`applycal` step is wired; it is the DDECal-internal normalization on the
slow-gain solve that is missing.

The earlier audit compared `master` at `ff1f29d9` with migration HEAD
`b5884ade`, and the 2026-10-09 follow-up reconciled the two MR-473 reviews
([Extra](https://git.astron.nl/RD/rapthor/-/blob/33c88d86/reviews/MR-473/Claude-Opus-Extra_pre-merge-review.md),
[Medium](https://git.astron.nl/RD/rapthor/-/blob/33c88d86/reviews/MR-473/Claude-Opus-Medium_pre-merge-review.md)).
The review IDs (R01–R16, E01–E05) are mapped to the current task IDs in the
[task identifier table](#task-identifiers) below.
These are source reviews with isolated real-tool probes, not a new scientific
equivalence run.

## Task Identifiers

Task IDs have the form `P<priority>-<category>-<NN>`, so sorting by ID sorts
by priority and the category is visible in issue titles and branch names.

| Priority | Meaning |
| --- | --- |
| **P0** | Release blocker. Wrong results on default paths, or restart unreliable. |
| **P1** | Fix before release if the gated feature is in scope (cluster, dashboard, multi-chunk, standalone image, SKA-Low). |
| **P2** | Can ship with a documented limitation; schedule soon after release. |
| **P3** | Later. Lower user impact, or blocked on external tools or cluster access. |

| Category | Meaning |
| --- | --- |
| **BUG** | Incorrect or unexpectedly changed behaviour on `main`. |
| **PORT** | Functionality present on master 2.2 but missing on `main`. |
| **DEC** | A compatibility or scientific decision that must be recorded before the item can close. |
| **TEST** | Regression coverage, CI, or validation work with no production code change of its own. |

Numbers restart within each priority/category pair and follow the recommended
order. The MR-473 review IDs are kept for traceability: R-numbers are Extra
and Medium review findings, E-numbers are Extra review findings, P/D/L-numbers
were the previous plan's port, decision and limitation tasks.

| Task ID | Review ID | Title | Workstream | Effort |
| --- | --- | --- | --- | --- |
| P0-BUG-01 | R16 | Prediction normalization never reaches written visibilities | WS-A | S / Medium |
| P0-BUG-09 | new (2026-10-15) | DD calibration with a pre-apply step rewrites its input MS in place | WS-A | S / Medium |
| P0-BUG-02 | R03 | Slow-gain normalization and WSClean pre-application wiring | WS-A | M / High |
| P0-BUG-03 | R11 | DD and DI amplitude-application state | WS-B | M / Medium |
| P0-BUG-04 | R02 | Per-cycle semantics for an omitted `calibration_strategy` | WS-C | M / Medium |
| P0-BUG-05 | R12 | Partial-product restart checks | WS-D | L / High |
| P0-BUG-06 | E01 | Combined phases applied twice with `dde_method = single` | WS-B | S / Low |
| P0-BUG-07 | E03 | Final strategy step aliases the last selfcal step | WS-C | S / Low |
| P0-BUG-08 | R13 | Ineffective `prefect_retries` option | WS-D | S / Low |
| P1-BUG-01 | R07 | Command descendants survive cancellation | WS-E | M / Medium |
| P1-BUG-02 | R06 | Slurm guard bypass and worker-count default | WS-E | S / Low |
| P1-BUG-03 | R04 | Unbounded artifact publication | WS-E | M / Medium |
| P1-BUG-04 | R15 | Predict output discovery ignores the chunk | WS-D | S / Low |
| P1-DEC-01 | E02 | Discarded medium-phase result in phase-only DD cycles | WS-B | S code / Medium |
| P1-DEC-02 | D04 | SKA-Low configuration interface | WS-G | M / Medium |
| P1-PORT-01 | P10 | Standalone image NumPy build/runtime | WS-G | M / Medium |
| P1-TEST-01 | R05 | Shared Prefect API and MPI placement on external Dask | WS-E | L / High |
| P2-BUG-01 | R01 | Solution directions in image-only cycles | WS-B | L / High |
| P2-BUG-02 | R14 | Calibration plot output ownership | WS-D | S / Low |
| P2-BUG-03 | E04 | Repeated final cycles in reset discovery | WS-D | S / Low |
| P2-BUG-04 | R08 | Reset of scratch workspaces and log attempts | WS-D | M / Medium |
| P2-BUG-05 | R10 | Astrometry-offset JSON products | WS-D | S / Low |
| P2-BUG-06 | R09 | Empty model mosaics | WS-D | M / Medium |
| P2-PORT-01 | P01 | Full beam during WSClean prediction | WS-F1 | M / Medium |
| P2-PORT-02 | P04 | Normalization metadata for supplied survey catalogs | WS-F2 | S / Low |
| P2-PORT-03 | P03 | Observations with different station sets | WS-F2 | L / High |
| P2-PORT-04 | P05 | Offline catalog-preparation workflow | WS-F2 | M / Medium |
| P2-PORT-05 | P07 | Duplicate console records and ANSI codes in logs | WS-E | M / Medium |
| P2-PORT-06 | P08 | Task log and failure cause in the surfaced exception | WS-E | M / Medium |
| P2-PORT-07 | P09 | Ubuntu 24.04 APT source rewriting | WS-G | S / Low |
| P2-PORT-08 | P11 | Package metadata and compiled test environment | WS-G | S–M / Low |
| P2-DEC-01 | D01 | Supported solve combinations | WS-C | M / Medium |
| P2-DEC-02 | D05 | WSClean prediction disables solve BDA | WS-F1 | M / Medium |
| P2-TEST-01 | E05 | Integration markers on WSClean restoration tests | WS-G | S / Low |
| P2-TEST-02 | P16 | Flags and weights in the synthetic normalization MS | WS-G | S / Low |
| P3-BUG-01 | L01 | Shared-facet I/O tool failure | WS-G | M–L / High |
| P3-PORT-01 | P02 | WSClean model rendering and reordered-data reuse | WS-F1 | L / High |
| P3-PORT-02 | P13 | Docker dependency-version inspection helper | WS-G | S / Low |
| P3-PORT-03 | P17 | Remaining documentation fixes | WS-G | S / Low |
| P3-DEC-01 | D02 | Named facet-region formats | WS-F1 | M / Medium |
| P3-DEC-02 | D03 | SageCal/libdirac source pin | WS-G | M / Medium |
| P3-TEST-01 | P12 | Astron/SKA CI and balanced test execution | WS-G | L / Medium |
| P3-TEST-02 | P14 | Facet-RMS diagnostics through a real run | WS-G | M / Low |
| P3-TEST-03 | P15 | Parallel-gridding integration matrix | WS-G | M / Low |

## Release Readiness

### Release blockers

These change calibration results on default or commonly used paths, or make
restart unreliable. Items marked *regression* behave correctly on master.

| Task | Why it blocks | Regression? | Effort |
| --- | --- | --- | --- |
| P0-BUG-01 | Prediction normalization never reaches the written visibilities; affects every DD subtraction after the default cycle-1 normalization. **Fixed 2026-10-14** (real-DP3 test added). | Yes | S / Medium |
| P0-BUG-02 | Slow-gain solves ignore the normalization h5parm; WSClean prediction drops DI pre-application. Default path after cycle 1. **Fixed 2026-10-15** (real-DP3 test added). | Yes | M / High |
| P0-BUG-09 | A DD calibration chain with the stand-alone `applycal` step and `msout=` makes DP3 write the DI-corrected DATA back into the calibration input MS, which can be an original input MS. Any strategy with DI solves. | No (also on master) | S / Medium |
| P0-BUG-03 | DD amplitudes stop being applied after a DI solve in the same run. **Fixed 2026-10-15.** | Yes | M / Medium |
| P0-BUG-04 | A strategy cycle that omits `calibration_strategy` silently reuses the previous cycle's solves; first-cycle default differs from master. | Yes | M / Medium |
| P0-BUG-05 | Restart accepts partial MS directories as complete and fails on already-compressed images. | New (Toil handled restart) | L / High |
| P0-BUG-06 | `dde_method = single` applies the combined phase correction twice. Cheap to fix. **Fixed 2026-10-15** (real-DP3 test added). | No (also on master) | S / Low |
| P0-BUG-07 | The built-in final step aliases the last selfcal step. Cheap to fix. | No (also on master) | S / Low |
| P0-BUG-08 | `prefect_retries` is accepted but does nothing. Retire or document before release. | New | S / Low |

### Fix before release if the feature is in scope

| Task | Scope it gates | Effort |
| --- | --- | --- |
| P1-BUG-01 | Cancellation on any runner; orphaned DP3/WSClean processes on clusters. | M / Medium |
| P1-BUG-02 | Any Slurm use. Guard bypass and worker-count defaults. | S / Low |
| P1-BUG-03 | Long runs with the Prefect dashboard enabled. | M / Medium |
| P1-BUG-04 | Multi-chunk runs (data fraction 1 on several nodes). | S / Low |
| P1-DEC-01 | Decision on the phase-only DD product; documentation must match either way. | S code, decision needed |
| P1-DEC-02 | SKA-Low users: the removed `defaults_skalow.parset` interface. | M / Medium |
| P1-PORT-01 | Shipping the standalone Docker image. | M / Medium |
| P1-TEST-01 | Multi-node Slurm with a shared Prefect API. Validation, not code. | L / High (needs cluster) |

### Can ship with a documented limitation

All P2 and P3 tasks: bugs P2-BUG-01 to P2-BUG-06, ports P2-PORT-01 to
P2-PORT-08, the P2 decisions and tests, and everything at P3. Each needs a line
in the upgrade guide if it is not fixed. P2-PORT-03 needs an explicit note:
master 2.2 accepts observations with different station sets and `main` raises.

## Effort Scale

Size is the expected calendar effort for one experienced developer including
tests and documentation. Complexity reflects how much the task depends on
scientific judgement, real-tool verification or cross-module contracts.

| Size | Meaning |
| --- | --- |
| S | Up to one day. Localised change, existing tests cover the area. |
| M | One to three days. Several modules or a new test fixture. |
| L | Three to seven days. New contract, real-tool comparison, or several owners. |
| XL | More than a week, or blocked on external resources. |

Complexity: **Low** is mechanical; **Medium** needs design or a real-tool check;
**High** needs a scientific comparison, cluster access or a cross-owner contract.

## Workstreams for Parallel Work

Each workstream owns a disjoint set of files. Tasks inside a workstream share
code and should be done by one person or in sequence. Cross-workstream file
overlaps are listed so that rebases are planned, not discovered.

| Workstream | Tasks (suggested order) | Owned code | Overlaps to coordinate |
| --- | --- | --- | --- |
| **WS-A Normalization and pre-application wiring** | P0-BUG-01, P0-BUG-02, P0-BUG-09 | `rapthor/execution/predict/commands.py`, `rapthor/execution/calibrate/solves.py`, `rapthor/execution/calibrate/commands.py`, `rapthor/operations/calibrate/plan.py`, `Calibrate._build_applycal()` in `rapthor/operations/calibrate/base.py` | WS-F1 also edits `calibrate/commands.py` and `plan.py` (beam mode). Land WS-A first. |
| **WS-B Solution state and application** | P0-BUG-03, P0-BUG-06, P1-DEC-01, P2-BUG-01 | `Field.scan_h5parms()` in `rapthor/lib/field.py`, `Calibrate.finalize()`/`_dd_active_solution()` in `rapthor/operations/calibrate/base.py`, `rapthor/operations/image/plan.py`, `rapthor/operations/image/base.py`, `rapthor/operations/predict.py`, `rapthor/execution/image/commands.py` | WS-A edits other functions in `operations/calibrate/base.py`. WS-C edits `Field.update()`. |
| **WS-C Strategy semantics** | P0-BUG-07, P0-BUG-04, P2-DEC-01 | `rapthor/lib/strategy.py`, `Field.update()`/`set_calibration_strategy()` in `rapthor/lib/field.py`, `docs/source/strategy.rst`, `docs/source/upgrading.rst` | WS-B edits `Field.scan_h5parms()`. |
| **WS-D Restart, reset and output ownership** | P0-BUG-05, P0-BUG-08, P1-BUG-04, P2-BUG-02, P2-BUG-03, P2-BUG-04, P2-BUG-05, P2-BUG-06 | `rapthor/execution/image/preparation.py`, `rapthor/execution/image/outputs.py`, `rapthor/execution/concatenate/flow.py`, `rapthor/execution/mosaic/`, `rapthor/execution/calibrate/collection.py`, `rapthor/execution/predict/flow.py`, `rapthor/modifystate.py`, `Image.finalize()` | WS-E edits `rapthor/execution/pipeline/flow.py` for publication; P0-BUG-08 touches `execution/config.py` (also WS-E). |
| **WS-E Runtime, cluster and diagnostics** | P1-BUG-01, P1-BUG-02, P1-BUG-03, P2-PORT-05, P2-PORT-06, P1-TEST-01 | `rapthor/execution/shell.py`, `rapthor/execution/runtime_bootstrap.py`, `rapthor/execution/slurm.py`, `rapthor/execution/capabilities.py`, `rapthor/execution/config.py`, `rapthor/execution/artifacts.py`, `rapthor/execution/pipeline/flow.py`, `rapthor/_logging.py`, `rapthor/execution/prefect_logging.py`, `docs/source/running.rst` | WS-D's P0-BUG-08 in `config.py`. |
| **WS-F1 WSClean prediction ports** | P2-PORT-01, P2-DEC-02, P3-DEC-01, P3-PORT-01 | `rapthor/execution/calibrate/prediction.py`, `rapthor/execution/calibrate/builders.py`, WSClean command builders in `rapthor/execution/calibrate/commands.py`, `rapthor/execution/regions.py`, `rapthor/lib/parset.py` | Start after WS-A lands in `calibrate/commands.py` and `plan.py`. |
| **WS-F2 Data layout and normalization inputs** | P2-PORT-02, P2-PORT-03, P2-PORT-04 | `rapthor/execution/image/flux_normalization.py`, station handling in `rapthor/lib/field.py`, `rapthor/execution/image/preparation.py` (MS list), `rapthor/execution/image/builders.py`, `pyproject.toml` entry point | P2-PORT-03 edits `preparation.py` after WS-D's P0-BUG-05. |
| **WS-G Build, packaging, CI, tests and docs** | P2-PORT-07, P1-PORT-01, P2-PORT-08, P2-TEST-01, P3-TEST-01, P3-PORT-02, P3-TEST-02, P3-TEST-03, P2-TEST-02, P3-PORT-03, P3-DEC-02, P1-DEC-02, P3-BUG-01 | `Docker/`, `ci/`, `.gitlab-ci.yml`, `pyproject.toml`, `tests/`, `docs/` | None in production code except `np.NaN` in `execution/predict/sector_model_subtraction.py`. |

Suggested sequencing for a release: start WS-A, WS-B, WS-C, WS-D and WS-E at
once; they are independent. WS-F1 waits for WS-A. WS-G runs throughout. The
release blockers are P0-BUG-01, P0-BUG-02 (WS-A), P0-BUG-03, P0-BUG-06 (WS-B), P0-BUG-04, P0-BUG-07 (WS-C), P0-BUG-05,
P0-BUG-08 (WS-D).

## Part A — Missing Functionality, Bug Fixes and Regression Coverage

Unchecked tasks were outstanding at `main` `d0e7e148`. Master references name
the file and function on `master` at `76a7ff2f` that implements the behaviour
correctly, so inspect them with `git show master:<path>`. Where master relies
on Toil/CWL (restart, process management, Slurm), there is no portable code and
the task is new work. Port behaviour into the current execution owners, using
plain serializable payloads and the current output/finalizer/restart contracts.
Do not transplant old CWL workflows or merge an older migration snapshot.

### WS-A. Normalization and Pre-Application Wiring

#### P0-BUG-01. Apply prediction normalization to the predicted visibility buffer

- [x] **Release blocker; regression. Size S, complexity Medium. Fixed 2026-10-14.**
- **Resolved (2026-10-14):** `build_predict_model_data_command()` now runs a
  single `predict` step and applies the normalization through
  `predict.applycal.steps`, with `predict.applycal.normalization.parmdb`,
  `.correction=amplitude000` and `.solset=sol000` emitted whenever a
  normalization h5parm is given (master only emitted the last two when a
  solution h5parm was also present). The stand-alone `applycal` step and
  `usemodeldata` are gone. `tests/execution/test_predict_dp3.py` runs real DP3
  on the test MS with the production h5parm layout from
  `create_normalization_h5parm()` (one amplitude per frequency, no antenna or
  direction axis) and checks a visibility ratio of 4 for a normalization of 4,
  for both `predict` and `h5parmpredict` with a unit DD h5parm; against the old
  command the same tests give a ratio of 1. Token tests in
  `tests/execution/test_predict_flow.py` cover DI and DD predictions with and
  without a DD h5parm and the SageCal predictor. SageCal probe: DP3's
  `SagecalPredict` rejects `predict.directions` without an h5parm and accepts
  at most two `applycal` sub-steps, so it cannot combine fast-phase, slow-gain
  and normalization corrections. This limit is shared with master and is now
  noted under `sagecalpredict` in `docs/source/parset.rst`.
- **Origin (git history, 2026-10-14):** the first Prefect port (`70859094` on
  `gec-468-ai-migrate-to-prefect-backup`) copied the CWL step exactly,
  including master's gating of `predict.applycal.normalization.correction` and
  `.solset` on a solution h5parm being present. The four-cycle benchmark
  strategy added on 2026-07-09 (cycle 1 DI phases with `do_normalize`, cycle 4
  DI full-Jones) produced a DI prediction with a normalization h5parm and no
  solution h5parm, which DP3 rejects with `Key predict.applycal.correction
  unknown` (reproduced with DP3 6.6; master emits the same command shape, so by
  source it has the same latent failure). Commit `5a6b8620` ("Fix bug in apply
  normalization solutions during di predict") worked around the error by moving
  the normalization into a stand-alone `applycal` step with `usemodeldata=True`
  and `invert=False`, and rewrote the command tests to expect those tokens. The
  step corrected an empty model buffer because `predict.operation=replace`
  writes to the data buffer. The 2026-07-06 option-matrix normalization
  scenario was a single cycle with one sector, no peeling and no reweighting,
  so no prediction ever consumed the normalization h5parm and the
  branch-vs-master gate (rerun as `0cb3c11b`) could not expose the no-op. The
  benchmark that triggered the workaround measured timing and task structure,
  not outputs. The current fix emits the normalization correction and solset
  whenever a normalization h5parm is given, which closes both the branch no-op
  and master's missing-key crash on the DI path.
- **Verified (2026-10-14):** `build_predict_command()` in
  `rapthor/execution/predict/commands.py` runs `steps=[predict,applycal]` with
  `predict.operation=replace` and a stand-alone `applycal` step whose
  `applycal.normalization.usemodeldata=True`. With `operation=replace` the
  prediction is written to the data buffer, not the model buffer, so the
  normalization is applied to an empty model buffer and the output is
  unchanged. A DP3 `v6.6-136-g2cf52e0f` probe with a constant amplitude of 2
  gave a visibility ratio of 1 instead of 4; `usemodeldata=False` gave 4.
  The existing test checks command tokens only.
- **Master reference:** `rapthor/pipeline/steps/predict_model_data.cwl` has a
  single `predict` step and applies normalization inside the predict step's own
  applycal: `predict.applycal.steps` includes `normalization`, with
  `predict.applycal.normalization.parmdb`, `.correction=amplitude000` and
  `.solset=sol000`. `Predict._get_dp3_applycal_steps()` in
  `rapthor/operations/predict.py` appends `"normalization"` to the step list
  when `field.apply_normalizations` is set and passes `normalize_h5parm`
  separately. Master does not use a second `applycal` step or
  `usemodeldata`.
- **Change:** move normalization back into `predict.applycal.steps` and supply
  `predict.applycal.normalization.parmdb`, as master does. Remove the
  stand-alone `applycal` step from the predict command. Check the sign
  convention: DP3's predict-internal applycal corrupts the model with the
  (non-inverted) solutions, which is the intended direction for a normalization
  stored as an amplitude correction. Keep this distinct from P0-BUG-02's solve-time
  model corrections.
- **Done when:** a real DP3 test predicts with and without a normalization
  h5parm and shows the expected amplitude ratio for the production h5parm
  layout (one amplitude per frequency); command tests cover DI and DD
  predictions with and without a DD h5parm; the SageCal predictor path is
  checked or explicitly excluded. All met; see **Resolved** above.

#### P0-BUG-02. Restore missing DI/normalization wiring in calibration commands

- [x] **Release blocker; regression. Size M, complexity High. Fixed 2026-10-15
  on `gec-669-fix-normalisation-and-preapply-calibrate`.**
- **Resolved (2026-10-15):** gap 1: `build_calibration_dp3_steps()` keeps the
  stand-alone `applycal` step for WSClean prediction (only DP3 image-based
  prediction moves it after `predict`/`applybeam`). A DP3 probe confirmed the
  step corrects the data only and leaves model data columns untouched, so the
  WSClean model columns are not corrected twice. Gap 2:
  `calibration_applycal_steps` now has a distinct meaning, the DDECal-internal
  model corrections (`[normalization]` or `None`, from the new
  `build_calibration_solve_applycal_steps()` and `DDECAL_MODEL_APPLYCAL_STEPS`
  in `rapthor/lib/calibration.py`); it is forwarded through the payload,
  validated in `builders.py`, and `_solve_slots_for_chunk()` attaches it with
  `normalize_h5parm` to the solve that predicts the model. The real-DP3 test
  `tests/execution/test_calibrate_dp3.py` runs `run_calibrate_chunk()` with a
  fast-phase plus slow-gain chunk on data equal to the predicted model and gets
  slow-gain amplitudes of 1.0 without and 0.5 with a normalization of 4, with
  unchanged phases; the old wiring gives 1.0 in both cases.
- **Deviation from master, with evidence (DP3 6.6.0, fresh MS copy per case):**
  DDECal applies its `applycal` sub-steps only to the model it predicts itself.
  Normalization on a sourcedb solve gives amplitude 0.5; on a solve that reuses
  the kept model (`reusemodel=[solve1.*]`) it gives 1.0, that is no effect; on a
  solve reading model data columns 1.0. Normalization on the predicting solve1
  alone propagates through the kept model to solve3 (0.5), and on both solves it
  is not applied twice (0.5). Master wires `solve1.keepmodel=True`,
  `solve3.reusemodel=[solve1.*]` and the normalization on `solve3` only, which
  is exactly the no-op case. DP3's source agrees: the solve-internal applycal
  is part of `OnePredict`, the per-direction predictor built from the sky
  model, and is applied right after prediction, before the buffer reaches
  DDECal or is kept; reused models and model data columns never pass through
  it. The stand-alone
  `applycal.normalization.usemodeldata=True` sub-step changes neither the data
  nor model data columns (1.0 in every probe); per DP3's `OneApplyCal` it acts
  only on named direction buffers from a DP3 predict step, which is the
  image-based path. Consequently the flux-scale normalization cannot reach
  WSClean model columns in DP3 6.6: the Calibrate adapter logs a warning and
  `docs/source/parset.rst` documents it under `use_wsclean_predict`. The DI
  calibrate block keeps `normalize_h5parm=None` as on master; the DI predict
  (P0-BUG-01) already normalizes the DI model columns.
- **Verified (2026-10-14), two separate gaps:**
  1. `build_calibration_dp3_steps()` in `rapthor/operations/calibrate/plan.py`
     prepends `applycal` only when `preapply_solutions` is set *and* neither
     image-based nor WSClean prediction is active (`plan.py:206`). The WSClean
     prediction path therefore never applies DI or full-Jones solutions to the
     data before a DD solve.
  2. `CALIBRATION_SOLVE_ARGUMENTS` in `rapthor/execution/calibrate/commands.py`
     already emits `solveN.applycal.normalization.correction/solset` constants,
     and `SOLVE_SLOT_ARGUMENTS` maps slot keys `applycal_steps` and
     `normalize_h5parm` to `solveN.applycal.steps` and
     `solveN.applycal.normalization.parmdb`. But `_solve_slots_for_chunk()` in
     `rapthor/execution/calibrate/solves.py` never puts those keys into the
     slot dictionaries, and `Calibrate._build_applycal()` computes
     `calibration_applycal_steps` that nothing consumes. The slow-gain solve
     therefore solves against an un-normalized model.
- **Master reference:**
  - Gap 1: `Calibrate._get_dp3_steps()` in `rapthor/operations/calibrate.py`
    (around line 680) uses `if preapply_solutions and not
    self.field.use_image_based_predict: common_steps = ["applycal",
    *common_steps]`. WSClean prediction keeps the `applycal` step because the
    model columns are already predicted and only the data needs correcting.
  - Gap 2: `rapthor/pipeline/parsets/calibrate_pipeline.cwl` (around line
    1347) passes `solve3_normalize_h5parm: normalize_h5parm` and
    `solve3_applycal_steps: ddecal_applycal_steps` to the slow-gain solve only.
    `rapthor/pipeline/steps/ddecal_solve.cwl` lines 39–52 define the constant
    `solveN.applycal.normalization.correction/solset` for all four solves, and
    `Calibrate._build_applycal()` (line 705) returns `ddecal_applycal_steps`
    alongside `applycal_steps`. Phase solves do not receive the normalization
    because the phase-only model is not affected by a scalar amplitude scale.
- **Change:** for gap 1, drop the `use_wsclean_predict` exclusion in
  `build_calibration_dp3_steps()` and confirm the WSClean model columns are not
  corrected a second time. For gap 2, forward `calibration_applycal_steps` and
  `normalize_h5parm` into the slow-gain slot in `_solve_slots_for_chunk()` (and
  the DI slow-gain slot if DI slow gains are supported), matching master's
  solve3-only wiring. Trace data and model corrections through the DP3,
  image-based and WSClean prediction paths to ensure nothing is applied twice.
- **Done when:** command tests show `solveN.applycal.steps` and
  `solveN.applycal.normalization.parmdb` on the solve that predicts the model
  and on no other solve (changed from "the slow-gain solve" after the DP3
  probes above); a real-tool comparison of normalized slow-gain amplitudes
  against a known reference agrees within tolerance; WSClean-prediction
  commands include the `applycal` step when `preapply_solutions` is true.
  All met; see **Resolved** above. Coordinate with WS-F1
  (P2-PORT-01/P3-PORT-01) which edits the same command builders afterwards.

### WS-B. Solution State and Application

#### P0-BUG-09. Stop the pre-apply calibration chain from rewriting its input MS

- [ ] **Release blocker; also present on master. Size S, complexity Medium. Found
  2026-10-15 while writing the developer guide.**
- **Verified (2026-10-15, DP3 6.6 probe on a fresh copy of the test MS):**
  `steps=[applycal,solve1]` with `msout=` and a DI amplitude of 2 left the
  input MS with DATA divided by 4; `steps=[solve1]` left it unchanged;
  `steps=[applycal,solve1,null]` left it unchanged. DP3's driver
  (`DP3/base/DP3.cc:583-600`) appends an in-place `MSUpdater` when `msout` is
  empty and any step declares that it provides data or flags; `OneApplyCal`
  and `ApplyBeam` do (`OneApplyCal.h:48-51`, `ApplyBeam.h:103-106`), DDECal
  alone does not. `build_calibration_dp3_steps()` in
  `rapthor/operations/calibrate/plan.py` appends `null` only on the BDA chain,
  so the plain chain `[applycal, solve1, ...]` (DI pre-application, from
  P0-BUG-02 also the WSClean path) and the image-based chain
  `[predict, applybeam, applycal?, solve1, ...]` rewrite `msin.datacolumn` of
  the calibration input. That input is the time-chunk MS or, with a data
  fraction of 1 on one node, the original input MS, and later operations read
  it again and apply the DI corrections a second time.
- **Master reference:** none; `ddecal_solve.cwl` uses `msout=` with the same
  step chains, so master has the same behaviour. The DI pre-apply exists only
  for strategies with a `"di"` mode, which explains why it was not noticed on
  the default strategy.
- **Change:** terminate every solve-only chain with the `null` step (not only
  the BDA chain), or give the chain an explicit non-writing output; keep the
  `null.type=null` emission in `build_calibration_solve_command()`. Add a
  real-DP3 test that runs `run_calibrate_chunk()` with a DI pre-apply and
  asserts the input MS DATA column is unchanged, and a command test for the
  trailing `null` on every chain shape.
- **Done when:** the command tests show `null` at the end of the plain,
  WSClean and image-based chains; the real-DP3 test passes; the AGENTS rule
  "do not modify original input Measurement Sets" is covered by a test for
  the calibrate operation.

#### P0-BUG-03. Keep DI and DD amplitude-application state separate

- [x] **Release blocker; regression. Size M, complexity Medium. Fixed 2026-10-15
  on `gec-670-fix-amplitude-state-and-phase-application` (stacked on the two
  normalization branches).**
- **Resolved (2026-10-15):** `Field.scan_h5parms()` now scans
  `dd_h5parm_filename` and `di_h5parm_filename` separately through a new
  `_scan_scalar_h5parm()` and keeps `dd_apply_amplitudes` and
  `di_apply_amplitudes` next to the legacy `apply_amplitudes` (which still
  follows `h5parm_filename`); a lone supplied h5parm counts as the DD product.
  `Image` takes its amplitude state from `dd_apply_amplitudes` (WSClean facet
  soltabs and the prepare-imaging `slowgain` step), `Predict` uses the flag of
  the product it actually applies, and the DI pre-apply in `Calibrate` uses
  `di_apply_amplitudes`; the solve-plan-based assignment of the DI flag in the
  finalizer is gone, so the file content is the single source. Tests cover both
  mode orders in `tests/lib/test_field.py` (real h5parms via the new
  `write_scalar_h5parm` helper in `tests/conftest.py`), and the imaging,
  prediction and calibration adapters in `tests/operations/`. Not ported from
  master: `scalar_h5parm_directions` (P2-BUG-01) and `input_apply_amplitudes`.
- **Verified (2026-10-14):** `Calibrate.finalize()` in
  `rapthor/operations/calibrate/base.py:945` sets
  `field.h5parm_filename = field.di_h5parm_filename` after a DI solve.
  `Field.scan_h5parms()` in `rapthor/lib/field.py:1727` scans
  `self.h5parm_filename` alone and sets a single `apply_amplitudes` flag.
  `Image` in `rapthor/operations/image/base.py:301` passes
  `dd_apply_amplitudes=self.apply_amplitudes`, so a DD amplitude product
  followed by a DI phase product loses its amplitude application in imaging
  and prediction. `Field.di_apply_amplitudes` exists but there is no
  `dd_apply_amplitudes` state.
- **Master reference:** `Field.scan_h5parms()` in `rapthor/lib/field.py:1729`
  resets `dd_apply_amplitudes` and `di_apply_amplitudes`, then scans each of
  `h5parm_filename`, `dd_h5parm_filename` and `di_h5parm_filename` with
  `_scan_scalar_h5parm()` (returns whether `amplitude000` is present), reusing
  the result when two attributes point at the same file. It also records
  `scalar_h5parm_directions[filename]` for P2-BUG-01 and `input_apply_amplitudes` for
  the parset-supplied h5parm. Master's `finalize()` also sets
  `h5parm_filename` to the DI product (`calibrate.py:1054`), so the separation
  comes entirely from the scan, not from the finalizer.
- **Change:** port `_scan_scalar_h5parm()` and the per-file flag derivation;
  pass `dd_apply_amplitudes` from the DD product in `Image` and `Predict`;
  preserve the flags through restart (they are re-derived from files, so a
  restart only needs the filenames restored).
- **Done when:** tests run both mode orders (DD then DI, DI then DD) with
  different soltab contents and assert the imaging applycal steps and predict
  h5parm selection; a DD-amplitude/DI-phase run applies `slowgain` in the
  prepare-imaging command and the DD h5parm in DD prediction. All met; see
  **Resolved** above.

#### P0-BUG-06. Apply combined phases once with `dde_method = single`

- [x] **Release blocker (cheap); also present on master. Size S, complexity Low.
  Fixed 2026-10-15 on `gec-670-fix-amplitude-state-and-phase-application`.**
- **Resolved (2026-10-15):** `SOLVE_TYPE_TO_APPLYCAL_STEP` in
  `rapthor/operations/image/plan.py` maps `medium_phase` to `fastphase`, so all
  phase solves of a mode share one step and `slowgain` is added only when the
  product carries amplitudes (P0-BUG-03). The real-DP3 test
  `tests/execution/integration/test_image_dp3.py` runs the production
  `prepare_visibility_ms()` on the predicted model with a combined product
  (per-antenna phases 0.1·i rad, amplitude 2): the planner's step list applies
  the phases once and the amplitudes as 1/4, for a phase-only and a
  phase-plus-amplitude product, within Dysco compression noise (1 %); the former
  `[fastphase,mediumphase]` list is shown to apply the phases twice. Facet
  application is unchanged. Master keeps the double application.
- **Verified (2026-10-14):** `build_image_applycal_steps()` in
  `rapthor/operations/image/plan.py` maps DD `fast_phase` and `medium_phase` to
  separate `fastphase` and `mediumphase` steps (`SOLVE_TYPE_TO_APPLYCAL_STEP`),
  while DI phases are collapsed to one `fastphase` step (`plan.py:232`).
  `build_prepare_imaging_data_command()` in `rapthor/execution/image/commands.py`
  defines `applycal.fastphase.*` and `applycal.slowgain.*` but no
  `applycal.mediumphase.*`, so DP3 gives the `mediumphase` sub-step the
  top-level `applycal.correction=phase000` and applies the combined `phase000`
  twice. A DP3 probe measured 0.14 radians after two steps versus 0.07 after
  one. Tests in `tests/operations/test_image.py` (lines 606–796) assert the
  duplicate step.
- **Master reference:** none; master has the same step list
  (`rapthor/operations/image.py:241`, `step = "fastphase" if mode == "di" else
  "mediumphase"`) and the same missing `mediumphase` settings in
  `prepare_imaging_data.cwl`. Master's combined DD product also holds a single
  `phase000`, so the same double application occurs there.
- **Change:** select the applycal steps from the soltabs actually present in
  the selected h5parm (one phase step for `phase000`, one amplitude step for
  `amplitude000`), as the DI branch already does. Update the affected tests.
- **Done when:** a real DP3 test shows one phase application for a combined
  phase-only product and for a phase-plus-amplitude product with
  `dde_method = single`; facet application is unchanged. All met; see
  **Resolved** above.

#### P1-DEC-01. Resolve the discarded medium-phase result in phase-only DD cycles

- [ ] **Decision needed before release; also present on master. Size S for
  code, M including verification; complexity Medium.**
- **Verified (2026-10-14):** `Calibrate._dd_active_solution()` in
  `rapthor/operations/calibrate/base.py:1013` returns the combined product only
  when both a phase and a slow-gain solve are present; for
  `[fast_phase, medium_phase]` it returns `_dd_collected_h5parm(solve_plan[0])`,
  which is the fast-phase h5parm. The medium-phase solve runs and its product
  is kept, but it is never applied. The DI branch (`_di_combined_solution()`)
  combines phases. The built-in strategy's early phase-only cycles use this
  sequence, and the product documentation describes combined solves.
- **Master reference:** master applies only the fast phases in this case as
  well (`rapthor/operations/calibrate.py:278`, which selects
  `field.fast_phases_h5parm_filename` for the phase-only product). Master's CWL
  does produce `combined_fast_medium1_phases.h5parm` (`calibrate.py:215`) but
  does not apply it. There is therefore no "correct" master behaviour to port;
  this is a scientific decision.
- **Change:** decide whether phase-only cycles should apply the combined
  fast-plus-medium phases (then make `_dd_active_solution()` return the
  combined product) or whether the medium solve should be omitted in those
  cycles (then change `dd_calibration_strategy(include_slow_gains=False)` in
  `rapthor/lib/strategy.py`). Update `docs/source/products.rst` and the
  science reference to match.
- **Done when:** phase values in the applied h5parm match the decision; imaging
  products are compared for one phase-only cycle; the initial-model-dependent
  phase-only cycles are retained.

#### P2-BUG-01. Carry solution directions consistently into image-only cycles

- [ ] **Can ship with a documented limitation. Size L, complexity High.**
- **Verified (2026-10-14):** `Field.update()` in `rapthor/lib/field.py:1963`
  always calls `update_skymodels()`, regrouping the sky model even in image-only
  cycles. `Predict._get_applycal_h5parm_filename()` in
  `rapthor/operations/predict.py:210` rejects a DD product from an earlier
  cycle via `_is_current_cycle_solution()` and continues without it (with a
  warning). `Image._facet_skymodel_file()` in `rapthor/operations/image/base.py:172`
  permits carry-forward but selects `calibration_skymodel_file_prev_cycle`,
  which after two consecutive image-only cycles need not be the layout that
  produced the applied h5parm.
- **Master reference:** `Field.update()` in `rapthor/lib/field.py:2023`
  computes `reuse_solution_layout = index > 1 and not do_calibrate and
  do_image and (a DD, DI or scalar h5parm exists)` and skips
  `update_skymodels()` entirely in that case, logging "Reusing the existing
  calibration patch layout for image-only cycle". `scan_h5parms()` records
  `scalar_h5parm_directions[filename]` and `rapthor/operations/image.py:280`
  reads the facet directions from that map for the applied h5parm. Master's
  `Predict._get_applycal_h5parm_filename()` (`predict.py:201`) returns the DD
  product whenever it exists, with no cycle check.
- **Change:** adopt master's layout-reuse rule in `Field.update()`, associate
  each applied h5parm with its producing sky model/facet layout, and use that
  single contract in both `Predict` and `Image`. Fail clearly on incompatible
  directions instead of silently dropping the h5parm. Preserve the separate,
  more permissive rules for optimizer seeds.
- **Done when:** tests cover outliers, multiple sectors, peeling/reweighting and
  two consecutive image-only cycles (including repeated final cycles) and
  assert the h5parm, facet file and sky model used by both operations.

### WS-C. Strategy Semantics

#### P0-BUG-07. Give the built-in final step an independent strategy dictionary

- [ ] **Release blocker (cheap); also present on master. Size S, complexity Low.**
- **Verified (2026-10-14):** `set_selfcal_strategy()` in
  `rapthor/lib/strategy.py:228` does `strategy_steps.append(strategy_steps[-1])`
  then sets `channel_width_hz = 4e6` on the shared object, so the last selfcal
  step's width also changes from 8 MHz to 4 MHz.
- **Master reference:** none; `rapthor/lib/strategy.py:191` on master has the
  same aliasing.
- **Change:** `strategy_steps.append(copy.deepcopy(strategy_steps[-1]))`.
- **Done when:** a test asserts the final and last selfcal dictionaries are
  independent, including nested `calibration_strategy` lists, and final-pass
  selection with equal data fractions is unchanged.

#### P0-BUG-04. Define per-cycle semantics for an omitted calibration strategy

- [ ] **Release blocker; regression. Size M, complexity Medium.**
- **Verified (2026-10-14):** `Field.update()` in `rapthor/lib/field.py:1983`
  does `self.__dict__.update(step_dict)` and then `set_calibration_strategy()`,
  which only substitutes `default_calibration_strategy()` when the attribute is
  falsy. A later cycle that omits `calibration_strategy` therefore inherits the
  previous cycle's value, contrary to `docs/source/strategy.rst`. First-cycle
  omission yields the full four-solve DD default, whereas master resolves to
  fast plus medium phases. The preflight warning reports a default of `None`,
  and the upgrade guide omits the change.
- **Master reference:** `Field.update()` in `rapthor/lib/field.py:1980` adds
  `if "calibration_strategy" not in step_dict: self.calibration_strategy =
  None` before `set_calibration_strategy()`, so every cycle is resolved afresh.
  `Field.set_calibration_strategy()` calls `resolve_calibration_strategy()` in
  `rapthor/lib/calibration.py:27`, which returns the explicit strategy if
  given, otherwise `{"dd": ["fast_phase", "medium_phase"] + (["slow_gains"] if
  do_slowgain_solve else []), "di": ["full_jones"] if do_fulljones_solve else
  []}` plus a `defaulted` flag. Master's `set_user_strategy()` fills missing
  required parameters with defaults per cycle and logs the value used.
- **Change:** reset the attribute per cycle as master does; decide whether the
  default for an omitted strategy is master's legacy resolution (fast+medium,
  plus slow gains only via the legacy flag) or the new explicit default, and
  document it. Distinguish image-only carry-forward (`do_calibrate = False`)
  from choosing new solves. Correct the preflight warning to print the resolved
  strategy. Record the behaviour in `docs/source/upgrading.rst`.
- **Done when:** tests cover first and later omissions, explicit `None`, and
  legacy flags; the warning names the resolved solves; the strategy and upgrade
  docs agree with the code.

#### P2-DEC-01. Supported solve combinations (`0ebc0690`)

- [ ] **Decision. Size M (investigation), complexity Medium.**
- Master's `_validate_calibrate_strategy()` in `rapthor/lib/strategy.py:387`
  validates mode/solve names permissively; `main`'s version
  (`rapthor/lib/strategy.py:460`) allows only specific sequences. Inventory the
  master workflows now rejected (for example DI `medium_phase` alone and DD
  `full_jones`), establish which actually execute correctly on master, then
  either port each required sequence with ordered solve, application, seeding
  and product tests, or document the deliberate restriction in the upgrade
  guide. Do not infer support from parsing alone.

### WS-D. Restart, Reset and Output Ownership

#### P0-BUG-05. Make partial-product restart checks reliable before adding retries

- [ ] **Release blocker; new work (Toil owned restart on master). Size L,
  complexity High.**
- **Verified (2026-10-14):**
  - `prepare_visibility_ms()` and `concatenate_prepared_visibilities()` in
    `rapthor/execution/image/preparation.py:27,70` skip the DP3 command when the
    output directory exists and return it via `require_directory()`, so an
    interrupted DP3 write is accepted as complete.
  - `run_concatenate_epoch()` in `rapthor/execution/concatenate/flow.py:90`
    never checks for an existing output and the concatenation command rejects
    an existing partial MS with `overwrite=False`.
  - `compress_image_records()` in `rapthor/execution/image/outputs.py:116` and
    the mosaic compression in `rapthor/execution/mosaic/flow.py:310` run `fpack`
    unconditionally; a second run against an existing `.fz` returns 255.
- **Master reference:** none portable. Master's restart relies on Toil's job
  store and CWL output staging (`rapthor/lib/cwlrunner.py`), which the
  migration retired. The operation-level `is_done()` skip in
  `rapthor/lib/operation.py` is the only comparable logic and only covers whole
  operations.
- **Change:** write outputs to a worker-local or `.partial` path and promote
  atomically on success, or validate completeness (for an MS: table lock
  absent, expected row count or `msout` footer; for `.fz`: skip when the
  compressed file exists and is newer than the source). Remove owned partial
  outputs before regeneration. Never remove or overwrite original input MSs.
- **Done when:** failure-injection tests interrupt copy, DP3, concatenation and
  compression and after product creation, then restart successfully with
  consistent records; an empty or partial directory is never returned as a
  valid record. P0-BUG-08 depends on this.

#### P0-BUG-08. Implement or retire the ineffective `prefect_retries` option

- [ ] **Release blocker (cheap if retired). Size S (retire) or M (implement),
  complexity Low/Medium.**
- **Verified (2026-10-14):** `rapthor/execution/config.py:84` parses the value
  into `ExecutionConfig.retries` and `rapthor/execution/pipeline/flow.py:407`
  propagates it into the parset dictionary, but no `@task` or `@flow` passes
  `retries=`.
- **Master reference:** none; master has no equivalent option (Toil retries
  were configured through the CWL runner).
- **Change:** for the release, remove the option from both defaults, the parser
  and `docs/source/parset.rst`, or document it as reserved and warn when set.
  Implement real retries only after P0-BUG-05, limited to tasks whose outputs are
  regenerated safely.
- **Done when:** either the option is gone from defaults/docs/tests, or actual
  attempt counts are tested for zero/nonzero settings with correct records and
  cleanup.

#### P1-BUG-04. Restrict predict output discovery to the producing chunk

- [ ] **Fix if multi-chunk runs are in scope. Size S, complexity Low.**
- **Verified (2026-10-14):** `run_predict_postprocess()` in
  `rapthor/execution/predict/flow.py:195` globs
  `<obs>*_di.ms`, `<obs>*_field` and `<obs>*.sector_*` across the working
  directory and ignores the chunk `infix`, so a `_t0` task returns `_t1`
  products too.
- **Master reference:** `rapthor/pipeline/steps/subtract_sector_models.cwl`
  and `add_sector_models.cwl` use the same glob patterns
  (`$(inputs.msobs.basename)*_field`, `*.sector_*`, `*_di.ms`), but every CWL
  step ran in its own isolated working directory, so a glob could only match
  that step's outputs. In the Prefect flow all chunk tasks share
  `pipeline_working_dir`, so the pattern must be made chunk-specific.
- **Change:** build the exact expected paths from `msobs`, `infix` and the
  sector names (the helpers already know them) instead of globbing.
- **Done when:** tests with two time chunks, DI addition, DD subtraction and
  peeling assert that every returned record belongs to the task, including on
  restart.

#### P2-BUG-02. Give calibration plot tasks explicit output ownership

- [ ] **Size S, complexity Low.**
- **Verified (2026-10-14):** `run_plot_solutions()` in
  `rapthor/execution/calibrate/collection.py:250` diffs a shared `*.png` glob
  before and after execution; concurrent writers are included and reruns with
  existing filenames return nothing.
- **Master reference:** `rapthor/pipeline/steps/plot_solutions.cwl` also
  globs `*.png`, but each CWL step ran in its own isolated working directory,
  so ownership was implicit. The Prefect tasks share `pipeline_working_dir`,
  which is why the before/after diff was introduced.
- **Change:** give each plot task its own output directory or an explicit
  manifest of expected filenames derived from the h5parm and soltab.
- **Done when:** overlapping solves and reruns each retain exactly one copy of
  every expected plot; publication (P1-BUG-03) uses the manifest.

#### P2-BUG-03. Include repeated final cycles in reset discovery

- [ ] **Size S, complexity Low. Also present on master.**
- **Verified (2026-10-14):** `modifystate.run()` in `rapthor/modifystate.py:59`
  enumerates `strategy_steps` only, while `rapthor/execution/pipeline/flow.py:169`
  runs `ntimes_to_repeat_final_cycle + 1` final cycles.
- **Master reference:** none; `rapthor/modifystate.py:59` on master has the
  same loop.
- **Change:** discover `pipelines/<operation>_<index>` directories beyond the
  strategy length and order them by index.
- **Done when:** resetting a repeated final cycle removes that cycle and every
  later operation and retains earlier completed work. Coordinate with P2-BUG-04.

#### P2-BUG-04. Reset owned scratch workspaces and distinguish log attempts

- [ ] **Size M, complexity Medium.**
- **Verified (2026-10-14):** `modifystate.run()` uses
  `shutil.rmtree(..., ignore_errors=True)` (`modifystate.py:124,145`), which
  leaves an operation symlink to shared scratch intact and can leave a
  `.<operation>.scratch` recovery record. Aggregate `logs/commands.jsonl`,
  `logs/tasks.jsonl` and `rapthor.log` survive reset.
- **Master reference:** master uses the same `rmtree` calls but had no scratch
  symlinks or aggregate JSONL logs, so the gap is migration-specific.
- **Change:** resolve and remove owned workspace links, targets and recovery
  records together; never follow an unowned link. Rotate or tag aggregate logs
  with an attempt identifier so metrics exclude reset attempts.
- **Done when:** tests cover normal, staged and partially promoted workspaces
  and selective reset of later operations.

#### P2-BUG-05. Preserve or explicitly retire astrometry-offset JSON products

- [ ] **Size S, complexity Low.**
- **Verified (2026-10-14):** the image flow returns `sector_offsets`
  (`rapthor/execution/image/flow.py:395`) but `Image.finalize()` neither copies
  nor excludes it from cleanup.
- **Master reference:** `Image.finalize()` in `rapthor/operations/image.py:945`
  adds `sector_offsets` to the diagnostics set copied to
  `plots/image_<index>/` alongside `sector_diagnostics` and
  `sector_diagnostic_plots`.
- **Change:** copy the per-sector offsets JSON with the other diagnostics and
  add it to the restart records, or document its removal in
  `docs/source/products.rst` and the upgrade guide.
- **Done when:** multi-sector output names are tested and final cleanup leaves
  the file in place.

#### P2-BUG-06. Handle empty model mosaics and document the rendering default

- [ ] **Size M, complexity Medium.**
- **Verified (2026-10-14):** `Mosaic._model_skymodels_for_image_name()`
  supplies filtered sky models to the default WSClean rendering path, changing
  the `model-pb` product relative to regridding unfiltered sector models.
  `combine_sector_skymodels()` raises when all supplied models are empty.
- **Master reference:** `rapthor/operations/mosaic.py:70` regrids the sector
  `model_file_true_sky` images directly through `regrid_image.cwl`; no sky
  model rendering is involved, so empty models cannot fail there.
- **Change:** define the intended model product, handle all-empty models
  explicitly (produce an empty mosaic or skip with a record), compare rendered
  and regridded products, and document the default in the upgrade guide.
- **Done when:** mixed and all-empty sector tests pass and the product
  comparison is recorded.

### WS-E. Runtime, Cluster and Diagnostics

#### P1-BUG-01. Stop command descendants on cancellation and bound captured output

- [ ] **Fix before release for any cluster use. Size M, complexity Medium.**
- **Verified (2026-10-14):** `_run_captured_shell_command()` in
  `rapthor/execution/shell.py:457` starts `bash <script>` with
  `subprocess.Popen` without `start_new_session=True`; the `finally` block
  calls `process.kill()` on the wrapper only (`shell.py:512`). A child
  `sleep` survived an interrupted runner in a probe. All output is retained in
  `output_lines` while also being written to the log file.
- **Master reference:** none portable; Toil managed process trees.
- **Change:** start the command in its own process group, send `SIGTERM` to the
  group with a grace period then `SIGKILL`, and retain only a bounded tail of
  output for the exception (P2-PORT-06 uses it). Verify behaviour under Prefect
  cancellation and Dask worker shutdown, including MPI wrappers.
- **Done when:** tests interrupt a wrapper with a long-running descendant and
  assert the descendant is gone; memory use of a chatty command is bounded.

#### P1-BUG-02. Validate Slurm intent before bootstrap rewrites the task runner

- [ ] **Fix before release for any Slurm use. Size S, complexity Low.**
- **Verified (2026-10-14):**
  - `collect_slurm_config_issues()` in `rapthor/execution/slurm.py:70` requires
    `task_runner == "external_dask"`, but `bootstrapped_runtime()` in
    `rapthor/execution/runtime_bootstrap.py:217` starts a local cluster for
    `local_dask` and then rewrites the config to `external_dask`, so the guard
    never sees the effective configuration.
  - `ExecutionConfig.local_dask_worker_count` (`config.py:146`) returns
    `max_nodes` when `local_dask_workers = 0`, while `docs/source/parset.rst:789`
    documents one worker.
  - `slurm_cluster_spec()` (`slurm.py:27`) defaults `environ` to `{}` rather
    than `os.environ`, so `SLURM_NNODES` is ignored unless passed explicitly.
- **Master reference:** none portable; Toil's Slurm batch system owned node
  allocation.
- **Change:** run the Slurm guard on the configured (pre-rewrite) runner and
  reject `batch_system = slurm` with `local_dask`; make the worker default one
  worker or fix the docs; read `os.environ` by default in production.
- **Done when:** tests cover CLI bootstrap, environment/config precedence and a
  missing scheduler; explicit per-node resource limits are retained.

#### P1-BUG-03. Bound artifact publication and keep monitoring failures non-fatal

- [ ] **Fix before release for long dashboard-enabled runs. Size M,
  complexity Medium.**
- **Verified (2026-10-14):** `_run_operation()` in
  `rapthor/execution/pipeline/flow.py:443` calls
  `publish_plot_artifacts_for_field()` after every operation, which `rglob`s
  the whole `plots/` tree. `publish_file_artifacts()` in
  `rapthor/execution/artifacts.py:39` reads and base64-encodes every file,
  sends images as data URLs and JSON as Markdown, with no deduplication and no
  error boundary. The only switches are for FITS and postage-stamp previews.
- **Master reference:** none; master has no dashboard publication.
- **Change:** publish a file once per content hash, add a `prefect_publish_plots`
  switch, wrap publication in a warning-only error boundary and bound the total
  bytes per run.
- **Done when:** tests publish across several operations and count API calls;
  a failing artifact writer does not fail the run; database size over a
  many-cycle run is measured and recorded.

#### P2-PORT-05. Eliminate duplicate console records and ANSI codes in file logs

- [ ] **Size M, complexity Medium.**
- **Verified (2026-10-14):** `rapthor/_logging.py:79-90` adds a console
  handler alongside existing handlers and wraps the file handler's `emit` in
  ANSI colouring that mutates the shared record. `rapthor/execution/image/restoration.py:15`
  calls `logging.basicConfig()` at import time.
- **Master reference:** `fda65c8a` in `rapthor/_logging.py`: removes the
  `fh.emit = add_coloring_to_emit_ansi(fh.emit)` wrapper so colour applies to
  the console handler only, adds `logging.root.handlers.clear()` before adding
  Rapthor's handlers, and simplifies level parsing with
  `getattr(logging, level.upper(), logging.NOTSET)`.
- **Change:** port the master fix, but clear only Rapthor-owned handlers so
  Prefect's handlers (`rapthor/execution/prefect_logging.py`) survive; remove
  the import-time `basicConfig()`; make worker Python logs durable through the
  task log rather than inherited handlers.
- **Done when:** repeated setup and a Prefect run emit each message once to its
  intended destination; file logs contain no ANSI codes and keep module logger
  names; worker diagnostics survive API shutdown.

#### P2-PORT-06. Put the task log and failure cause in the surfaced exception

- [ ] **Size M, complexity Medium.**
- **Verified (2026-10-14):** `ShellCommandError` in `rapthor/execution/shell.py`
  carries the command and return code but not the output log path or the
  tool's final diagnostic lines.
- **Master reference:** `aabe35f2`: `Operation.handle_failure()` and
  `extract_log_errors()` in `rapthor/lib/operation.py` parse the CWL log with
  `LOG_ERROR_PARSER` and raise with the extracted error messages. The parser is
  CWL-specific and not needed; the user-facing behaviour (cause, exit code and
  log path in the raised error) is what to preserve.
- **Change:** attach the output log path and a bounded tail (from P1-BUG-01) to
  `ShellCommandError`; preserve it across the Prefect/Dask boundary so the CLI
  prints cause, exit code and log path.
- **Done when:** a failing DP3/WSClean fixture surfaces its cause through the
  CLI and the full log remains on disk.

#### P1-TEST-01. Validate a shared Prefect API and MPI placement on external Dask

- [ ] **Cluster release gate; validation task. Size L, complexity High
  (needs the target Slurm cluster).**
- **Verified (2026-10-14):** the Slurm example in
  `docs/source/running.rst:299` starts `dask worker` with `srun` and no
  `PREFECT_API_URL`/`PREFECT_HOME`; `bootstrapped_runtime()` sets the
  temporary API settings only in the launcher's environment. Nothing in the
  repository establishes a reachable shared API for remote workers.
  `tests/execution/test_slurm_execution.py` checks scheduler connectivity only.
- **Master reference:** none; Toil had no equivalent requirement.
- **Change:** either require a persistent Prefect API (B2) for multi-node runs
  and document how workers receive its URL, or verify and document the
  propagation that Prefect-Dask performs. Run calibration, multi-sector and
  real MPI imaging on the cluster and check task records, rank placement and
  resource exclusivity.
- **Done when:** the launch docs and diagrams match a configuration that was
  run end to end, or unsupported combinations are rejected at preflight.

### WS-F1. WSClean Prediction Ports

#### P2-PORT-01. Apply the full beam during WSClean prediction

- [ ] **Port `23ed80e9`. Size M, complexity Medium.**
- **Update (2026-10-15):** `main` commit `aff26023` adds `use_array_beam`, which
  makes the predict and concatenate beam mode a command option; the calibrate
  solve still hard-codes `solve1.beammode=array_factor`, so the WSClean half of
  this task remains.
- **Gap (verified 2026-10-14):** `main` has no `wsclean_predict_beam_interval`
  option, `build_wsclean_predict_command()` in
  `rapthor/execution/calibrate/commands.py` predicts without
  `-apply-facet-beam`, and `CALIBRATION_SOLVE_ARGUMENTS` fixes
  `solve1.beammode=array_factor`.
- **Master reference:** `23ed80e9` adds `wsclean_predict_beam_interval`
  (default 120 s) to both defaults and `Field`; `Calibrate` passes it as
  `beam_interval` to `rapthor/scripts/wsclean_predict.py`, which adds
  `-apply-facet-beam -facet-beam-update <interval>` and reads the
  `-model-fpb.fits` images; `ddecal_solve.cwl` turns `solve1.beammode` and
  `applybeam.beammode` into inputs set to `none` when WSClean has applied the
  beam.
- **Change:** add the option to `defaults.parset`, `defaults.json`, parsing,
  `Calibrate` inputs, the calibrate payload, validators and
  `build_wsclean_predict_command()`; make the DP3 beam mode a command option
  set to `none` for WSClean prediction. Main implementation in
  `rapthor/execution/calibrate/{prediction,commands,builders}.py`.
- **Done when:** both defaults, parset docs and test templates include the
  option; command tests and a real multi-facet test show one beam application,
  correct model column names and consistent fluxes between WSClean and DP3
  prediction.

#### P2-DEC-02. WSClean prediction disables solve BDA (Medium review)

- [ ] **Decision. Size M, complexity Medium.**
- `Parset.__check_and_adjust()` in `rapthor/lib/parset.py:336` and
  `build_calibration_dp3_steps()` disable time and frequency BDA for WSClean
  prediction. Master's `_get_dp3_steps()` also drops the `bdaavg` step for
  WSClean prediction (with a comment that model columns would need averaging
  too), so this is consistent with master, but master does not reset the
  parset values or warn. Document the limitation and its resource consequences,
  or establish the supported DP3 model-column combination before removing the
  guard. Test explicit BDA requests in each prediction mode.

#### P3-DEC-01. Named facet-region formats (`d90786e8`)

- [ ] **Decision. Size M, complexity Medium.**
- Master writes separate point-labelled and polygon-labelled DS9 files; `main`
  shares one helper in `rapthor/execution/regions.py`. Test multiple named
  facets with the supported DP3/WSClean/LSMTool versions, add separate formats
  only where needed, and verify selected facets match model columns and h5parm
  directions.

#### P3-PORT-01. Port improved WSClean model rendering and reordered-data reuse

- [ ] **Adapt `d587eec5`. Size L, complexity High.**
- **Gap:** `main` uses the supplied image geometry and predicts each frequency
  group/facet separately with `-no-reorder`.
- **Master reference:** `rapthor/scripts/wsclean_predict.py` after `d587eec5`
  derives rendering resolution from the sky extent, UVW coordinates and
  frequency, limits the image size, predicts the full band per facet with
  `-channels-out`, and reuses reordered data with `-parallel-reordering`,
  `-save-reordered` and `-reuse-reordered`.
- **Change:** implement the rendering calculation and bounded fallback in
  `rapthor/execution/calibrate/prediction.py`; extend the command options; keep
  reusable scratch alive through all facet consumers, isolated by chunk, with
  recovery on failure.
- **Done when:** single-band, multi-band and single-channel cases cover every
  requested channel; model visibilities agree within documented tolerances;
  CPU, memory, scratch and elapsed time are measured. Preserve the migration's
  correction of WSClean's end-exclusive channel ranges.

### WS-F2. Data Layout and Normalization Inputs

#### P2-PORT-02. Preserve normalization metadata for supplied survey catalogs

- [ ] **Port `dfa7f11a`/`6d4df857`. Size S, complexity Low.**
- **Gap (verified 2026-10-14):** `_get_survey_metadata()` in
  `rapthor/execution/image/flux_normalization.py:741` assigns every supplied
  catalog `flux_correction = 1.0` and `flux_err = 0.0`.
  `normalization_reference_frequencies` is missing from `defaults.json`
  although `defaults.parset` and parsing support it.
- **Master reference:** `_get_survey_metadata()` in
  `rapthor/scripts/normalize_flux_scale.py:742` calls
  `_identify_survey_from_frequency()` (line 795), which matches each supplied
  frequency against `SURVEY_METADATA` with `np.isclose` and reuses that
  survey's `flux_correction` and `flux_err`, falling back to 1.0/0.0 for
  unknown frequencies. `6d4df857` adds the JSON default.
- **Done when:** supplied and downloaded versions of the same survey give the
  same normalization; tests cover known/unknown frequencies, invalid lists and
  offline use.

#### P2-PORT-03. Accept observations with different station sets

- [ ] **Port `4c00305d`. Size L, complexity High.**
- **Gap (verified 2026-10-14):** `Field` in `rapthor/lib/field.py:301` raises
  when station lists differ; `rapthor/execution/image/preparation.py` always
  concatenates prepared MSs in time.
- **Master reference:** `4c00305d` replaces the check with the union
  `self.stations = list(set(all_stations))` and documents that DP3 ignores
  missing stations per observation. `Image` computes `concat_in_time = all
  station sets equal`, passes it to `image_sector_pipeline.cwl` so WSClean
  receives the list of prepared MSs instead of one concatenated MS, and sets
  `make_residual_visibilities = make_residual_visibilities and
  concat_in_time`.
- **Change:** build a deterministic station union; allow the image payload and
  command path to carry a list of prepared MSs; disable residual-MS creation
  for mixed station sets as master does (or implement a validated residual
  path); adapt the normalization MS and all output/finalizer/restart records.
- **Done when:** same-station and mixed-station multi-epoch tests run through
  calibration and imaging; serial and MPI commands receive the correct inputs;
  the limitation is documented.

#### P2-PORT-04. Restore a usable offline catalog-preparation workflow

- [ ] **Adapt `dfa7f11a`. Size M, complexity Medium.**
- **Gap:** master provides `rapthor/scripts/fetch_skymodel.py` and an HBA
  preparation workflow (`rapthor/pipeline/execution/fetch_model.cwl`,
  `hba_nl.cwl`) that downloads TGSS, Pan-STARRS, VLSSr and WENSS before an
  offline run. `main` accepts supplied catalogs but has no preparation helper.
- **Change:** provide a maintained command that reads the MS pointing,
  downloads the requested surveys on an internet-enabled machine and documents
  how the files and frequencies feed the existing parset options. Put the
  logic in the appropriate owners and the entry point in `pyproject.toml`.
- **Done when:** a prepared example completes normalization and diagnostics
  with `allow_internet_access = False`; pointing/radius, output paths and
  download failures are tested.

### WS-G. Build, Packaging, CI, Tests and Documentation

#### P2-PORT-07. Fix Ubuntu 24.04 APT source rewriting in all image stages

- [ ] **Port from `a39e517b`/`d1fd7249`. Size S, complexity Low.**
- `Docker/Dockerfile:31` and `ci/ubuntu_24_04-base:33` still rewrite
  `/etc/apt/sources.list`; master uses `/etc/apt/sources.list.d/ubuntu.sources`
  and also handles runtime stages. The Ubuntu 26.04 default is not part of
  this port.

#### P1-PORT-01. Align the standalone image's NumPy build and runtime

- [ ] **Complete `7eb0b03f`. Size M, complexity Medium. Blocks shipping the
  standalone image.**
- `Docker/Dockerfile:74` still installs `numpy<2`; the CI image uses NumPy 2
  with Boost.NumPy/PyBDSF compatibility work. `np.NaN` remains at
  `rapthor/execution/predict/sector_model_subtraction.py:223,282` (NumPy 2.4.6
  lacks it; the path is only reached with `use_compression`). Replace with
  `np.nan`, align the standalone build with the CI image, and verify NumPy,
  PyBDSF and python-casacore import in the final images with a small
  source-finding run.

#### P2-PORT-08. Update package metadata and the compiled test environment

- [ ] **Adapt `a39e517b`/`583dc808`. Size S–M, complexity Low.**
- `pyproject.toml` advertises Python >=3.9 and tests 3.9–3.13; master declares
  >=3.10 and tests 3.10–3.14 with an SPDX `license` expression,
  `license-files`, `setuptools>=77`, `EVERYBEAM_DATADIR` passthrough and
  `sitepackages = true`. Check the Prefect stack before narrowing Python;
  decide how tox obtains compiled astronomy libraries before adopting
  `sitepackages`.

#### P3-TEST-01. Restore Astron/SKA CI support and bounded, balanced test execution

- [ ] **Adapt `be6642f8`, `3e352b73`, `c3fac822`. Size L, complexity Medium.**
- Master has common/Astron/SKA CI files with mirror runner setup, stored
  duration balancing (`--durations-path`, `--splitting-algorithm
  least_duration`), eight unit workers to avoid OOM, and integration CPU
  requests capped at `min(cpu_limit, misc.nproc())`. `main` has one CI file,
  `-n auto` and fixed six-CPU integration settings. Keep field and
  Prefect-server tests serial and isolated state per shard.
- [ ] **P2-TEST-01 (Size S, Low):** `test_integration_restore_skymodel*` in
  `tests/execution/test_image_restoration.py:80,95` call real WSClean but carry
  no `integration` marker; the collection hook in `tests/conftest.py:216` adds
  only the `prefect` marker. Mark them and verify non-integration selection
  excludes them.

#### P3-PORT-02. Add the Docker dependency-version inspection helper

- [ ] **Port `f75dfe53`. Size S, complexity Low.**
- Add `Docker/extract_version_hashes.sh`, exposing the image's
  `nl.astron.rapthor.*.version` labels as `NAME_COMMIT=value`, with tests in
  the current layout using a fake Docker executable.

#### P3-TEST-02. Verify facet-RMS diagnostics through a real pipeline run

- [ ] **Adapt `eb1b6f2f`'s `test_sector_diagnostics.py`. Size M, complexity Low.**
- Add the end-to-end assertion on the saved `facets_rms` JSON (flat-noise and
  beam-corrected statistics per facet).

#### P3-TEST-03. Restore the parallel-gridding integration matrix

- [ ] **Adapt `38abdb92`'s `test_wsclean_parallel_gridding.py`. Size M,
  complexity Low.**
- Real-tool matrix for task counts 1/2, full/single DDE modes and serial/MPI
  routing, inspecting `commands.jsonl`. Respect P3-TEST-01's CPU limits.

#### P2-TEST-02. Reset flags and weights in the synthetic normalization MS

- [ ] **Port `da442dfc`. Size S, complexity Low.**
- `tests/integration/conftest.py:ms_for_normalisation` must set `FLAG` and
  `FLAG_ROW` false and `WEIGHT`/`WEIGHT_SPECTRUM` to one before predicting.

#### P3-PORT-03. Port remaining applicable documentation fixes

- [ ] **Apply the remaining parts of `ff1f29d9`. Size S, complexity Low.**
- `docs/source/preparation.rst` still says initial DI calibration must be in
  `DATA` rather than `data_colname`; `Sector.set_imaging_parameters()` lacks
  the literal-block markup for its expected keys.

#### P3-DEC-02. SageCal/libdirac source compatibility

- [ ] **Decision. Size M, complexity Medium.**
- Master pins `33d21c45000bf13e5e29077ba3413405c42c503f` in
  `Docker/fetch_commit_hashes.sh`; `main` resolves upstream `HEAD`. Build and
  test the intended tuple before deciding; keep resolver, labels and cache key
  consistent.

#### P1-DEC-02. SKA-Low configuration compatibility (Extra review)

- [ ] **Decision; relevant to SKA users. Size M, complexity Medium.**
- Master ships `rapthor/settings/defaults_skalow.parset`; the migration removed
  it. The master file references an absent `custom_ska_low.py`, so restoring it
  unchanged is not a verified port. Decide between an in-repository preset or
  the ICAL configuration, document the migration and test its parsing.

#### P3-BUG-01. Shared-facet I/O tool failure (existing limitation)

- [ ] **Size M–L, complexity High (depends on WSClean).**
- `tests/integration/test_shared_facet_rw.py` marks the enabled case
  `xfail(run=False)` because serial WSClean 3.7 aborts. Reproduce with the
  supported stack, fix or constrain support, and run the enabled test.

### Changes Already Covered or Superseded

Keep these out of the port backlog unless the merged tree loses the behavior.
This table records source-review findings, not fresh runtime validation.

| Master change | Evidence / disposition on `main` |
| --- | --- |
| Reduced frequency-BDA default (`a998d5f7`, former review task P06) | `d0e7e148` on `main` sets `bda_frequencybase = 5000.0` for calibration and imaging in both `defaults.parset` and `defaults.json`. Verified 2026-10-14. |
| Sky model download error handling (`9fc97dc3`, GEC-656) | `rapthor/execution/image/diagnostic_calculation.py:safe_load_skymodel` is already the plain function returning `None` on `OSError`, with the corrected log message. Master's new tests in `tests/scripts/test_calculate_image_diagnostics.py` have no counterpart; add equivalents under P3-TEST-02 if desired. |
| Release preparation and CWL Docker tags (`b9c98086`, `f432c66d`, `ef76dd5c`), Toil tmpdir option (`76a7ff2f`) | CWL/Toil-only or version bookkeeping; not applicable. |
| Balanced observation chunking (`b8312075`) | Ported by `e74be379`. The chunking methods in `rapthor/lib/observation.py` match master; lifecycle wiring and real-MS boundary tests are present. Do not reapply older GEC-629 chunking patches. |
| LSMTool release dependency/API (`70b561e5`, `9fa90768`) | `2168c1c6` sets `lsmtool>=1.9.0`; field/region helpers use `read_from_skymodel`. |
| Example strategies and initial-model documentation (`302714f8`, `4d62d3a8`) | The four updated custom/default calibration/imaging example files already match master. The strategy docs explain the initial-model-dependent phase-only cycles. |
| Time-ordered concatenation (`bf4608ef`), relaxed station diameter (`56a43f84`) | Present through `e7d2727a` and `d201dded`. Mixed station *sets* still need P2-PORT-03. |
| Facet RMS, missing/invalid regions, off-image facets (`eb1b6f2f`, `908f83c9`, `da442dfc`) | Runtime behavior exists in the image execution owner (`ffa64f6d`, `f3617f41` and subsequent fixes). Only the test/fixture gaps in P3-TEST-02/P2-TEST-02 remain. |
| Astrometry corrections (`ebe35408`) | Ported by `c0127ac6`, with subsequent error handling and survey-fallback fixes including `b5884ade`. |
| Later-cycle normalization and empty-source handling (`fc79ef7f`, `3e4eca19`) | Implemented in strategy/normalization execution and tests; `5974f354` records the catch-up. Supplied-survey metadata still needs P2-PORT-02. |
| Basic WSClean prediction and configurable narrow-band models (`d90786e8`, `e8867abd`, `c4dfcfd8`) | Present through `5c3c9471` and `d201dded`; end-exclusive channels fixed by `706556f6`. Remaining changes are P2-PORT-01/P3-PORT-01/P3-DEC-01. |
| Model-data and residual-visibility products (`971a2b25`, `17448437`) | Ported by `d51e08a2`, including current output records/finalizers. |
| Parallel gridding/shared-facet selection (`38abdb92`, `01a81e11`) and per-sector input shape (`18cf2d72`) | Current `Image` builds one gridding value per sector; builders preserve that mapping. Selection behavior was ported by `5974f354`/`d201dded`; remaining integration coverage is P3-TEST-03/P3-BUG-01. |
| Flexible strategies and image-only application (`0ebc0690`) | Present with explicit execution validation and image-only integration cases. Only the support decision P2-DEC-01 remains; retain documented cycle/seeding differences. |
| Imaging frequency BDA and averaging limits (`c6f4375b`, `15d4ccb0`, `3fd9e69f`) | Implemented with operation, command and integration coverage. |
| Calibration-memory preflight/per-cycle checks (`043c15d4`) | Ported by `4fd75842`; unit/integration coverage remains. Node-wide scheduling budgets in B1 are additional work. |
| Reset with missing directories (`e25b4a6a`) | `rapthor/modifystate.py` skips absent product directories. |
| Ubuntu 24.04, libdeflate, current measures-table URL (`37f6fc06`, `0e163498`, `bc2c65a7`, `d1fd7249`) | Present. Do not restore `488f5c00`'s temporary fallback URLs superseded on master. `b27ba6e6`'s streamed download is an implementation difference. Outstanding build differences are P2-PORT-07/P1-PORT-01/P3-DEC-02. |
| Earlier CI fixes (`c853f707`) | Test sharding/thread-environment settings and separate supplied/downloaded normalization cases are present. Remaining CI changes are P3-TEST-01. |
| CWL-only fixes (`b457f49c`, `c28c9ae2`, `e8873f19`) | Duplicate CWL fields, `pickValue`/nullable outputs and staging symlinks belong to retired workflows. |
| Calibration `avg` → `bdaavg` rename (`dbb8993c`) | Both versions explicitly use `bdaaverager`; this is a step-label change only. |
| Formatting (`f826c262`, `13e3bd8e`, `6c74e20d`, `f379cf15`), pytest/helper refactors (including `0c422261`) and obsolete DP3 `writefullresflag` removal (`b307e769`) | Covered or superseded by the new modules/tests. Relevant remaining fixture/CI changes are P3-TEST-01/P2-TEST-02. |

## Part B — Future Improvements

After recording the merged baseline, enforce resource budgets before increasing
concurrency; then validate cluster deployment, profile real workloads and
extend coverage. Independent work such as a persistent Prefect service or
catalog caching can proceed alongside resource allocation. Coordinate with
Part A where a port changes the same execution path or test environment.

Use the dedicated resource and WSClean configuration branches listed in B1 as
the sources for this work. They separate these changes from the temporary CLI
overlays, which can be retired after comparison testing. Integrate the dedicated
changes onto merged master. The GEC-629 filtering/WSClean environment fixes and
newer chunking implementation are already on this branch.

### B1. Resource Allocation and Safe Concurrency

These are future improvements drawn from other development branches, separate
from the master ports in Part A. Review and integrate the existing implementations:

- `gec-618-improve-resource-allocation`: separate DP3/WSClean thread limits
  (`61157274`) and CPU/memory allocation, command-thread validation and Dask
  worker-layout checks (`7d575dac`).
- `gec-618-improve-multinode-wsclean-configuration-logic`: per-sector
  gridding/shared-facet selection based on channels, facets, node count and
  WSClean's actual thread count (`1cc7f4d5`).

Reconcile the overlapping imaging changes when combining these branches. Use
the following as an acceptance checklist and implement remaining gaps in
dependency order:

1. **Define and enforce a node-wide CPU/memory budget.** Account for every Dask
   worker and external subprocess. Divide worker memory limits within the node
   allocation; giving each worker the full node limit multiplies the assumed
   capacity. Dask memory limits alone do not constrain subprocess RSS.
2. **Apply per-tool limits and schedule heavy tasks within that budget.** Extend
   the existing environment policies to DP3 solves, prediction, applycal and
   Python adapters. Make command thread flags and native thread pools agree
   with the task allocation. Preserve existing filtering caps and WSClean's
   single OpenBLAS thread per MPI rank. Keep MPI imaging exclusive within its
   allocation and make resource requests affect scheduling.
3. **Choose safe local worker defaults.** Derive worker counts from available
   cores and memory, bounded by configured allocations. Keep one Prefect
   task-engine thread per worker process. Light tasks should be able to proceed
   without allowing several heavy commands to claim the same resources.
4. **Revisit parallel gridding and shared-facet selection.** Base decisions on
   actual resources and external-tool capabilities, including MPI layouts.
5. **Evaluate NUMA-aware execution with two Dask workers per node.** Review and
   link the existing work on improved WSClean resource usage, and assess DP3
   separately. On suitable hardware, bind each worker and its subprocesses to
   a distinct NUMA domain's CPUs and local memory, with thread and memory
   budgets derived from that domain's allocation. Check the actual topology;
   two workers alone do not establish memory locality. Update the proposed
   resource branch's external-Dask validation, which currently assumes one
   worker per host, to support two workers with disjoint resource allocations.
   Verify subprocess affinity and MPI rank placement, and prevent concurrent
   commands from claiming the same resources. Compare one worker per node,
   two unbound workers and two NUMA-bound workers using the same total node
   allocation and representative WSClean and DP3 workloads. Record CPU usage,
   peak memory, elapsed time, throughput and memory locality/bandwidth where
   measurable; compare scientific products and failure/restart behavior.
   Document the tested topology, Slurm/Dask launch configuration and per-tool
   results before recommending this layout or changing defaults.

Verify one-worker, multiple-worker and MPI configurations with real command
execution. Record total CPU usage, peak memory and elapsed time, and confirm
that products and failure/restart behavior remain consistent. Increase default
concurrency only when the combined workload stays within its allocation.

### B2. Cluster Deployment and Monitoring

- **Persistent Prefect service:** provide a managed Postgres-backed service for
  users who need one dashboard and durable history for independent jobs.
  Document startup, configuration and recovery. Keep isolated ephemeral state
  available for standalone runs; shared SQLite is unsuitable for the intended
  concurrent production service. P1-TEST-01 establishes the supported external-Dask
  API/worker configuration; P1-BUG-03 bounds publication volume before durable history.

Keep Ubuntu 24.04 as the supported container baseline while Ubuntu 26.04's
reported memory problems are investigated separately.

### B3. Measured Performance Improvements

First make fallback measurements trustworthy (Medium review). In
`execution/shell.py`, `_resource_usage_metrics()` uses process-wide
`RUSAGE_CHILDREN` deltas when GNU `time` is unavailable, so concurrent children
can contaminate a command's CPU/I/O figures. Its `ru_maxrss` is also a lifetime
high-water mark, even for sequential commands. Use per-child accounting where
available or label/omit unsupported metrics; test overlapping commands and a
small command after a high-memory one. The normal single-task worker policy
reduces overlap, but does not fix the lifetime memory measurement. Do this
before using fallback numbers to justify resource/performance changes.

Profile representative real-data runs before choosing changes. Re-measure
`filter_skymodel` after its environment fixes, then inspect WSClean imaging,
command resource use, I/O and task wait times. Optimize calibration plotting
only if it remains a meaningful cost on larger runs.

Cache reference catalogs per run to avoid repeated survey queries across
sectors and cycles. Key cached data by the query and sky coverage, reuse
supplied catalogs, and preserve offline, empty-result and fallback behavior.
Avoid unnecessary temporary-FITS round trips where profiling shows a benefit.

Benchmark WSClean multi-band and frequency-BDA workloads before extending
performance claims to them. For each optimization, record input/configuration,
revision, dependency versions, elapsed time, peak memory and product comparisons.
Use repeated runs when normal scatter could obscure the result.

#### Parallelization Investigations

These opportunities are investigations, not promised speedups. Establish the
resource controls above before increasing concurrency. Start with prediction/
solve overlap and residual/image overlap; investigate the remaining work where
profiling shows a useful benefit.

| Priority | Opportunity | Required boundary and comparison |
| --- | --- | --- |
| High | Overlap chunk prediction and calibration solving | Submit each solve against its own prepared-chunk dependency and shared prerequisites. Preserve solve order within a chunk and collect only after all chunks finish. Compare multi-chunk WSClean-prediction runs. |
| High | Overlap residual-MS creation and image post-processing | Start independent image products after WSClean completes, join residual and image records at finalization, and wait for all MS readers before cleanup. Compare residual-enabled runs. |
| Medium | Build requested Stokes cubes independently | Bound simultaneous dense allocations and reads, preserve specification order, and attach catalog dependencies only to the required cube. Compare I/Q/U/V products. |
| Medium | Run independent diagnostic checks concurrently | Give substantial photometry, astrometry and RMS work distinct outputs and plotting processes, followed by one JSON writer. Preserve offline, empty-catalog and survey-fallback behavior. |
| Medium | Compare astrometry facets concurrently | Return plain offset/statistics records and retain ordered reduction, query limits, skipped-facet handling and averaging semantics. Cache catalogs where their footprint permits. |
| Conditional | Render prediction bands concurrently | Keep a single writer per MS. Concurrent predictions require isolated outputs and a deterministic merge; measure the additional memory and scratch costs. |
| Low | Regrid mosaic sectors concurrently | Use distinct product/sector paths and bound memory and filesystem traffic. Compare multi-sector mosaic products. |
| Low | Compress independent images or batches | Keep outputs disjoint, preserve compression semantics and measure storage throughput. |
| Low | Plot slow-gain phase and amplitude concurrently | Establish per-task PNG ownership, use independent plotting processes and preserve safe read access. Compare plots and h5parm products. |

Keep worker payloads plain and serializable. Preserve ordered self-calibration
cycles, strategy solves and writes to shared MS/h5parm products. Cleanup must
wait for every consumer. Compare products, memory, I/O and elapsed time with one
and multiple workers before adopting a concurrency change.

### B4. Coverage and Maintenance

- **Broader scientific paths:** add representative SKA-Low and screens/IDGCal
  coverage as their target environments become available. Historical
  equivalence evidence covers selected LOFAR HBA paths.
- **Multi-sector mosaic:** maintain targeted smoke/product coverage. Give it
  lower priority than common imaging unless a failure affects shared contracts.
- **Legacy strategies:** document the transition to `calibration_strategy` and
  schedule removal of the `do_slowgain_solve`/`do_fulljones_solve` translation.
  Replace deprecated flags with an actionable error after the documented
  deprecation period, and retire the compatibility-only strategy helper.
- **Module and test cleanup:** simplify large diagnostics, normalization,
  h5parm-combination and operation modules when maintenance or behavior changes
  justify it. Consolidate shared test helpers where useful. Preserve guards for
  serializable payloads, thin operation adapters, calibration semantics,
  image-only application, restart and scientific products.
  Use the GEC-486 helper work (`5f59fe87`, `6c58212b`, `14377a61`, `4cc0b732`)
  where it removes duplication; scope integration-only fixtures to integration
  tests. Current pytest conversions and repository-wide formatting already
  cover the corresponding master changes.

## Tracking and Evidence

Use one issue/merge request per focused task. Record its source commits, chosen
behavior, dependencies and verification; split a task if its parts can be
reviewed independently. Close a Part A item only when its acceptance checks pass
or a deliberate compatibility decision is recorded. Remove completed tasks as
work lands and keep new improvements in Part B.

Follow [TESTING.md](TESTING.md). Scientific, default or scheduling changes need
product comparisons against the recorded baseline. Update both applicable
defaults, parsing/domain state, operation inputs, execution payloads, validators,
commands, docs/examples and test templates together for option changes. Keep
architecture diagrams aligned with ownership and execution changes.

Manual testing supersedes the historical migration equivalence reports. The
[upgrade guide](docs/source/upgrading.rst) records intentional behavior and
output differences. Keep new verification results with the relevant issue or
merge request, without committing large Measurement Sets or raw products.

### MR-473 Follow-Up Verification (2026-10-09)

The evidence above combines source tracing with isolated probes in the prepared
dev container (Python 3.12, NumPy 2.4.6, DP3 6.6.0,
`v6.6-136-g2cf52e0f`). All generated MSs, FITS files and h5parms were temporary;
the existing test MS was read as input only. No production code was changed.

- **Real DP3 probes:** a one-time, two-channel MS selection and synthetic
  nonzero station phases reproduced P0-BUG-06 (0.07 to 0.14 radians). For P0-BUG-01,
  beam-disabled point-source prediction with constant gain amplitude 2 yielded
  ratios 1 with the current command and 4 with `usemodeldata=False`. This tests
  buffer application, not full beam/scientific equivalence or every predictor.
- **Python/helper probes:** reproduced DD amplitude-state loss, rejection of
  carried DD solutions, strategy carry-over/aliasing, the phase-only DD active
  product, missing WSClean pre-application, Slurm guard bypass/environment
  omission, partial-directory acceptance and concatenation rejection. Simulated
  overlapping writers reproduced plot and predict-record contamination; an
  unchanged image was published twice. Empty-model stubs hit the mosaic error.
- **Process/filesystem probes:** a second real `fpack` invocation returned 255;
  reset-style `rmtree` left a directory symlink; a command child survived an
  interrupted shell runner and was explicitly terminated by the probe.
- **Existing focused tests:** 69 passed, 14 deselected, one ERFA dubious-year
  warning. Command (with an isolated `RAPTHOR_TEST_RUN_ROOT`):
  `python -m pytest -q -m 'not integration and not prefect' tests/execution/test_config.py tests/execution/test_slurm.py tests/execution/test_image_preparation_helpers.py tests/execution/test_mosaic_model_rendering.py tests/execution/test_predict_flow.py -k 'not run_'`.
  Passing existing tests does not close the newly reproduced gaps.
- **Not run:** full calibration/normalization comparisons, real multi-sector
  mosaicking, remote-Dask Prefect propagation, Slurm/MPI execution, the full
  field/Prefect/integration suites, or fresh verification of every existing
  master-port task. These remain explicit acceptance work above. The earlier
  reviews' test counts and remote-worker probe are not results of this follow-up.

### P0-BUG-01 Fix (2026-10-14)

- Real DP3 (`v6.6-136-g2cf52e0f`) in the dev container, test MS copy, point
  source at the phase centre, normalization 4 in the production h5parm layout:
  new command gives a visibility ratio of 4.0000 for `predict` and for
  `h5parmpredict` with a unit DD h5parm; the previous command gives 1.0 in both
  cases (`tests/execution/test_predict_dp3.py`, marked `integration`).
- SageCal probe with the same inputs: `predict.directions` without an h5parm
  fails ("H5parm name not specified but directions specified"), and three
  `applycal` sub-steps fail ("Only one or two types of gain application can be
  done at a time"). With a unit h5parm and two sub-steps the container's
  SageCal predictor returned all-zero visibilities; not investigated further,
  as SageCal is optional and the behaviour is independent of this fix.
- Checks run: `tests/execution/test_predict_flow.py`,
  `tests/execution/test_commands.py`, `tests/operations/test_predict.py`
  (`not integration and not prefect`): 100 passed, 5 deselected.

### P0-BUG-02 Fix (2026-10-15)

- DP3 6.6.0 (`v6.6-136-g2cf52e0f`) in the dev container; DATA set to the
  predicted point-source model; normalization 4 (amplitude 2) in the
  production h5parm layout; a fresh MS copy per case because DP3 updates the
  input MS in place when `msout=` is empty (an earlier probe round was
  contaminated by this and discarded). Diagonal solve amplitudes, expected 1.0
  for an untouched model and 0.5 for a model scaled by 4:

  | Case | Amplitude |
  | --- | --- |
  | sourcedb solve, no normalization | 1.0 |
  | sourcedb solve, `solve.applycal.steps=[normalization]` | 0.5 |
  | solve1 keepmodel, solve3 reusemodel, normalization on solve3 only | 1.0 |
  | normalization on solve1 only, solve3 reuses the model | 0.5 |
  | normalization on solve1 and solve3 | 0.5 |
  | `modeldatacolumns=[MODEL_DATA]` solve, internal normalization | 1.0 |
  | stand-alone applycal slowgain (amp 2) then modeldatacolumns solve | 0.5 (data only) |
  | stand-alone applycal normalization `usemodeldata=True`, then column or sourcedb solve | 1.0 |

  Master's chain (`solve1.keepmodel=True`, `solve3.reusemodel=[solve1.*]`,
  normalization on `solve3` only) is the third row. An earlier probe of
  `solve3.modeldatacolumns=[solve1.*]` was mislabelled "master style"; master
  does not use that syntax and the result is withdrawn.

- Checks run: `tests/execution/test_calibrate_dp3.py` (2 passed; 1 fails
  against the old `solves.py`), `tests/operations/test_calibrate.py` and
  `tests/execution/test_calibrate_flow.py` (160 passed), command, payload and
  architecture tests, Prefect-marked calibrate flow tests (12 passed), ruff.

### Release Recheck (2026-10-14)

Source inspection of `main` at `d0e7e148` against `master` at `76a7ff2f`
(`v2.2rc1` plus two commits). No code was changed and no new real-tool probes
were run; the 2026-10-09 probe results above remain the runtime evidence.

- **Branch comparison:** 69 master-only and 2 main-only commits since
  `2e21be62`. Of the five master commits after `ff1f29d9`, only GEC-656
  (`9fc97dc3`) touches behaviour, and `main` already contains the equivalent
  `safe_load_skymodel` function. `d0e7e148` on `main` closes review task P06.
- **Confirmed present on `main` by source:** P0-BUG-06 (`operations/image/plan.py`
  step mapping and missing `applycal.mediumphase.*` settings), P0-BUG-03
  (`operations/calibrate/base.py:945`, `lib/field.py:1727`,
  `operations/image/base.py:301`), P0-BUG-01 (`execution/predict/commands.py`
  stand-alone `applycal` with `usemodeldata=True` after `operation=replace`),
  P0-BUG-02 (`operations/calibrate/plan.py:206`; slot dictionaries in
  `execution/calibrate/solves.py` lack `applycal_steps`/`normalize_h5parm`),
  P1-DEC-01 (`_dd_active_solution()`), P0-BUG-04 (`Field.update()` and
  `set_calibration_strategy()`), P0-BUG-07 (`lib/strategy.py:228`), P0-BUG-05
  (`execution/image/preparation.py:27,70`, `execution/concatenate/flow.py:90`,
  `execution/image/outputs.py:116`, `execution/mosaic/flow.py:310`), P1-BUG-01
  (`execution/shell.py` `Popen` without a process group), P1-BUG-03
  (`execution/pipeline/flow.py:443`, `execution/artifacts.py:39`), P1-BUG-02
  (`execution/runtime_bootstrap.py:217`, `execution/config.py:146`,
  `execution/slurm.py:27`), P1-TEST-01 (`docs/source/running.rst:299`), P0-BUG-08, P2-BUG-02,
  P1-BUG-04, P2-BUG-03, P2-BUG-04, P2-BUG-05, P2-PORT-01, P2-PORT-03, P2-PORT-02, P2-PORT-05, P2-PORT-07, P1-PORT-01, P2-PORT-08 and P2-TEST-01.
- **Corrections to earlier text:** P0-BUG-02's payload gap is limited to the
  DDECal-internal normalization on the slow-gain solve; the stand-alone
  `applycal` step is wired (`operations/calibrate/base.py:691`). P0-BUG-06, P1-DEC-01 and
  P0-BUG-07 are also present on master and are not regressions.
- **Master references added:** each Part A task now names the master file and
  function implementing the correct behaviour where one exists. For P0-BUG-05, P1-BUG-01,
  P1-BUG-02, P1-TEST-01, P1-BUG-03 and P0-BUG-08 master relied on Toil/CWL and there is no portable code.
