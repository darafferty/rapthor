# Post-Merge Development Plan

Updated: 2026-10-20. Release baseline: `origin/main` at `98a06302`;
CWL reference: `origin/master` / `origin/rapthor-cwl` at `b83cf9bc` (v2.2).

This is the actionable backlog for the first Prefect/Dask release and later
improvements. Task IDs are unchanged. **Merged** means present on `main`;
feature-branch implementations remain open until reviewed, validated and merged.

## Recommended next item

**P0-BUG-05: restart safety for image preparation, concatenate and mosaic.**
No reviewed outstanding branch implements this fix. On `main`,
`prepare_visibility_ms()` and `concatenate_prepared_visibilities()` accept an
existing output directory as complete, including one left by a failed command.

1. **First MR:** fix those two helpers in
   [image/preparation.py](rapthor/execution/image/preparation.py). Reuse only
   validated, completed outputs belonging to the requested inputs/settings;
   recover incomplete owned outputs without changing original MS inputs.
   Test failure during creation, after creation but before completion is
   recorded, and a second successful restart with complete output records.
2. Extend the same contract to standalone concatenation and image/mosaic
   compression. Existing outputs currently cause concatenation to fail, and
   compression is rerun without accounting for existing compressed products.
3. Resolve P0-BUG-08: reserve/remove the ineffective `prefect_retries` option,
   or implement it only after repeatability is established.

**Merge existing work:** the two GEC-671 strategy fixes and GEC-664's scaled
integration fixture are already implemented; review and validate their combined
state rather than implement them again. GEC-678 input-MS protection is merged.
Resolve the separate phase-only DD product decision (P1-DEC-01) before release;
neither strategy fix changes that selector.

## Release status

Five P0 fixes are merged; **four P0 items remain**: two implemented on the
unmerged GEC-671 branches, restart safety, and the retries-option decision.
P1 items gate the features named below. P2/P3 work can follow release if
applicable limitations are recorded in the [upgrade guide](docs/source/upgrading.rst).

### Merged

| Task | Result | Merge |
| --- | --- | --- |
| P0-BUG-01 | Prediction normalization acts on the predicted model. SageCal normalization remains unsupported; use `sagecalpredict = False`. | `83f056c7` (GEC-669) |
| P0-BUG-02 | Calibration receives DI pre-application and normalization on the predicting solve. WSClean model-column normalization remains unsupported by the tested DP3 6.6 path. | `d3f236dc` (GEC-669) |
| P0-BUG-03 | DI and DD amplitude-application state is tracked separately. | `57f90ab7` (GEC-670) |
| P0-BUG-06 | Single-direction imaging applies the combined phase table once. | `57f90ab7` (GEC-670) |
| P0-BUG-09 | DD solve chains end with `null`, protecting input DATA/FLAG; DDECal-only DI behavior is retained. Includes command and real-DP3 regression checks. | `98a06302` (GEC-678 implementation; merge title says GEC-669) |

GEC-678's feature tip `94003846` and `main` at `98a06302` have identical trees;
no rebase or implementation remains for P0-BUG-09. GEC-670 does **not** fix
phase-only DD product selection (P1-DEC-01) or image-only solution/layout
carry-forward (P2-BUG-01).

### P0 — release blockers

| Task | Status / owner | Required result |
| --- | --- | --- |
| P0-BUG-07 | **Implemented, unmerged:** `gec-671-fix-final-cycle` at `4b4bd2f3`. | Deep-copy the final step, including nested solve lists: default final/last-selfcal widths remain 4/8 MHz. The differing steps trigger a final pass even with equal data fractions. Tests and migration notes included; CWL remains unfixed. |
| P0-BUG-04 | **Implemented, unmerged:** `gec-671-fix-calibration-strategy` at `69656ceb`. | Resolve absent/`None`/empty strategies per calibrating cycle; keep omissions intact at load time so the default remains identifiable. Legacy flags are translated first, including explicitly false flags. Image-only omissions retain the previous solve list. Preflight, logging, tests and docs are updated. |
| P0-BUG-05 | Open. Image preparation/compression, concatenate and mosaic execution; large. | Never accept an incomplete MS because its directory exists. Make owned outputs recoverable and compression repeatable. Inject failures during/after product creation and restart with complete records; never overwrite original inputs. |
| P0-BUG-08 | Open; depends on P0-BUG-05 if implemented. | `prefect_retries` is parsed but ineffective. Remove it, or explicitly reserve it with a warning; implement only for safely repeatable tasks and verify actual attempt counts, cleanup and records. |

### P1 — decisions and feature-specific release gates

| Task | Scope / owner | Required result |
| --- | --- | --- |
| P1-DEC-01 | Phase-only DD cycles; `Calibrate._dd_active_solution()`. | Decide whether to apply combined fast+medium phases or omit the unused medium solve. Test selected product contents, restart and downstream prediction/imaging; update products/science documentation. |
| P1-BUG-01 | Cancellation, especially clusters; `execution/shell.py`. | Terminate command descendants with a process-group TERM/KILL sequence. Bound captured output. Test Prefect cancellation, worker shutdown and MPI wrappers. |
| P1-BUG-02 | Slurm; bootstrap/config/slurm owners. | Validate the requested runner before bootstrap rewrites it; resolve the documented one-worker default and read Slurm environment values by default. Test CLI, precedence and missing-scheduler cases. |
| P1-BUG-03 | Long dashboard-enabled runs; artifacts/pipeline owners. | Deduplicate publication, bound bytes, provide a plot-publication switch and make monitoring errors warning-only. Test repeated operations and measure database growth. |
| P1-BUG-04 | Multi-chunk prediction; `execution/predict/flow.py`. | Return exact chunk-owned paths instead of broad shared-directory globs. Test two chunks with DI addition, DD subtraction, peeling and restart. |
| P1-PORT-01 | Standalone container; Docker/CI builds and sector subtraction. | Align NumPy build/runtime, replace `np.NaN`, and verify NumPy/PyBDSF/casacore imports plus source finding in final images. |
| P1-TEST-01 | Multi-node Slurm; running docs/runtime. | Validate a reachable shared Prefect API, remote-worker settings, real calibration and MPI imaging, rank placement and resource exclusivity. Document tested configurations or reject unsupported ones. |
| P1-DEC-02 | SKA-Low configuration. | Choose an in-repository preset or ICAL interface; test and document migration. Do not restore CWL's `defaults_skalow.parset` unchanged: it refers to an absent custom script. |

**P1-DEC-01 clarification:** both CWL and current Prefect select the first
fast-phase product for `[fast_phase, medium_phase]`, although a combined product
is generated. The medium contribution is discarded during finalization, not
optimizer seeding. With slow gains, combined phases are already selected and
need only one phase-application step; this is not a SageCal-specific issue.
Tests starting from prepared h5parms cannot detect the selector bug. Use distinct
fast/medium values and inspect the selected `field-solutions.h5`, including
restart, fast-only and phase-plus-slow-gain cases.

## Remaining ports and fixes

Each row is a separate task. Preserve the stated acceptance condition when
splitting work into issues. Existing branch implementations are mapped below.

### Strategy and solution products (WS-B / WS-C)

| Task | Remaining work |
| --- | --- |
| P2-BUG-01 | Associate applied h5parms with their producing sky-model/facet layout; preserve that association across consecutive image-only cycles. GEC-671 preserves the solve list, not this product/layout association. Align prediction and imaging without weakening optimizer-seed rules. Test multiple sectors, outliers, peeling/reweighting and repeated finals. |
| P2-DEC-01 | Inventory CWL solve sequences rejected by Prefect. Port required combinations with solve/application/product tests, or document deliberate restrictions; parsing alone does not establish support. |

### Restart and output ownership (WS-D)

| Task | Remaining work |
| --- | --- |
| P2-BUG-02 | Give calibration plot tasks explicit paths or manifests. Concurrent tasks and reruns must return all and only their own plots. |
| P2-BUG-03 | Preserve peeled data across repeated final passes and discover actual cycle directories during reset. Test early termination, changed repeat counts, selected MS/model and both `pipeline_outputs.json` and `.outputs.json`; retain earlier completed work. |
| P2-BUG-04 | Reset owned scratch links, targets and recovery records together; never follow unowned links. Rotate/tag aggregate logs by attempt. Test staged, partial and selectively reset workspaces. |
| P2-BUG-05 | Preserve per-sector astrometry-offset JSON through finalization, cleanup and restart, or explicitly document retirement. Test multi-sector names. |
| P2-BUG-06 | Define rendered versus regridded model-mosaic semantics. Handle mixed/all-empty models with valid records, compare products and document the default. |

### Logging and failures (WS-E)

| Task | Remaining work |
| --- | --- |
| P2-PORT-05 | Remove duplicate console messages, ANSI codes in files and import-time logging configuration. Preserve module loggers, Prefect handlers and durable worker diagnostics; test repeated setup and API shutdown. |
| P2-PORT-06 | Carry exit code, log path and bounded diagnostic tail through Prefect/Dask exceptions to the CLI. Coordinate with P1-BUG-01; test a failing external command. |

### WSClean prediction (WS-F1)

| Task | Remaining work |
| --- | --- |
| P2-PORT-01 | Solve beam selection is implemented on `correct-calibration-beam-application`; image-based `applybeam.beammode=array_factor` remains. Port WSClean full-beam prediction (`23ed80e9`): `wsclean_predict_beam_interval`, facet-beam flags and full-beam model images, with DP3 beam modes coordinated to prevent double application. Test both `use_array_beam` settings and compare DP3/WSClean multi-facet fluxes. |
| P2-DEC-02 | Keep the WSClean/BDA guard until averaged data/model rows and channels, beam handling, input preservation and restart are validated. Otherwise document the limitation and resource cost. |
| P3-PORT-01 | Adapt rendering geometry and reordered-data reuse (`d587eec5`). Keep scratch per chunk until all consumers finish. Test single/multi-band and single-channel coverage, visibility agreement, resources and end-exclusive channel ranges. |
| P3-DEC-01 | Validate named DS9 facets with supported tools; introduce separate point/polygon formats only where needed. Match facets to model columns and h5parm directions. |

### Catalogs and data layout (WS-F2)

| Task | Remaining work |
| --- | --- |
| P2-PORT-02 | Restore supplied-survey flux/error metadata and the missing JSON reference-frequency default (`dfa7f11a`, `6d4df857`). Supplied/downloaded versions of a survey must normalize consistently; cover unknown frequencies and offline inputs. |
| P2-PORT-03 | Support mixed station sets (`4c00305d`) with a deterministic union and per-MS imaging inputs instead of unconditional time concatenation. Disable or validate residual-MS creation; cover serial/MPI, normalization and restart. |
| P2-PORT-04 | Provide a maintained survey-download preparation command for offline runs. Document pointing/radius, catalog paths and frequencies; test failures and a normalization/diagnostics run with internet disabled. |

### Build, tests and documentation (WS-G)

| Task | Remaining work |
| --- | --- |
| P2-PORT-07 | Fix Ubuntu 24.04 APT source handling in build/runtime stages and CI (`a39e517b`, `d1fd7249`); use `ubuntu.sources`. This does not adopt Ubuntu 26.04. |
| P2-PORT-08 | Reconcile Python/package metadata, license/build requirements and tox access to compiled astronomy dependencies; verify the supported Prefect environment. |
| P2-TEST-01 | Mark real-WSClean restoration tests as integration and verify non-integration selection excludes them. |
| P2-TEST-02 | Reset flags and weights in the synthetic normalization MS before prediction (`da442dfc`). GEC-664's scaled integration fixture and h5parm assertions are separate improvements; this reset remains unimplemented. |
| P3-TEST-01 | Astron/SKA CI separation is implemented on `sync-with-master`. Duration-balanced shards, bounded CPU counts and measured test-speed improvements remain. Keep field/Prefect-server tests serial with isolated state. |
| P3-TEST-02 | Add real-run facet-RMS JSON assertions and the missing sky-model-download failure cases; runtime handling is already present. |
| P3-TEST-03 | Restore the task-count 1/2, full/single DDE and serial/MPI gridding matrix using command records. Adapt two-sector cases without CWL mocks. |
| P3-PORT-02 | Helper and usage docs are implemented on `sync-with-master` (`74e89904`). Add a fake-Docker test covering labels, missing labels and inspect failure, then validate and merge. |
| P3-PORT-03 | Merge current-branch documentation links, correct `DATA` versus `data_colname` guidance and fix the imaging-parameter docstring markup. |
| P3-PORT-04 | Adapt CWL's installed run/log-archive wrapper (`6052b19d`) to Prefect logs and supported environments. Preserve exit status, exclude MS/large products, and test partial logs, optional inputs and unusual paths. Document Bash/GNU requirements. |
| P3-DEC-02 | Choose and validate the SageCal/libdirac dependency tuple; coordinate resolver, mirrors, labels and cache keys. CWL v2.2's dependency file is a comparison baseline, not Prefect validation. |
| P3-BUG-01 | Reproduce WSClean's shared-facet I/O failure on the supported stack; fix or constrain support, then run the currently disabled integration case. |

## Existing branches to integrate

These implementations are not yet merged into `main`. The two GEC-671 branches,
scaled test data and calibration beam branch already descend from `98a06302`.
Rebase older branches using focused changes; avoid replaying pre-squash migration
history or temporary `rapthor3` CLI overlays. CWL branches are porting sources,
not implementations in the Prefect version.

| Branch / reviewed tip | Disposition and next action |
| --- | --- |
| `gec-671-fix-final-cycle` — `4b4bd2f3` | P0-BUG-07 implemented with strategy/lifecycle tests and docs. Review and merge. |
| `gec-671-fix-calibration-strategy` — `69656ceb` | P0-BUG-04 implemented with field/strategy/preflight regressions and docs. With no legacy flags, the fallback is the full DD sequence; presence of either flag selects legacy translation. Merge alongside the final-cycle fix and check the combined strategy behavior. |
| `gec-664-scale-test-ms-data` — `6ec95d2f` | Integration fixture DATA scaled by `1.5e4`, weights scaled inversely by its square; original inputs remain untouched. DI and DI+DD tests require expected h5parm tables with mostly valid solutions. Review/merge; does not close P2-TEST-02 or the phase-only selector decision. |
| `correct-calibration-beam-application` — `b9c80730` | Partial P2-PORT-01: propagates `use_array_beam` through calibration inputs/payloads and selects the solve beam, with command coverage. Image-based applybeam and the WSClean facet-beam port still need work. |
| `sync-with-master` — `74e89904` | Partial P3-TEST-01/P3-PORT-02: Astron/SKA CI split, dependency-label helper/docs and v2.2 changelog. Adds Ubuntu 26.04 build files, but CI still selects 24.04. Do not count this as resolving 24.04 APT, NumPy compatibility, test performance or adopting 26.04. |
| `RAP-1392-skip-outlier-subtraction` — `21657525` | Partial P2-BUG-03: repeated-final state and reset enumeration. Add focused peeling/reset regressions and actual-directory discovery. |
| `RAP-1444-get-BDA-parms-no_cal` — `5102840d` | Supplied-h5parm interval inference related to P2-BUG-01. Reconcile with GEC-670 scanning; test single-sample/irregular time axes. Does not fix direction-layout ownership. |
| `fix-references-to-master-branch` — `8c451a08` | Partial P3-PORT-03; preparation guidance and docstring fixes remain. |
| `use-local-git-mirror` — `0e318a61` | Build/CI URL changes; verify mirror refs and reproducibility. Does not fix NumPy/APT gaps or pin `HEAD`. |
| `RAP1418-enable-averaging` — `e5e14fdd` (CWL) | P2-DEC-02/P3-PORT-01 candidate; now includes a BDA on/off WSClean-prediction integration case. Adapt averaged-MS outputs, serializable records, scratch lifetime and calibration consumers; retain the Prefect guard until validated. |
| `add-multi-sector-tests` — `b7537329` (CWL) | Source for P3-TEST-03/B4; adapt per-sector assertions to Prefect records and retain real-tool coverage. |
| `RAP-1405_add-run-script` — `bfb06c9f` (CWL) | Merged there as `6052b19d`; port under P3-PORT-04. |

GEC-669 prediction/calibration, GEC-670 and GEC-678 are merged; the older
`gec-669-fix-amplitude-state-and-phase-application` branch is superseded.
CWL's missing-strategy warning removal (`7511e4a1`) is not a remaining port:
GEC-671 deliberately warns with the resolved default. The local
`gec-664-speed-up-unit-tests` branch still points to `98a06302`, with no changes
at review time; it does not yet implement P3-TEST-01. Older observation-chunking
fixes are already in the migration baseline. Resource and Hebog branches are
tracked below.

**Coordinate overlaps:** preserve merged GEC-678 input protection when changing
beam/averaging builders; strategy, image-only and repeated-final changes share `Field.update()`;
plot ownership feeds artifact publication; process-output limits feed failure
reporting; safe restart must precede retries. Update options through defaults,
parsing, payloads, validators, commands, docs and tests together.

## Later improvements

- **B1 — Resource budgets before more concurrency.** Integrate focused commits
  `61157274`/`7d575dac` from `gec-618-improve-resource-allocation` and
  `1cc7f4d5` from `gec-618-improve-multinode-wsclean-configuration-logic`.
  Enforce node-wide CPU/memory budgets including subprocesses, per-tool threads,
  scheduling limits and MPI exclusivity. Verify one/multiple workers with real
  commands. Evaluate two NUMA-bound workers per node with disjoint allocations;
  the proposed external-Dask validation currently assumes one worker per host.
  Compare affinity, throughput, memory and products before changing defaults.
- **B2 — Cluster service.** Provide an optional managed Postgres-backed Prefect
  service with startup/recovery guidance; keep isolated standalone execution.
  Coordinate with P1-TEST-01 and bounded artifact publication. Retain Ubuntu
  24.04 while reported Ubuntu 26.04 memory problems are investigated.
- **B3 — Measured performance.** Fix or label unreliable `RUSAGE_CHILDREN`
  fallback metrics before profiling. Measure filtering, imaging and I/O; cache
  catalogs by query/coverage while preserving offline/fallback behavior. After
  B1, investigate chunk prediction/solve and residual/image overlap first,
  then independent cubes, diagnostics, facets, regridding, compression and plots.
  Bound memory, preserve single-writer ownership and wait for consumers before
  cleanup; compare products and resources on representative multi-band/BDA runs.
- **B4 — Coverage and maintenance.** Extend SKA-Low, screens/IDGCal and
  multi-sector coverage; retire legacy solve flags through a documented
  deprecation; simplify large owners and shared fixtures when justified.
- **B5 — Optional Hebog backend.** Choose one local implementation:
  `add-hebog-source-finder` (`5e5d9dcd`), its continuum variant
  `hebog-continuum-default` (`632f6211`), or the earlier
  `feat/hebog-native-task` (`d830c1c8`). The remote source-finder ref still
  points at its pre-feature base. Keep PyBDSF as default pending a scientific
  decision. Validate masks/catalogs, apparent/true-sky models, flux scale,
  convergence, dependencies, resources and restart before claiming improvement.

## Completion and evidence

Follow [AGENTS.md](AGENTS.md) and [TESTING.md](TESTING.md). Close an item only
when the implementation or compatibility decision is merged with its acceptance
evidence and relevant release/upgrade documentation. Never modify original
Measurement Sets; preserve serializable worker payloads and complete
output/finalizer/restart records.

This update fetched branch refs, compared the outstanding implementations and
tests, verified GEC-678's tree against `main`, and rechecked restart, retry and
phase-only selector code. Branch tests were inspected, not rerun; this review
does not establish CI, scientific-equivalence or cluster validation. Unchanged
findings retain the earlier source-review baseline.

Detailed review-ID mappings, source references and dated runtime evidence are
preserved in the previous plan at commit `b228d6b6`:

```bash
git show b228d6b6:PLAN.md
```
