# Post-Merge Development Plan

Updated: 2026-10-19. Release baseline: `origin/main` at `57f90ab7`;
CWL reference: `origin/master` / `origin/rapthor-cwl` at `b83cf9bc` (v2.2).

This is the actionable backlog for the first Prefect/Dask release and later
improvements. Task IDs are unchanged. **Merged** means present on `main`;
feature-branch implementations remain open until reviewed, validated and merged.

## Recommended next item

**P0-BUG-07: make the final strategy step independent of the last selfcal step.**
It affects the built-in strategy, has a small implementation and needs no
scientific-policy decision or cluster environment. In `set_selfcal_strategy()`
([strategy.py](rapthor/lib/strategy.py)), appending `strategy_steps[-1]` shares
the dictionary: setting the final channel width to 4 MHz also changes the last
selfcal step from 8 MHz to 4 MHz. An isolated probe of current `main` reproduced
this with both generated and supplied initial models.

- Deep-copy the last selfcal step before setting final-cycle parameters.
- Test independent dictionaries and nested `calibration_strategy` lists, with
  last-selfcal/final widths of 8/4 MHz for both initial-model paths.
- Preserve solve order, early phase-only cycles and final-pass selection when
  data fractions are equal. Run the focused strategy tests and update the
  change/upgrade notes.

**Next merge: GEC-678's existing input-MS protection.**
GEC-670 is now merged; rebase only GEC-678's five own commits
(`8cc1c52e..b07de520`) onto current `main`, run its focused and real-DP3
input-DATA/FLAG checks, and merge. The import-order fix is already included.

Then address P0-BUG-04, resolve P1-DEC-01, and complete restart safety before
implementing retries. Runtime/cluster work can proceed independently, subject
to the overlaps listed below.

## Release status

Four P0 fixes are merged; **five P0 items remain**. P1 items gate the features
named below. P2/P3 work can follow release if applicable limitations are recorded
in the [upgrade guide](docs/source/upgrading.rst).

### Merged

| Task | Result | Merge |
| --- | --- | --- |
| P0-BUG-01 | Prediction normalization acts on the predicted model. SageCal normalization remains unsupported; use `sagecalpredict = False`. | `83f056c7` (GEC-669) |
| P0-BUG-02 | Calibration receives DI pre-application and normalization on the predicting solve. WSClean model-column normalization remains unsupported by the tested DP3 6.6 path. | `d3f236dc` (GEC-669) |
| P0-BUG-03 | DI and DD amplitude-application state is tracked separately. | `57f90ab7` (GEC-670) |
| P0-BUG-06 | Single-direction imaging applies the combined phase table once. | `57f90ab7` (GEC-670) |

The GEC-670 feature tip `818f34cb` and merge `57f90ab7` have identical trees.
It does **not** fix phase-only DD product selection (P1-DEC-01) or image-only
solution/layout carry-forward (P2-BUG-01). Check release/upgrade notes for the
merged application changes before release.

### P0 — release blockers

| Task | Status / owner | Required result |
| --- | --- | --- |
| P0-BUG-07 | **Next implementation.** `lib/strategy.py`; small. | Independent final-step state; acceptance checks above. Also affects CWL. |
| P0-BUG-09 | **Implemented, unmerged:** GEC-678 at `b07de520`. `operations/calibrate/plan.py`. | End every DD solve chain with `null`; retain DDECal-only DI behavior. Validate pre-apply, BDA, WSClean and image-based paths, with input DATA/FLAG unchanged even on failure. Rebase the five own commits, not the old prerequisite stack. |
| P0-BUG-04 | Open. `Field.update()` / strategy resolution; medium. | Resolve omitted strategies afresh each cycle. Decide legacy versus explicit-default fallback; test first/later omissions, `None` and legacy flags. Log resolved solves and document the default. Distinguish new solves from image-only carry-forward. |
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
| P2-BUG-01 | Associate applied h5parms with their producing sky-model/facet layout; preserve that association across consecutive image-only cycles. Align prediction and imaging without weakening optimizer-seed rules. Test multiple sectors, outliers, peeling/reweighting and repeated finals. |
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
| P2-PORT-01 | Port full-beam prediction (`23ed80e9`): `wsclean_predict_beam_interval`, facet-beam flags and full-beam model images, with DP3 beam modes coordinated to prevent double application. Test both `use_array_beam` settings and compare DP3/WSClean multi-facet fluxes. |
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
| P2-TEST-02 | Reset flags and weights in the synthetic normalization MS before prediction (`da442dfc`). |
| P3-TEST-01 | Adapt Astron/SKA CI, duration-balanced shards and bounded CPU counts. Keep field/Prefect-server tests serial with isolated state. |
| P3-TEST-02 | Add real-run facet-RMS JSON assertions and the missing sky-model-download failure cases; runtime handling is already present. |
| P3-TEST-03 | Restore the task-count 1/2, full/single DDE and serial/MPI gridding matrix using command records. Adapt two-sector cases without CWL mocks. |
| P3-PORT-02 | Add the Docker dependency-label/hash inspection helper (`f75dfe53`) with a fake-Docker test. |
| P3-PORT-03 | Merge current-branch documentation links, correct `DATA` versus `data_colname` guidance and fix the imaging-parameter docstring markup. |
| P3-PORT-04 | Adapt CWL's installed run/log-archive wrapper (`6052b19d`) to Prefect logs and supported environments. Preserve exit status, exclude MS/large products, and test partial logs, optional inputs and unusual paths. Document Bash/GNU requirements. |
| P3-DEC-02 | Choose and validate the SageCal/libdirac dependency tuple; coordinate resolver, mirrors, labels and cache keys. CWL v2.2's dependency file is a comparison baseline, not Prefect validation. |
| P3-BUG-01 | Reproduce WSClean's shared-facet I/O failure on the supported stack; fix or constrain support, then run the currently disabled integration case. |

## Existing branches to integrate

These are sources for the tasks above, not additional completed work. Rebase
focused changes onto current `main`; avoid replaying pre-squash migration
history or temporary `rapthor3` CLI overlays.

| Branch / reviewed tip | Disposition and next action |
| --- | --- |
| `gec-678-fix-calibrate-input-ms-rewrite` — `b07de520` | P0-BUG-09; five own commits above `8cc1c52e`, including the import fix. Rebase, validate and merge. |
| `correct-calibration-beam-application` — `37215146` | Partial P2-PORT-01: fixes solve beam selection but retains image-based `applybeam.beammode=array_factor`; no WSClean facet-beam port yet. |
| `RAP-1392-skip-outlier-subtraction` — `21657525` | Partial P2-BUG-03: repeated-final state and reset enumeration. Add focused peeling/reset regressions and actual-directory discovery. |
| `RAP-1444-get-BDA-parms-no_cal` — `5102840d` | Supplied-h5parm interval inference related to P2-BUG-01. Reconcile with GEC-670 scanning; test single-sample/irregular time axes. Does not fix direction-layout ownership. |
| `fix-references-to-master-branch` — `8c451a08` | Partial P3-PORT-03; preparation guidance and docstring fixes remain. |
| `use-local-git-mirror` — `0e318a61` | Build/CI URL changes; verify mirror refs and reproducibility. Does not fix NumPy/APT gaps or pin `HEAD`. |
| `RAP1418-enable-averaging` — `5792ec62` (CWL) | P2-DEC-02/P3-PORT-01 candidate; no new tests. Adapt averaged-MS outputs, serializable records, scratch lifetime and calibration consumers. |
| `add-multi-sector-tests` — `b7537329` (CWL) | Source for P3-TEST-03/B4; adapt per-sector assertions to Prefect records and retain real-tool coverage. |
| `skip_warning_for_missing_calibration_strategy` — `ebedebe9` (CWL) | Merged there as `7511e4a1`; adapt under P0-BUG-04 after deciding fallback semantics. |
| `RAP-1405_add-run-script` — `bfb06c9f` (CWL) | Merged there as `6052b19d`; port under P3-PORT-04. |

GEC-669 prediction/calibration and GEC-670 are merged; the older
`gec-669-fix-amplitude-state-and-phase-application` branch is superseded.
Resource and Hebog branches are tracked below.

**Coordinate overlaps:** GEC-678 and beam/averaging work share calibration
builders; strategy, image-only and repeated-final changes share `Field.update()`;
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

This update fetched branch refs, verified the GEC-670 merge tree and GEC-678
commit range, and ran the isolated strategy probe described above. It did not
rerun DP3, WSClean, CI, scientific-equivalence or cluster tests. Unchanged
findings retain the earlier source-review baseline.

Detailed review-ID mappings, source references and dated runtime evidence are
preserved in the previous plan at commit `b228d6b6`:

```bash
git show b228d6b6:PLAN.md
```
