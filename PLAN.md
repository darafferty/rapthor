# Post-Merge Development Plan

Updated: 2026-11-03. Release baseline: `origin/main` at `ef12dafe`;
CWL reference: `origin/releases/v2.2` at `b83cf9bc` (v2.2) and
`origin/rapthor-cwl` at `24307de9` (v2.2 plus BDA averaging and the BDA applycal
restriction). The remote no longer has a `master` branch; "master" below means
that CWL line.

This is the actionable backlog for the first Prefect/Dask release and later
improvements. Task IDs are unchanged. **Merged** means present on `main`;
feature-branch implementations remain open until reviewed, validated and merged.

## Recommended next item

**Resolve the phase-only DD product decision (P1-DEC-01).** All P0 blockers and
the GEC-672 restart and reset work are merged, and this is the one remaining
item the plan requires before any release; the merged strategy fixes do not
change that selector. Then decide which P1 feature gates apply to the first
release. `gec-673-do-not-return-output` and `get-673-subprocess-run` conflict
and both change the command runner, so settle them before P1-BUG-01 (see
[existing branches](#existing-branches-to-integrate)).

## Release status

All nine P0 fixes are merged.
P1 items gate the features named below. P2/P3 work can follow release if
applicable limitations are recorded in the [upgrade guide](docs/source/upgrading.rst).

### Merged

| Task | Result | Merge |
| --- | --- | --- |
| P0-BUG-01 | Prediction normalization acts on the predicted model. SageCal normalization remains unsupported; use `sagecalpredict = False`. | `83f056c7` (GEC-669) |
| P0-BUG-02 | Calibration receives DI pre-application and normalization on the predicting solve. WSClean model-column normalization remains unsupported by the tested DP3 6.6 path. | `d3f236dc` (GEC-669) |
| P0-BUG-03 | DI and DD amplitude-application state is tracked separately. | `57f90ab7` (GEC-670) |
| P0-BUG-04 | Absent, `None` or empty calibration strategies resolve per calibrating cycle; omissions stay identifiable at load time. Legacy flags are translated first, including explicitly false flags; image-only omissions keep the previous solve list. Preflight, logging, tests and docs updated. | `812743c5` (GEC-671) |
| P0-BUG-06 | Single-direction imaging applies the combined phase table once. | `57f90ab7` (GEC-670) |
| P0-BUG-07 | The final step is a deep copy of the last selfcal step, including nested solve lists, so default final/last-selfcal widths stay 4/8 MHz and differing steps trigger a final pass even with equal data fractions. CWL remains unfixed. | `352468c7` (GEC-671) |
| P0-BUG-09 | DD solve chains end with `null`, protecting input DATA/FLAG; DDECal-only DI behavior is retained. Includes command and real-DP3 regression checks. | `98a06302` (GEC-678 implementation; merge title says GEC-669) |
| P0-BUG-08 | `prefect_retries` is removed from defaults, execution settings, flow wiring, docs, the demo parset and templates; a parset that still sets it gets the usual invalid-option warning. No retry behaviour is added. | `006264d2` (GEC-672) |
| P0-BUG-05 | Image preparation and concatenation (in imaging and the standalone operation) write each Measurement Set to `.<name>.partial` and rename it into place only after the command succeeds, so a product under its final name is complete and a resumed run reuses it, as it reuses a finished operation; leftover partial products are removed and made again. Settings are not compared: a reset makes the products again with other settings. Compressed images and mosaics are always made again; model images and their mosaics are compressed without loss. Every removal of a directory record first moves it to its partial name. Inputs are refused as outputs, and `.outputs.json` is written before `.done`. Follow-ups are under [WS-D](#restart-and-output-ownership-ws-d). | `2cf9b62a` (GEC-672). Its merge message still describes the earlier digest design; the code is the final-name design above. |

GEC-670 does **not** fix phase-only DD product selection (P1-DEC-01) or
image-only solution/layout carry-forward (P2-BUG-01); the GEC-671 fixes keep the
solve list, not that product/layout association.

Also merged since `98a06302`, with the tasks they complete or advance:

| Merge | Change | Tasks |
| --- | --- | --- |
| `8fc94bb8` | Solve-beam selection follows `use_array_beam`, including `False`. | Part of P2-PORT-01 |
| `35731c8e` | Outliers are peeled only in the first final cycle; repeats reuse the peeled data. Reset lists repeated final cycles from the current repeat count. | Part of P2-BUG-03 |
| `d91e2dfe` | Serial/parallel test split with one shared Prefect server per worker, and an eight-node integration lane balanced by `.test_durations`. | Most of P3-TEST-01 |
| `d649aa3d` | Integration MS DATA scaled by `1.5e4` (weights inversely by its square) so solved gains are realistic; original inputs untouched. | Test fixture; not P2-TEST-02 |
| `62907d14` | Astron/SKA CI split, Docker dependency-version helper and docs, v2.2 changelog. CWL's `run-rapthor.sh` was deliberately not copied. | P3-PORT-02 helper/docs |
| `27e36a20` | The test MS is downloaded after collection, only when a selected test uses it; one download is shared between xdist workers and a damaged archive gives a warning. | Test fixture |
| `ee00555b` | Command profiling finds `time` only through `PATH`, fixes the `resource` fallback and documents `prefect_command_profile`; that text in `parset.rst` still has a slip ("disk I/O of are also recorded"). | Was `gec-673-get-time-path`; supports B3 |
| `2fcb74db` | All-empty sector models give an all-zero model mosaic with valid records; WSClean draws model mosaics with at least two threads and fails clearly when memory would force one; Stokes IQUV cycles omit filtered-model products; both methods are documented. | P2-BUG-06 |
| `f5bc99d0` | Each calibration plot call writes to its own directory, cleared before a rerun, so concurrent tasks and reruns return all and only their own plots; each plot has one Prefect artifact key. Per-sector astrometry offsets, diagnostics and diagnostic plots move to `plots/image_N` and survive cleanup and restart. | P2-BUG-02, P2-BUG-05 |
| `a25fffc9` | Reset lists operations from the actual `pipelines/` directories (including cycles the current settings no longer run), resets a cycle's DI and DD calibration operations together, and copies the observations so unchunked input data is not peeled twice. It removes owned scratch workspaces, links and recovery records together, and moves the log records of reset operations to per-attempt files. A `flock` on the working directory keeps a reset and a run apart. The running guide says when a reset is needed rather than a plain rerun; `rapthor/modifystate.py` becomes `rapthor/reset.py`. | Rest of P2-BUG-03, P2-BUG-04 |
| `3102f60a` | Commands run directly instead of through a temporary `.sh` file, and the predict flow loses its own runner helper. | GEC-673; shares the runner with P1-BUG-01 |
| `6057f490` | Solution time steps are derived from supplied h5parms at startup, to set BDA parameters when no calibration is done; single-sample and screen solutions are tested. | Was `RAP-1444-get-BDA-parms-no_cal`; does not fix P2-BUG-01 |
| `6e90be48` | Tox forwards `PYTEST_XDIST_AUTO_NUM_WORKERS`, so CI runners short of memory can use fewer workers. | Part of P3-TEST-01 |
| `45c1c978`, `ef12dafe` | Slurm sizing reads `os.environ`. The bootstrap keeps the requested runner when it starts a local Dask scheduler; `local_dask_workers` defaults to 1, must be positive, no longer follows `max_nodes` and its `local_dask_worker_count` alias is removed. | Most of P1-BUG-02 |

`35731c8e` also changes repeated final cycles **without** peeling: they now
calibrate against the imaged sources only, where they previously kept the
outlier sources, which are still in the data. Selfcal cycles without peeling
already behave this way; confirm that this is intended.

### P0 — release blockers

None open; all nine are [merged](#merged).

### P1 — decisions and feature-specific release gates

| Task | Scope / owner | Required result |
| --- | --- | --- |
| P1-DEC-01 | Phase-only DD cycles; `Calibrate._dd_active_solution()`. | Decide whether to apply combined fast+medium phases or omit the unused medium solve. Test selected product contents, restart and downstream prediction/imaging; update products/science documentation. |
| P1-BUG-01 | Cancellation, especially clusters; `execution/shell.py`. | Terminate command descendants with a process-group TERM/KILL sequence. Bound captured output: `_run_captured_shell_command()` still keeps all output in memory, although no caller uses it. Test Prefect cancellation, worker shutdown and MPI wrappers. `gec-673-do-not-return-output` and `get-673-subprocess-run` (see [existing branches](#existing-branches-to-integrate)) change the same runner; settle them first. |
| P1-BUG-02 | Slurm; bootstrap/config/slurm owners. | **Code merged** (`45c1c978`, `ef12dafe`): the Slurm guard now sees the requested runner, the one-worker default is resolved and Slurm environment values are read. Test the CLI, precedence and missing-scheduler cases. |
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
| P2-BUG-01 | Associate applied h5parms with their producing sky-model/facet layout; preserve that association across consecutive image-only cycles. GEC-671 preserves the solve list and `6057f490` derives time steps from supplied h5parms; neither gives this product/layout association. Align prediction and imaging without weakening optimizer-seed rules. Test multiple sectors, outliers, peeling/reweighting and repeated finals. |
| P2-DEC-01 | Inventory CWL solve sequences rejected by Prefect. Port required combinations with solve/application/product tests, or document deliberate restrictions; parsing alone does not establish support. |
| P2-BUG-07 | A calibrating cycle whose sky model has no sources (logged as "Sky model contains no sources") stops with `ValueError: min() iterable argument is empty` in `Observation.set_calibration_parameters()` (`rapthor/lib/observation.py`); present since the migration (`c6196cb4`). Fail with a clear message, or handle the empty model deliberately; test an empty sky model. Seen with the unscaled `tests/resources/test.ms`. |

### Restart and output ownership (WS-D)

P0-BUG-05 and P2-BUG-02 to P2-BUG-06 are [merged](#merged). Their follow-ups and
one new bug remain.

| Task | Remaining work |
| --- | --- |
| P0-BUG-05 follow-ups | The residual-visibility MS, mask and region still treat "exists" as done; an interrupted finalizer move (`copy_record_object(move=True)`) can nest an MS. |
| P2-BUG-04 follow-up | Reset refuses where the file system has no `flock` support (for example Lustre mounted without `flock`), and `localflock` only protects within one node. |
| P2-BUG-06 follow-ups | Calibration's WSClean model drawing has the same one-thread exposure; report the WSClean 3.7 single-thread `-draw-model` bug upstream; drawn mosaics can sit up to one pixel off the regridded grid; `sparse_fits` halves components in sector overlaps. |
| P2-BUG-08 | The sector filtered-model image is written uncompressed under a `.fits.fz` name (`compress_image_if_needed()` in `rapthor/execution/image/restoration.py` writes a plain HDU), whatever the compression settings. Decide whether to compress it without loss (for example `CompImageHDU` with `quantize_level=0.0`) or name it `.fits`; check against CWL and downstream readers. |

### Logging and failures (WS-E)

| Task | Remaining work |
| --- | --- |
| P2-PORT-05 | Remove duplicate console messages, ANSI codes in files and import-time logging configuration. Preserve module loggers, Prefect handlers and durable worker diagnostics; test repeated setup and API shutdown. |
| P2-PORT-06 | Carry exit code, log path and bounded diagnostic tail through Prefect/Dask exceptions to the CLI. Coordinate with P1-BUG-01; test a failing external command. |

### WSClean prediction (WS-F1)

| Task | Remaining work |
| --- | --- |
| P2-PORT-01 | Solve beam selection is merged (`8fc94bb8`); image-based `applybeam.beammode=array_factor` remains. Port WSClean full-beam prediction (`23ed80e9`): `wsclean_predict_beam_interval`, facet-beam flags and full-beam model images, with DP3 beam modes coordinated to prevent double application. Test both `use_array_beam` settings and compare DP3/WSClean multi-facet fluxes. |
| P2-DEC-02 | Keep the WSClean/BDA guard until averaged data/model rows and channels, beam handling, input preservation and restart are validated. Otherwise document the limitation and resource cost. CWL now has BDA averaging (`c0509eae`) and skips the pre-applying applycal for WSClean prediction with BDA, because DP3 ApplyCal does not support BDA data (`24307de9`); use both as porting references. |
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
| P3-TEST-01 | Astron/SKA CI separation (`62907d14`), the serial/parallel split with shared Prefect servers and duration-balanced integration shards (`d91e2dfe`) are merged, and CI can now lower the worker count (`6e90be48`). Decide whether `pytest-8-threads`, which fixes the parallel lane at eight workers, is still needed. Record measured speed-ups and refresh `.test_durations` when integration tests change. |
| P3-TEST-02 | Add real-run facet-RMS JSON assertions and the missing sky-model-download failure cases; runtime handling is already present. |
| P3-TEST-03 | Restore the task-count 1/2, full/single DDE and serial/MPI gridding matrix using command records. Adapt two-sector cases without CWL mocks. |
| P3-PORT-02 | Helper and usage docs are merged (`62907d14`). Add a fake-Docker test covering labels, missing labels and inspect failure. |
| P3-PORT-03 | Merge current-branch documentation links, correct `DATA` versus `data_colname` guidance and fix the imaging-parameter docstring markup. |
| P3-PORT-04 | Adapt CWL's installed run/log-archive wrapper (`6052b19d`, `sbin/run-rapthor.sh`; not copied by `62907d14`) to Prefect logs and supported environments. Preserve exit status, exclude MS/large products, and test partial logs, optional inputs and unusual paths. Document Bash/GNU requirements. |
| P3-DEC-02 | Choose and validate the SageCal/libdirac dependency tuple; coordinate resolver, mirrors, labels and cache keys. CWL v2.2's dependency file is a comparison baseline, not Prefect validation. |
| P3-BUG-01 | Reproduce WSClean's shared-facet I/O failure on the supported stack; fix or constrain support, then run the currently disabled integration case. |

## Existing branches to integrate

These implementations are not yet merged into `main`. Rebase older branches
using focused changes; avoid replaying pre-squash migration history or temporary
`rapthor3` CLI overlays. CWL branches are porting sources, not implementations
in the Prefect version.

All GEC-672 branches are merged. `gec-672-combined-test` (`0e4da8e9`) was for
testing only, is out of date and can be deleted.

| Branch / reviewed tip | Disposition and next action |
| --- | --- |
| `gec-664-integration-coverage` — `2400779c` | Combines unit and integration coverage into one report. Its test-speed commits are merged as `d91e2dfe` and its pre-fetch commit is superseded by MR !536; rebase the two coverage commits onto the Astron/SKA CI split. It now conflicts with `main` in CI, tox and test files. |
| `pytest-8-threads` — `22e0f88f` | Part of P3-TEST-01, on `27e36a20` and still merging cleanly with `main`: the parallel tox lane uses eight xdist workers instead of `auto`. `6e90be48` now lets CI choose the count, so check whether this is still needed. |
| `gec-673-do-not-return-output` — `9f5de785` | Eight commits on `45c1c978`, merging cleanly with `main`: `run_external_command` returns nothing and the runner stops keeping output in memory, which is part of P1-BUG-01; tests read the output from the log instead. Conflicts with `get-673-subprocess-run` in `rapthor/execution/shell.py`. |
| `get-673-subprocess-run` — `2dfe6011` | One commit on `3102f60a` (its direct-execution work is merged), merging cleanly with `main`: `subprocess.run` replaces `Popen`. Output is captured whole and logged when the command ends, so `prefect_stream_output` no longer streams, memory use grows with the output, and a stopped command leaves no output or footer in its log. Reconcile with the bounded output and process-group termination of P1-BUG-01 and the failure tail of P2-PORT-06. |
| `gec-673-one-worker-per-node` — `7aa11c34` | Documents 0 as one Dask worker per node. Superseded by `ef12dafe`, which removes that behaviour; it conflicts with `main` and can be closed. |
| `fix-references-to-master-branch` — `1ff897db` | Part of P3-PORT-03, rebased onto `main`: two `master` references in the docs become `main`. The remote `master` branch is gone, so the `blob/master` link to `defaults.parset` in `parset.rst` no longer resolves; merge this soon. Preparation guidance and docstring fixes remain. |
| `use-local-git-mirror` — `3cd82217` | Build/CI URL changes in the Dockerfile, commit-hash script and CI base image; verify mirror refs and reproducibility. Does not fix NumPy/APT gaps or pin `HEAD`. |
| `rapthor-cwl` — `c0509eae`, `24307de9` (CWL) | Merged CWL versions of `RAP1418-enable-averaging` and `RAP1418-remove-applycal`: BDA averaging with a BDA on/off WSClean-prediction integration case, and no pre-applying applycal with BDA. P2-DEC-02/P3-PORT-01 sources; adapt averaged-MS outputs, serializable records, scratch lifetime and calibration consumers, and keep the Prefect guard until validated. |
| `add-multi-sector-tests` — `b7537329` (CWL) | Source for P3-TEST-03/B4; adapt per-sector assertions to Prefect records and retain real-tool coverage. |

Merged and no longer listed: both GEC-671 branches, `gec-672-remove-retries`,
`gec-672-restart-safety`, `gec-672-reset`, `gec-672-model-mosaic`,
`gec-672-plot-outputs`, `gec-673-direct-shell-command`, `gec-673-get-time-path`,
`RAP-1444-get-BDA-parms-no_cal`,
`gec-664-pre-fetch-test-data`, `gec-664-scale-test-ms-data`,
`gec-664-speed-up-unit-tests`, `correct-calibration-beam-application`,
`sync-with-master` and `RAP-1392-skip-outlier-subtraction` (see
[Merged](#merged)). The local `gec-664-fix-bootstrap-timeout` and
`gec-664-simplify-test-setup` branches are fully contained in `d91e2dfe`. The
older `gec-669-fix-amplitude-state-and-phase-application` branch is superseded.
CWL's missing-strategy warning removal (`7511e4a1`) is not a remaining port:
GEC-671 deliberately warns with the resolved default. Older observation-chunking
fixes are already in the migration baseline. Resource and Hebog branches are
tracked below.

**Coordinate overlaps:** preserve merged GEC-678 input protection when changing
beam/averaging builders; strategy, image-only and repeated-final changes share `Field.update()`;
plot ownership feeds artifact publication; process-output limits feed failure
reporting; any future task retries need restart-safe products first. Update
options through defaults, parsing, payloads, validators, commands, docs and tests together.

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

This update fetched branch refs and checked each listed branch against
`ef12dafe` by merging it in memory, and read the commits merged since
`f5bc99d0`. `gec-672-restart-safety` (`54c83946`) and `gec-672-reset`
(`7348172f`), merged together on `f5bc99d0`, passed the non-integration suite
(4 serial, 1,621 parallel), lint, the import-order check, the docs build and the
integration lane (45 passed, 1 Slurm test skipped, 4 expected failures that are
also marked on `main`). `2cf9b62a` has the same tree as `54c83946`, and
`a25fffc9` differs from that merge only in a full stop in the docs and tidied
tests. The tests were not rerun on `ef12dafe` for this update, and other branch
tests were inspected, not rerun. This review does not establish CI,
scientific-equivalence or cluster validation.
Unchanged findings retain the earlier source-review baseline.

Detailed review-ID mappings, source references and dated runtime evidence are
preserved in the previous plan at commit `b228d6b6`:

```bash
git show b228d6b6:PLAN.md
```
