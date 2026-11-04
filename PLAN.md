# Post-Merge Development Plan

Updated: 2026-11-03. Baseline: `origin/main` at `ef12dafe`. CWL reference:
`origin/releases/v2.2` at `b83cf9bc` and `origin/rapthor-cwl` at `24307de9`.

This lists outstanding work only. All P0 release blockers are merged. P1 items
gate the features they name; P2 and P3 work can follow the release if its
limitations are recorded in the [upgrade guide](docs/source/upgrading.rst).
An item is closed when it is merged with its acceptance evidence and any release
or upgrade documentation. Completed work and earlier evidence are in the plan at
`0b14bb52` and, with review-ID mappings, at `b228d6b6`.

## Next steps

1. Finish P1-DEC-01, the one item needed before any release.
2. Merge `fix-references-to-master-branch`: the remote `master` branch is gone,
   so a link in `parset.rst` no longer resolves.
3. Settle the two conflicting command-runner branches, then do P1-BUG-01.
4. Decide which of the other P1 feature gates apply to the first release.

## Differences from the CWL branch

### Bugs introduced during the Prefect migration

These are not on `rapthor-cwl`.

| Task | Outstanding |
| --- | --- |
| P1-BUG-01 | Cancellation, especially on clusters (`execution/shell.py`). Terminate command descendants with a process-group TERM/KILL sequence. Bound captured output: `_run_captured_shell_command()` keeps it all in memory. Test Prefect cancellation, worker shutdown and MPI wrappers. |
| P1-BUG-02 | Slurm guard. Add a test through the CLI; the code is merged, and precedence and the missing scheduler are covered below the CLI. |
| P1-BUG-03 | Long runs with the dashboard. Deduplicate artifact publication, bound bytes, add a plot-publication switch and make monitoring errors warning-only. Test repeated operations and measure database growth. |
| P1-BUG-04 | Multi-chunk prediction (`execution/predict/flow.py`). Return exact chunk-owned paths instead of broad globs. Test two chunks with DI addition, DD subtraction, peeling and restart. |
| P0-BUG-05 follow-ups | The residual-visibility MS, mask and region still treat "exists" as done. An interrupted finalizer move (`copy_record_object(move=True)`) can nest an MS. |
| P2-BUG-01 | Image-only cycles regroup the sky model, so applied h5parms can lose the layout that produced them; CWL's `Field.update()` reuses that layout (`reuse_solution_layout`). Keep the association across consecutive image-only cycles and align prediction and imaging without weakening optimizer-seed rules. Test multiple sectors, outliers, peeling, reweighting and repeated finals. |
| P2-BUG-04 follow-up | Reset refuses where the file system has no `flock` support (for example Lustre without `flock`), and `localflock` only protects within one node. |
| P2-BUG-06 follow-ups | Drawn model mosaics can sit up to one pixel off the regridded grid. `sparse_fits` halves components in sector overlaps. |
| P2-BUG-07 | A calibrating cycle with an empty sky model logs that it is continuing, then stops with `ValueError: min() iterable argument is empty` in `Observation.set_calibration_parameters()`. Fail with a clear message or handle it deliberately, and test it. CWL has the same `min()` call but not the code path that continues. |
| To confirm | Since `35731c8e`, repeated final cycles without peeling calibrate against the imaged sources only, although the outlier sources are still in the data. CWL keeps them; selfcal cycles already omit them. Confirm that this is intended. |

### Features and fixes on the CWL branch, missing from `main`

The commits named are on `rapthor-cwl`. Adapt them rather than copying them: CWL
branches are porting sources, not Prefect implementations.

| Task | Outstanding |
| --- | --- |
| P1-PORT-01 | Standalone container (`7eb0b03f`). `Docker/Dockerfile` still installs `numpy<2`. Align the NumPy build and runtime versions and verify NumPy, PyBDSF and casacore imports plus source finding in final images. |
| P2-PORT-01 | Full beam in WSClean prediction (`23ed80e9`). Port image-based `applybeam.beammode=array_factor`, `wsclean_predict_beam_interval`, facet-beam flags and full-beam model images, with DP3 beam modes coordinated to prevent double application. Test both `use_array_beam` settings and compare DP3 and WSClean multi-facet fluxes. |
| P2-PORT-02 | Supplied-survey metadata (`dfa7f11a`, `6d4df857`). Restore flux and error metadata and the missing JSON reference-frequency default. Supplied and downloaded versions of a survey must normalize consistently; cover unknown frequencies and offline inputs. |
| P2-PORT-03 | Mixed station sets (`4c00305d`). Use a deterministic union and per-MS imaging inputs instead of unconditional time concatenation. Disable or validate residual-MS creation; cover serial and MPI runs, normalization and restart. |
| P2-PORT-04 | Offline catalog preparation (`dfa7f11a`, `fetch_skymodel.py`). Provide a maintained survey-download command. Document pointing, radius, catalog paths and frequencies; test failures and a normalization and diagnostics run with internet disabled. |
| P2-PORT-05 | Logging (`fda65c8a`). Remove duplicate console messages, ANSI codes in files and the import-time logging configuration in `execution/image/restoration.py`. Preserve module loggers, Prefect handlers and worker diagnostics; test repeated setup and API shutdown. |
| P2-PORT-06 | Failure causes (`aabe35f2`). Carry the exit code, log path and a bounded diagnostic tail through Prefect and Dask exceptions to the CLI. Coordinate with P1-BUG-01; test a failing external command. |
| P2-PORT-07 | Ubuntu 24.04 APT sources (`a39e517b`, `d1fd7249`). Use `ubuntu.sources` in build and runtime stages and CI. This does not adopt Ubuntu 26.04. |
| P2-PORT-08 | Package metadata (`a39e517b`, `583dc808`). Reconcile Python versions, license and build requirements and tox access to compiled astronomy dependencies; verify the supported Prefect environment. |
| P2-DEC-01 | Solve combinations (`0ebc0690`). Inventory CWL solve sequences that Prefect rejects. Port required ones with solve, application and product tests, or document the restriction. |
| P2-DEC-02 | BDA with WSClean prediction (`c0509eae`, `24307de9`). Keep the guard until averaged rows and channels, beam handling, input preservation and restart are validated; otherwise document the limitation and resource cost. |
| P2-TEST-02 | Normalization test MS (`da442dfc`). Reset flags and weights in the synthetic MS before prediction. |
| P3-PORT-01 | WSClean rendering and reordered-data reuse (`d587eec5`). Keep scratch per chunk until all consumers finish. Test single- and multi-band and single-channel coverage, visibility agreement, resources and end-exclusive channel ranges. |
| P3-PORT-02 | Dependency-version helper (`f75dfe53`). The helper is merged; add a fake-Docker test covering labels, missing labels and inspect failure. |
| P3-PORT-03 | Documentation (`ff1f29d9`). Correct `DATA` versus `data_colname` guidance and fix the imaging-parameter docstring markup. Also fix "disk I/O of are also recorded" in `parset.rst`, which is only on `main`. |
| P3-PORT-04 | Run and log-archive wrapper (`6052b19d`, `sbin/run-rapthor.sh`). Adapt it to Prefect logs and supported environments. Preserve exit status, exclude MS and large products, test partial logs, optional inputs and unusual paths, and document Bash/GNU requirements. |
| P3-DEC-01 | Named DS9 facets (`d90786e8`). Validate them with supported tools; add separate point and polygon formats only where needed. Match facets to model columns and h5parm directions. |
| P3-DEC-02 | SageCal/libdirac pin. CWL pins a commit; `main` resolves `HEAD`. Choose and validate the dependency tuple; coordinate resolver, mirrors, labels and cache keys. |
| P3-TEST-02 | Facet-RMS diagnostics (`eb1b6f2f`). Add real-run JSON assertions and the missing sky-model-download failure cases. |
| P3-TEST-03 | Parallel gridding (`38abdb92`). Restore the task-count 1/2, full/single DDE and serial/MPI matrix using command records. Adapt two-sector cases without CWL mocks. |

## Bugs on both branches

| Task | Outstanding |
| --- | --- |
| P1-DEC-01 | Phase-only DD cycles apply only the fast phases and discard the medium solve. `gec-697-apply-combined-dd-phases` applies the combined phases on `main`. Compare imaging on a phase-only cycle to confirm this over omitting the medium solve, test the selected product on restart, then review and merge. |
| P1-PORT-01, `np.NaN` | Sector subtraction uses `np.NaN`, which NumPy 2 removed. Replace it with `np.nan`. |
| P2-BUG-06 follow-ups | Calibration's WSClean model drawing can run with one thread, which WSClean 3.7's `-draw-model` cannot do reliably. Handle it as model mosaics now do, and report the WSClean bug upstream. |
| P2-BUG-08 | The sector filtered-model image is written uncompressed under a `.fits.fz` name (`compress_image_if_needed()`). Compress it without loss or name it `.fits`; check downstream readers. |
| P3-BUG-01 | WSClean's shared-facet I/O fails, and the integration case is disabled on both branches. Reproduce it on the supported stack, fix or constrain support, then enable the test. |

## Other test work

| Task | Outstanding |
| --- | --- |
| P2-TEST-01 | Mark real-WSClean restoration tests as integration and verify that non-integration selection excludes them. |
| P3-TEST-01 | Decide whether `pytest-8-threads` is still needed. Record measured speed-ups and refresh `.test_durations` when integration tests change. |

## Open branches

Rebase older branches using focused changes; avoid replaying pre-squash migration
history.

| Branch and tip | Next action |
| --- | --- |
| `gec-697-apply-combined-dd-phases` — `968a9ce3` | P1-DEC-01: see above. |
| `fix-references-to-master-branch` — `1ff897db` | Fixes the `master` references in the docs. Merge soon. |
| `gec-673-do-not-return-output` — `9f5de785` | Part of P1-BUG-01: `run_external_command` returns nothing and the runner stops keeping output in memory. Conflicts with `get-673-subprocess-run` in `shell.py`. |
| `get-673-subprocess-run` — `2dfe6011` | `subprocess.run` replaces `Popen`, so output is captured whole and logged at the end: `prefect_stream_output` no longer streams, memory grows with output and a stopped command leaves no output in its log. Reconcile with P1-BUG-01 and P2-PORT-06. |
| `pytest-8-threads` — `22e0f88f` | P3-TEST-01: fixes the parallel lane at eight workers. `6e90be48` lets CI choose the count, so check whether it is still needed. |
| `gec-664-integration-coverage` — `2400779c` | Combines unit and integration coverage into one report. Rebase its two coverage commits onto the CI split; it conflicts with `main` in CI, tox and test files. |
| `use-local-git-mirror` — `3cd82217` | Build and CI URL changes. Verify mirror refs and reproducibility; does not fix the NumPy or APT gaps. |
| `rapthor-cwl` — `c0509eae`, `24307de9` (CWL) | Source for P2-DEC-02 and P3-PORT-01. Adapt averaged-MS outputs, serializable records, scratch lifetime and calibration consumers. |
| `add-multi-sector-tests` — `b7537329` (CWL) | Source for P3-TEST-03 and B4. Adapt per-sector assertions to Prefect records and keep real-tool coverage. |
| `gec-673-one-worker-per-node` — `7aa11c34` | Superseded by `ef12dafe`; close it. |
| `gec-672-combined-test` — `0e4da8e9` | Out of date; delete it. |

## Later improvements

- **B1: resource budgets before more concurrency.** Integrate `61157274` and
  `7d575dac` from `gec-618-improve-resource-allocation` and `1cc7f4d5` from
  `gec-618-improve-multinode-wsclean-configuration-logic`. Enforce node-wide CPU
  and memory budgets including subprocesses, per-tool threads, scheduling limits
  and MPI exclusivity. Verify one and multiple workers with real commands.
  Evaluate two NUMA-bound workers per node, comparing affinity, throughput,
  memory and products before changing defaults.
- **B2: cluster service.** Provide an optional managed Postgres-backed Prefect
  service with startup and recovery guidance; keep isolated standalone execution.
  Stay on Ubuntu 24.04 while reported Ubuntu 26.04 memory problems are investigated.
- **B3: measured performance.** Fix or label unreliable `RUSAGE_CHILDREN` fallback
  metrics, then measure filtering, imaging and I/O. Cache catalogs by query and
  coverage, keeping offline behaviour. After B1, investigate chunk prediction and
  solve and residual/image overlap first, then cubes, diagnostics, facets,
  regridding, compression and plots. Compare products and resources on
  representative multi-band and BDA runs.
- **B4: coverage and maintenance.** Extend SKA-Low, screens/IDGCal and multi-sector
  coverage; retire legacy solve flags through a documented deprecation; simplify
  large owners and shared fixtures when justified.
- **B5: optional Hebog backend.** Choose one of `add-hebog-source-finder`
  (`5e5d9dcd`), `hebog-continuum-default` (`632f6211`) or `feat/hebog-native-task`
  (`d830c1c8`). Keep PyBDSF as the default pending a scientific decision. Validate
  masks, catalogs, apparent- and true-sky models, flux scale, convergence,
  dependencies, resources and restart.

## Overlaps to coordinate

Preserve the input protection on DD solve chains when changing beam or averaging
builders. Strategy, image-only and repeated-final changes share `Field.update()`.
Plot ownership feeds artifact publication, and process-output limits feed failure
reporting. Any future task retries need restart-safe products first.
