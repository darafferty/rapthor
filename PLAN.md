# Rapthor Switch-Readiness Plan

Status snapshot: 2026-09-18. Manual testing is in progress and is the main
remaining switch blocker.

## Goal

Make the current Prefect/Dask branch the branch developers and users want to
run: scientifically trustworthy, faster or no worse than `master` on the tested
paths, easier to observe, easier to debug, and pleasant to develop. The
user-facing workflow stays:

```bash
rapthor input.parset
```

This branch should replace `master` only when the decision is evidence-driven:
science equivalence, performance equivalence, manual testing, documentation,
and known limitations must all be visible to reviewers.

## Switch Criteria

The branch is ready to recommend over `master` when all of these are true:

1. **Evidence package is complete and reviewer-friendly.**
   `EQUIVALENCE_REPORT.md` summarizes the latest science and performance gate
   results, links to compact archived reports, explains accepted differences,
   and lists caveats plainly.
2. **Representative manual tests pass.**
   Developers outside the refactor run the current branch with real parsets and
   record outcomes, adaptations needed, runtime experience, output sanity, and
   dashboard/log usability.
3. **Parset migration is documented.**
   `docs/source/migrating_from_cwl.rst` lets users adapt a `master` parset
   quickly: calibration strategy changes, runtime options, existing
   h5parm/image-only workflows, local versus external Dask, and Prefect
   dashboard setup.
4. **Runtime UX is low-friction.**
   `rapthor input.parset` works with no existing Prefect server or Dask
   cluster, and users can opt into persistent dashboards or external Dask with
   copy/paste commands. Production users can also run multiple independent
   Rapthor jobs without a shared Prefect server until a Postgres-backed Prefect
   service is available.
5. **Quality gates are green.**
   Non-integration tests, representative integration tests, science
   equivalence, and performance equivalence all pass or have documented,
   accepted caveats.
6. **Known limitations are explicit.**
   Multi-sector mosaic, screens/IDGCal, and any site-specific tool issues are
   documented as either accepted caveats or required follow-up. Slurm with
   external Dask and MPI WSClean must have at least one representative
   production/staging validation before recommending this branch for
   multi-node imaging.
7. **Deployment packaging is available.**
   Developers can test without Spack using the dev container or an existing
   Python/tool environment plus editable install. Production-like deployments
   use the `py-rapthor-prefect-dask` recipe in `../ska-sdp-spack/packages/`,
   leaving the legacy `py-rapthor` recipe in place until the switch decision is
   made.

## Decision Status

**Science: in progress** — automated gates pass, but final acceptance is part
of manual testing. For the covered LOFAR HBA self-calibration contract, the
August sync is verified at current commit `59be6d94` against exact `master`
commit `b307e769`; the default frequency-BDA and generated-initial-sky-model
paths are verified against `043c15d4` with three repetitions per branch. Two
classified differences remain, neither an unexplained current-branch
regression: an EveryBeam baseline shift in old-reference normalization, and an
intentional WSClean channel-coverage fix where `master` leaves two of eight
channels unpredicted by using inclusive endpoints with WSClean's end-exclusive
`-channel-range`. Science is accepted once testers running their own parsets on
real data confirm that the products hold up.

**Performance: accepted** for the current optimisation phase.

- Phase-only core gate: `303.160 s` current versus `429.557 s` `master`
  (`-29.425%`).
- DD phase plus DI full-Jones gate: `94.004 s` versus `151.183 s` (`-37.821%`).
- Frequency-BDA gate: 9/9 cross-branch pairs pass; `125.944 s` versus
  `298.744 s` in that environment.

**Manual testing: in progress.** This is the main remaining switch blocker.
Interactive testing on developer machines is a first-class path: many testers
run `rapthor` directly without Slurm, using local/no-server Prefect and local
Dask. Slurm, external Dask, and MPI WSClean are a separate
production-readiness track.

**Multi-sector mosaic: low priority for the switch decision.** Keep targeted
smoke/equivalence coverage, but it should not block switching unless it exposes
a broader single-sector, imaging, or product-contract regression.

## Remaining Work

- [ ] **Demonstrate multi-node dashboards locally.**
  Run Rapthor through Slurm on multiple nodes and view both dashboards in a
  local browser: Prefect for flow/task state and Dask for worker/task
  occupancy. The Slurm launcher should print or write copy/paste SSH tunnel
  commands for both.
- [ ] **Benchmark the WSClean multi-band and frequency-BDA scenarios** if
  performance claims will be made for them. The existing core performance gates
  remain applicable to the unchanged default paths and need only be refreshed
  at the final switch gate.
- [ ] **Update the decision evidence.**
  Summarize manual-test outcomes, the install method used by each tester,
  Spack/module smoke checks, and Slurm staging status in
  `EQUIVALENCE_REPORT.md` or a linked switch-readiness report.
- [ ] **Run final gates** after the final switch-readiness edits:

  ```bash
  python3 -m ruff check --fix --select I <touched-python-files>
  python3 -m ruff format <touched-python-files>
  python3 -m pytest -m "not integration" tests
  RAPTHOR_TEST_RUN_ROOT=/tmp/rapthor-integration-runs \
    python3 -m pytest -m integration -vv -ra --durations=0 \
    tests/integration tests/operations/integration
  ```

  Then rerun or refresh the saved-reference science gate if scientific products
  changed, branch repeatability/equivalence for the main decision scenarios,
  the current CI benchmark scenario set, and at least one real-user manual
  parset.

## Master Catch-Up Tasks

Audit snapshot: 2026-09-18. Branch point is `2e21be62` (GEC-428, 2026-05-11);
the branch is 605 ahead and 54 behind `master`. The last hand-sync rounds were
2026-08-14 and 2026-09-14 (`be9c552f`).

Nothing here is cherry-pickable: `git cherry` matches zero commits by patch-id
because this branch removed `rapthor/pipeline/` (CWL) and `rapthor/scripts/`
entirely. Every item below is a re-implementation against
`rapthor/execution/<owner>/`. Of the 54 commits `master` has and this branch
does not, 21 are already ported, 6 are superseded by the migration, and the
rest are listed here in priority order.

### Priority 1 — quick, low-risk, do first

- [ ] **Port the double-logging fix** (`fda65c8a`, GEC-610).
  `rapthor/_logging.py` still applies `fh.emit = add_coloring_to_emit_ansi(fh.emit)`
  and never calls `logging.root.handlers.clear()`. The file is otherwise close
  to `master`, so this is close to a straight port. Add a test asserting a
  single handler and no duplicated records; `master` has none.
- [ ] **Fix the `normalization_reference_frequencies` default.**
  The option is in `rapthor/settings/defaults.parset:374` and is read in
  `rapthor/lib/parset.py`, `rapthor/lib/field.py` and `rapthor/execution/`, but
  it is missing from `rapthor/settings/defaults.json`. `master` has it in both.
  Also work out why `tests/lib/test_parset_option_coverage.py` does not catch
  this, since the same test must catch task 5 below.
- [ ] **Adopt the SPDX license expression** (`583dc808`).
  `pyproject.toml` still uses `license = {file = "LICENSE"}` plus the deprecated
  classifier, and `setuptools>=64` instead of `>=77`.
- [ ] **Reconcile toolchain drift in `pyproject.toml`.**
  `requires-python = ">=3.9"` vs `master`'s `">=3.10"`; tox envlist `py39-313`
  vs `py310-314`; missing `pandas` test dependency and `EVERYBEAM_DATADIR`
  passenv; missing ruff `exclude = ["debug/**"]` although `debug/scan_ms` exists
  here. Also decide whether to keep the `lsmtool` pin at `3b27105b`, which is
  stale relative to GEC-115 and the astrometry work, or move back to `@master`.
  Internal inconsistency to settle at the same time: `Docker/Dockerfile` pins
  `numpy<2` while `ci/ubuntu_24_04-base` pins `numpy>=2,<3`.

### Priority 2 — CI and build

- [ ] **Port the pre-26.04 container fixes, but stay on Ubuntu 24.04**
  (`7eb0b03f`, `b27ba6e6`). `master` upgraded to 26.04 in `a39e517b` and then
  reverted to 24.04 by default in `d1fd7249` (RAP-1469, 2026-09-18) because of
  unresolved memory issues running Rapthor under 26.04, so do **not** port
  `a39e517b` as a default change; this branch is already on `ci/ubuntu_24_04-*`
  and therefore already matches `master`'s current default. Still unported from
  that window: `Docker/Dockerfile` pins `numpy<2` where `master` unpinned it.

  As of 2026-09-22, `Docker/fetch_commit_hashes.sh` resolves six source
  dependencies from upstream `HEAD`, including DP3 and SAGECal. EveryBeam is
  temporarily pinned to v0.8.5 (`882b7c0b`) and WSClean to `2d5c1ed8` because
  DP3 master requires EveryBeam `<0.9`, while newer WSClean requires 0.9.x.
  Both Dockerfiles use the same pins in their builder and runtime stages.
  Remove the two pins together once DP3 master supports EveryBeam 0.9; track
  the upstream update in [DP3 !1525](https://git.astron.nl/RD/DP3/-/merge_requests/1525).

  The ~25-line build-Boost-from-source workaround in `ci/ubuntu_24_04-base`
  cannot be deleted yet: it depended on 26.04 shipping Boost.NumPy built
  against NumPy 2. `master` retains `ci/ubuntu_26_04-*` as non-default build
  files for further investigation; decide whether to mirror them here or wait
  until the memory issue is understood.
- [ ] **Restore duration-based integration test splitting** (`3e352b73`,
  `c3fac822`). `tests/integration/.test_durations` is absent, tox lacks
  `--durations-path` and `--splitting-algorithm least_duration`, and
  `.gitlab-ci.yml` runs `parallel: 4` against `master`'s `parallel: 8`.

### Priority 3 — feature gaps

- [ ] **Port array-beam application in prediction** (`23ed80e9`, Rap1461).
  Nothing ported. Needs the `wsclean_predict_beam_interval` option in both
  defaults files, `-apply-facet-beam` and `-facet-beam-update` on the WSClean
  predict command, the `-fpb` model naming (`predict-…-model-fpb.fits`), and the
  `ddecal_solve` array-beam change. This branch currently hardcodes
  `beam_interval=120` for DP3 only, in `rapthor/execution/predict/commands.py:61`
  and `rapthor/execution/calibrate/commands.py:39`. Target:
  `rapthor/execution/calibrate/prediction.py`.
- [ ] **Port predict reuse-ordering** (`d587eec5`, Rap1418).
  Nothing ported: no `-parallel-reordering`, `-reuse-reordered` or
  `-save-reordered`, no full-band model-image combining, no
  `optimal_rendering_parameters()`. Same target file as the task above.

  Do these two together: both touch `prediction.py` and both invalidate
  `tests/execution/fixtures/command_reference.json`, which encodes the current
  command lines in roughly seven places and must be regenerated once.
- [ ] **Re-implement failure error extraction** (`aabe35f2`, GEC-603).
  `rapthor/lib/operation.py:249` still raises a bare
  `RuntimeError(f"Operation {self.name} failed due to an error")` with no
  extracted cause. `master`'s `extract_log_errors` / `handle_failure` /
  `format_cli_command` parse CWL logs, so this needs real design work against
  Prefect task logs rather than a port. A Prefect-flavoured equivalent of
  `tests/resources/failed_workflow_sample.log` is needed as a fixture.

### Priority 4 — test coverage holes for already-ported features

These cover work that is already on the branch but untested here.

- [ ] **Add `tests/lib/test_calibration.py`.**
  `rapthor/lib/calibration.py` was ported in `4fd75842` but nothing under
  `tests/` imports it. Its sibling `test_calibration_memory.py` exists.
- [ ] **Port the per-facet RMS diagnostics test** (GEC-441):
  `tests/integration/test_sector_diagnostics.py`, plus the
  `test_image_from_reg.fits` and `test_image_regions_rendered.fits` resources.
- [ ] **Port the parallel-gridding integration test** (GEC-487):
  `tests/integration/test_wsclean_parallel_gridding.py`.
- [ ] **Rewrite the GEC-444 calibration-strategy tests.**
  `master`'s `tests/integration/test_image_only_applycal.py` and
  `test_legacy_calibration_strategy.py` have no equivalent here, although the
  logic is present in `rapthor/operations/image/base.py:137-164`. `master`'s
  `image_only` naming does not transfer, so these need rewriting against this
  branch's refactor rather than copying. Fixtures needed:
  `integration_field_solutions.h5`, `manual_testing.parset`,
  `manual_testing_strategy.py`.

### Decision needed before the next `master` merge

- [ ] **Decide on `rapthor/testing.py`** (GEC-486: `5f59fe87`, `0c422261`,
  `4cc0b732`, `14377a61`, `6c58212b`).
  `master` moved shared test helpers into an importable `rapthor/testing.py`;
  this branch never adopted it and grew `tests/conftest.py` to 738 lines against
  `master`'s 490. `assert_logged`, `make_source_catalog` and
  `generate_parset_from_template` are undefined anywhere in `tests/` here; only
  `generate_parset` exists, at `tests/conftest.py:420`. Every later `master`
  test commit builds on the module, so settling this before the next merge is
  cheaper than settling it during one.

### Superseded by the migration — do not port

- `e8873f19` (GEC-571, flat-noise symlink) and `b457f49c` (duplicate CWL
  fields) are CWL-only. `rapthor/execution/image/skymodel_filter.py:66` passes
  explicit output paths, so the PyBDSF `export_image` problem cannot occur.
- `18cf2d72` (image input generation) fixes a bug that is structurally absent:
  `rapthor/operations/image/base.py:583` already builds `parallel_gridding_tasks`
  as a genuine per-sector list.
- `c853f707` (GEC-513) and `37f6fc06` (Ubuntu 24.04) are superseded by the
  26.04 upgrade above.
- `0ebc0690` (GEC-444) originated on this branch and landed on `master` later;
  this branch is further along, having replaced `do_slowgain_solve` and
  `do_fulljones_solve` with `calibration_strategy` dicts and added
  `supported_combinations` validation. Port only the tests. Expect conflicts in
  `rapthor/lib/strategy.py` and the strategy docs.
- The GEC-349 formatting commits (`f826c262`, `13e3bd8e`, `6c74e20d`,
  `f379cf15`) are satisfied in outcome, since this branch formats all
  discovered files rather than `master`'s explicit `format_targets` list. The
  tox `[tool.tox.env.format]` sections have diverged textually and will conflict
  on merge.

## Completed

- **Execution architecture.** Owner-package execution for image, calibrate,
  concatenate, predict, mosaic, and pipeline flows. Operation adapters are
  thin; command builders, payload validation, output discovery, migrated helper
  logic, and flow wiring live under `rapthor/execution/<owner>/`.
- **Runtime bootstrap.** No-server local runs, explicit Prefect API runs, local
  Dask, external Dask, and run tags. No-server runs blank `PREFECT_API_URL`,
  disable Prefect analytics, and use an isolated temporary Prefect home per
  process — the right interim production mode until a shared Prefect server has
  a Postgres backend. Resource validation understands MPI command requests and
  checks that MPI WSClean is exclusive and stays within the configured Slurm
  node allocation.
- **Calibration semantics.** Solve order is strategy-driven through
  `calibration_strategy`, replacing legacy implicit solve slots with explicit
  types and order. DI scalar phase, DI diagonal slow-gain, and DI full-Jones
  products are pre-applied for image-only workflows; DD products are applied on
  the fly when directions match.
- **Observability.** Readable flow/task names, tool tags, task timing JSONL,
  command timing artifacts, persistent postage-stamp previews, and durable
  per-command logs at `dir_working/logs/<operation>/<task-run-name>.log` linked
  from `commands.jsonl`. `prefect_stream_output` controls forwarding to Prefect
  only; `prefect_log_commands` controls the durable logs, so no-dashboard and
  failed runs keep their output.
- **Master syncs (July and August).** Ported: calibration-aware imaging/BDA
  frequency limits, the production imaging frequency-BDA default, the official
  `https://iers.astron.nl/WSRT_Measures.ztar` measures URL, robust RMS
  diagnostics for facets outside an image, advisory/strict DP3 calibration
  memory checks, and removal of DP3's unsupported `writefullresflag`. Legacy
  solve toggles and retired CWL mechanics were deliberately not reintroduced;
  the `e8873f19` flat-noise symlink was CWL-specific and the migrated helper
  supplies explicit RMS output paths with a regression test. Focused suites
  pass (354 ownership-boundary, 29 field/facet, and 71 equivalence-harness
  tests), as does the strict OOM preflight case, which confirms that no
  calibration command starts.
- **Generated-initial-sky-model equivalence.** Paired
  `initial-skymodel-regroup` and `initial-skymodel-bda-regroup` scenarios
  compare source identities, patch membership, positions, fluxes, spectral
  terms, and shapes rather than counts alone. Both three-repetition gates pass:
  9/9 cross-branch pairs are repeatability-bounded in the no-imaging-BDA
  control; the production-BDA case has two strict passes and 7/9
  repeatability-bounded pairs.
- **Frequency-only imaging BDA.** Resolved after DP3 preparation while
  preserving master's intended BDA and primary-beam semantics: pass WSClean's
  required `-reorder`, retain the calibration-derived `image_bda_minchannels`
  safeguard, and require EveryBeam 0.8.3 or later, since earlier releases build
  a single-band telescope model and reject DP3's multi-SPW layout. Validated on
  2026-07-16 with a two-SPW imaging MS (`NUM_CHAN = [4, 8]`), facet-beam
  application, and a fully finite primary-beam FITS product. The
  branch-vs-master row stays skipped because `master` fails this path, which is
  a documented reference-branch bug rather than an equivalence reference.
- **Testing and deployment paths.** The interactive-testing guide, the
  `py-rapthor-prefect-dask` and `py-prefect-dask` Spack recipes with
  module-load smoke checks, the first interactive tester wave, the two-job
  no-server concurrency check (isolated Prefect state, no collisions on local
  SQLite state or output paths), and one representative Slurm allocation with
  external Dask and `imaging.use_mpi = True` without oversubscription.

## Current Caveats

- Screens/IDGCal remain target-environment dependent.
- Ubuntu 26.04 is not a supported container runtime. `master` reverted to
  Ubuntu 24.04 by default after unresolved memory issues under 26.04
  (RAP-1469), so keep this branch on 24.04 until that is understood.
- MPI WSClean and Slurm/external-Dask are production readiness checks. They are
  not local science-gate blockers, but they must pass in a representative
  cluster allocation before recommending this branch for multi-node production
  imaging.
- Do not run many production jobs against a shared local Prefect server backed
  by SQLite. Use no-server/ephemeral mode per job, or a properly managed
  Prefect service with a Postgres backend when persistent history is required.
- Multi-sector mosaic is low-priority and should not block the switch unless a
  regression also affects common single-sector paths.
- Historical `master` behavior around some slow-gain/full-Jones combinations is
  not always a desirable scientific target; accepted differences are recorded
  in the equivalence reports.
- Raw run products are intentionally not tracked in git. Keep compact reports
  under `docs/source/development/` and raw products under ignored run roots or
  CI artifacts.

## Deferred Improvement Backlog

These are intentionally not switch blockers unless manual testing exposes them
as everyday-user problems. They are kept here so they are not lost while the
main plan stays focused on the branch-switch decision.

- **Image-side performance:** target `filter_skymodel` first, then WSClean
  image resource/concurrency policy. Calibration plotting is worth optimizing
  only if larger real runs keep showing it as a meaningful post-processing
  cost.
- **WSClean prediction parallelism:** investigate splitting the internal
  frequency/facet loop inside WSClean prediction tasks only with a targeted
  benchmark and explicit resource limits.
- **Single-machine task concurrency:** `local_dask_workers` falls back to
  `max_nodes`, which `rapthor/lib/parset.py:431` sets to 1 for
  `batch_system = single_machine`, so a default single-machine run gets one
  single-threaded Dask worker and executes the per-sector futures in the image
  and calibrate flows one at a time. Derive the default from available cores
  and memory, bounded by `cpus_per_task` and `mem_per_node_gb`, so the
  parallelism the flows already express is actually used.
- **Per-task resource gating:** `ResourceRequest` in
  `rapthor/execution/resources.py` is validated but never converted into Dask
  scheduling constraints; there are no worker-resource annotations anywhere in
  the flows. Without them, raising the worker count lets several WSClean or DP3
  tasks each claim `cpus_per_task` threads at once and oversubscribe the node.
  Annotate heavy tasks with CPU/memory resources so the scheduler serialises
  them while light Python tasks keep running. This is a prerequisite for
  raising single-machine concurrency.
- **Uniform thread capping for external commands:** environment policies and
  `thread_environment()` now live in `rapthor/execution/environments.py`.
  WSClean imaging explicitly selects its local or MPI thread policy. DP3
  solves, predicts, applycals, and the `python -m` adapters still inherit ambient
  `OMP_NUM_THREADS`/`OPENBLAS_NUM_THREADS`, so each concurrent task can spawn
  one thread per core. Extend the explicit policy helpers when implementing
  thread capping, using each task's resource budget and preserving tool-specific
  limits such as WSClean's single OpenBLAS thread per MPI rank.
- **Honour `local_scratch_dir` for I/O-heavy temporaries:** the option is
  parsed and passed into the pipeline capabilities dict but never consumed, and
  `rapthor/execution/image/wsclean.py:64` always places WSClean's `-temp-dir`
  under `dir_working`. On multi-node runs that puts reordering and gridding
  temporaries on the shared filesystem. Route WSClean temp dirs and other
  I/O-heavy intermediates to node-local scratch when the option is set.
- **Cache reference catalogues per run:** `_download_survey_data` in
  `rapthor/execution/image/flux_normalization.py:622` repeats the 5-degree VO
  query for every normalization call, once per sector per cycle, and
  `_get_data_from_skymodel` round-trips each result through a temporary FITS
  file. The phase centre is fixed for a run, so a run-scoped cache removes the
  repeated network round-trips and makes runs resilient to VO outages.
- **Parallelise the per-facet astrometry check:** `check_astrometry` in
  `rapthor/execution/image/diagnostic_calculation.py:777` loops over facets
  serially, copies the full PyBDSF sky model for each one, and, when no
  comparison sky model is supplied, fetches a Pan-STARRS cone per facet. With
  many facets that is tens of sequential network queries per imaging cycle.
  Fetch once per field where the 0.5-degree cone limit allows, or run the
  per-facet comparisons concurrently.
- **Multi-sector mosaic:** keep smoke/stored-reference coverage available, but
  treat this as lower priority than common single-sector paths.
- **Remove the legacy solve-flag translation:** `do_slowgain_solve` and
  `do_fulljones_solve` are currently translated into an explicit
  `calibration_strategy` with a deprecation warning
  (`rapthor/lib/strategy.py`), which keeps legacy strategy files runnable and
  identical on both branches during the migration. After the switch, turn the
  translation into an error and convert the equivalence inputs under
  `tests/resources/equivalence/inputs/base/`, which still use the flags. That
  also retires `legacy_flag_calibration_strategy`, whose trailing
  `medium_phase` reproduces a subtle CWL-side expansion rule.
- **Persistent Prefect service:** set up a shared Prefect server backed by
  Postgres so production users can monitor multiple parallel Rapthor jobs from
  one Prefect UI without relying on local SQLite state.
- **Deferred code tidying:** split or simplify modules such as
  `rapthor.execution.image.diagnostic_calculation`,
  `rapthor.execution.image.flux_normalization`,
  `rapthor.execution.calibrate.h5parm_combination`,
  `rapthor.operations.calibrate.base`, and `rapthor.operations.image.base` only
  when changing behavior or when profiling/maintenance pressure justifies the
  edit.
- **Testing suite polish:** keep architecture and regression guards focused on
  payload serializability, thin operation adapters, task-boundary visibility,
  calibration strategy semantics, image-only apply behavior, and branch
  equivalence reporting.

## Benchmark Scenario Rule

- keep the default automatic `ci-benchmark`
- use `ci-benchmark-image-products` when changing image products,
  `filter_skymodel`, WSClean image behavior, or image post-processing
- use `ci-benchmark-predict-chunks` only for prediction scheduling changes
- use `ci-benchmark-wsclean-predict` only for calibration prediction setup or
  WSClean-predict paths
- leave many-sector mosaic benchmarks out of automatic CI unless changing that
  path

Do not start speculative optimisation until manual testers can run the branch.

## Evidence Locations

- Stakeholder summary: `EQUIVALENCE_REPORT.md`
- Science contract: `docs/source/development/science_equivalence_contract.rst`
- Performance contract:
  `docs/source/development/performance_equivalence_contract.rst`
- Archived science, performance, and benchmark reports (the
  `science_equivalence_runs/`, `performance_equivalence_runs/`, and
  `benchmark_baselines/` trees under `docs/source/development/`) were removed
  in commit `fa4259a8` (2026-08-06) and remain retrievable from git history,
  for example with `git show fa4259a8^:<path>`.
- Run products are local-only: `runs/` is gitignored, so compact reports such
  as `runs/equivalence-gate-20260820-august-sync/` exist only on the machine
  that produced them. Rerunnable inputs are versioned under
  `tests/resources/equivalence/`.

## Development Rules Going Forward

- Prefer user/developer joy over cleverness: clear names, explicit errors,
  easy reports, and copy/paste commands.
- Keep payloads serializable and task boundaries benchmarkable.
- Keep operation adapters thin.
- Do not add compatibility shims for unreleased behavior unless they reduce
  manual-testing friction.
- Keep memory efficiency explicit in FITS/MS/image-heavy paths.
- Do not split tiny helpers into Prefect tasks; split large work units when it
  improves observability, failure isolation, or measured scalability.
