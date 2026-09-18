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
- **Multi-sector mosaic:** keep smoke/stored-reference coverage available, but
  treat this as lower priority than common single-sector paths.
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
