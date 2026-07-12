# Rapthor Switch-Readiness Plan

Status snapshot: 2026-07-12.

## Goal

Make the current Prefect/Dask branch the branch developers and users want to
run: scientifically trustworthy, faster or no worse than `master` on the
tested paths, easier to observe, easier to debug, and pleasant to develop.

The user-facing workflow should remain:

```bash
rapthor input.parset
```

This branch should replace `master` only when the decision is evidence-driven:
science equivalence, performance equivalence, manual testing, documentation,
and known limitations must all be visible to reviewers.

## Current Decision Status

**Science:** accepted for the covered LOFAR HBA self-calibration contract.

- The latest stakeholder-facing summary is `EQUIVALENCE_REPORT.md`.
- Detailed science evidence lives under
  `docs/source/development/science_equivalence_runs/`.
- Repeatability-aware branch comparisons now generate
  `science-equivalence-report.*`, `performance-equivalence-report.*`, and
  `repeatability-summary.*` from the same branch executions.

**Performance:** accepted for the current optimisation phase.

- Phase-only core gate passes: current median runtime was `303.160 s` versus
  `429.557 s` for `master` (`-29.425%`).
- DD phase plus DI full-Jones gate passes: current median runtime was
  `94.004 s` versus `151.183 s` for `master` (`-37.821%`).
- Detailed performance evidence lives under
  `docs/source/development/performance_equivalence_runs/`.

**Manual testing:** not complete. This is the main remaining switch blocker.

The next phase is to make it easy for developers who were not involved in the
refactor to run the branch, adapt their own parsets, inspect the new
Prefect/Dask dashboards, and report whether the branch is ready to replace
`master` for everyday use.

**Multi-sector mosaic:** low priority for the switch decision.

The path should retain targeted smoke/equivalence coverage, but it is rarely
used on `master` and should not block switching unless it exposes a broader
single-sector, imaging, or product-contract regression.

## Switch Criteria

The branch is ready to recommend over `master` when all of these are true:

1. **Evidence package is complete and reviewer-friendly.**
   `EQUIVALENCE_REPORT.md` summarizes the latest science and performance gate
   results, links to compact archived reports, explains accepted differences,
   and lists caveats plainly.
2. **Representative manual tests pass.**
   Developers outside the refactor run the current branch with real parsets
   and record outcomes, adaptations needed, runtime experience, output sanity,
   and dashboard/log usability.
3. **Parset migration is documented.**
   Users can adapt a `master` parset quickly, including calibration strategy
   changes, runtime options, existing h5parm/image-only workflows, local versus
   external Dask, and Prefect dashboard setup.
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
   A Spack recipe in `../ska-sdp-spack/packages/` can install/load this
   Prefect/Dask branch with the required Python and external-tool dependencies,
   without replacing the existing `py-rapthor` recipe until the switch decision
   is made.

## Outstanding Work Before Switching

Do these in order unless a regression blocks progress.

### 1. Manual Testing And Parset Migration Guide

Exercise and harden the manual-testing guide for developers testing this branch
with their own data. The first draft lives at
`docs/source/development/manual_testing_prefect_dask.rst`; keep it concise and
make sure it includes:

- quick-start commands for the dev container, a persistent Prefect dashboard,
  local Dask, and `rapthor input.parset`
- the smallest recommended parset edits for moving from `master` to the
  current branch
- how to tag runs with `prefect_run_tags`
- how to choose local Dask versus external Dask
- how to run several production jobs without a Prefect server: no
  `PREFECT_API_URL`, unique working directories, isolated temporary Prefect
  state per process, and optional external Dask where site policy allows it
- how to stage multi-node Slurm runs with external Dask and MPI WSClean,
  including the requirement that any Prefect API used by remote Dask workers is
  network-reachable from the allocation
- how to enable or disable command logging, command profiling, FITS previews,
  and postage-stamp previews
- how to adapt calibration configuration to the strategy-driven
  `calibration_strategy` interface
- image-only/applycal guidance: DI h5parm products are pre-applied; DD products
  are applied on the fly when matching directions are available
- where outputs, logs, command records, Prefect artifacts, Dask reports, and
  restart markers live
- how to record manual test outcomes and report issues

### 2. Manual Test Matrix For Switch Confidence

Ask developers not involved in the refactor to run a small but meaningful set
of real workflows:

- one default-like single-sector self-calibration parset
- one phase-only or calibration-light parset
- one DD phase plus DI full-Jones or otherwise mixed-calibration parset
- one image-only/applycal parset using existing DI and/or DD solutions
- one parset using a custom `calibration_strategy`
- one local dashboard run with `PREFECT_API_URL`
- two independent no-server runs submitted at the same time, each with a unique
  working directory, to confirm production-style parallel launches do not share
  Prefect SQLite state
- one external-Dask or Slurm staging run when the environment is available
- one multi-node Slurm imaging run with `imaging.use_mpi = True`, confirming
  WSClean MPI launches across the allocated nodes and does not oversubscribe
  node/thread resources

For each run, capture:

- branch/commit, parset, strategy, input-data summary, and run tag
- whether any parset changes were required
- return code and final operation state
- notable dashboard/log/artifact observations
- output sanity checks by the scientist who owns the data
- any performance surprises or operational friction

Multi-sector mosaic should be smoke-tested only if a tester already has a
relevant workflow. It is useful coverage, but it is not a primary switch
criterion.

### 3. Evidence-Driven Decision Pack

Prepare a reviewer/stakeholder pack before recommending the switch:

- latest `EQUIVALENCE_REPORT.md`
- latest science gate reports
- latest performance gate reports
- manual test matrix summary
- list of required parset adaptations
- list of accepted differences from `master`
- list of caveats and deferred work
- clear recommendation: switch now, switch with caveats, or keep `master`

This should be written for people who understand self-calibration and software
risk, but do not know the internal refactor history.

### 4. Spack Packaging For Staging And Production Tests

Add a new Spack package under `../ska-sdp-spack/packages/` as an alternative to
the existing `py-rapthor` recipe. Recommended working name:
`py-rapthor-prefect-dask` unless the deployment team prefers another module
name.

Recipe requirements:

- install this branch by tag/commit with `no_cache=True`, because Rapthor uses
  `setuptools_scm`
- expose the `rapthor` and `concat_linc_files` console scripts
- keep the old `py-rapthor` recipe untouched until the branch switch is
  approved
- remove the legacy `py-toil`/CWL dependency from the new recipe
- include Prefect/Dask runtime dependencies: `py-prefect`, `py-prefect-shell`,
  `py-dask+distributed+diagnostics`, and add/verify a `py-prefect-dask` recipe
  if the Spack repository does not already provide it
- include current Rapthor Python dependencies from `pyproject.toml`, including
  `py-bdsf`, `py-casacore`, `py-losoto`, `py-lsmtool`, `py-reproject`,
  `py-rtree`, `py-shapely`, `py-h5py`, `py-pyyaml`, and compatible
  `py-fastapi`
- include external runtime tools: `dp3`, `wsclean` with MPI support including
  `wsclean-mp`, `cfitsio+utils`, `aoflagger`, and the EveryBeam/Casacore
  dependency stack required by DP3/WSClean
- add module-load smoke checks: `rapthor --help`, `concat_linc_files --help`,
  Python imports for `rapthor`, `prefect`, `prefect_dask`, and
  `dask.distributed`, plus `DP3 --version`, `wsclean --version`, and
  `wsclean-mp --version`
- document the exact `spack install` and `spack load` commands used for manual
  and Slurm staging tests

### 5. Runtime UX Polish For Testers

Before broad manual testing, remove avoidable friction:

- make sure `rapthor input.parset` logs the selected runtime mode, Prefect API
  mode, Dask scheduler/dashboard, run tags, and working directory clearly
- make missing external-tool messages actionable
- make preflight/dry-run output easy to scan
- keep dashboard/task names readable and stable
- ensure command timing and task timing artifacts are easy to find
- verify reset/resume instructions still match the Prefect/Dask runtime
- add or keep a focused runtime smoke test proving concurrent no-server
  launches use isolated Prefect homes and independent working directories
- keep Slurm/MPI preflight messages clear: show allocated nodes, Dask scheduler,
  worker count, `imaging.use_mpi`, requested MPI processes, and WSClean thread
  counts before imaging starts
- adapt the Slurm/Dask launch pattern from the Prefect prototype scripts as
  the staging template: use a Dask scheduler on the first allocated node, one
  Dask worker per node, health checks before `rapthor` starts, explicit
  dashboard tunnel instructions, and a Prefect API URL that is reachable by
  remote workers when a persistent/temporary dashboard is used. Local checkout
  reference: `../ska-sdp-rapthor-prefect-prototype/aws-run-poc-multi-node.sbatch`
  (the same prototype may be available as `../rapthor-prefect-prototype` in
  other workspaces).

### 6. Benchmark And Performance Follow-Up

Do not start speculative optimisation until manual testers can run the branch.
When optimisation resumes, focus on the currently visible image-side
bottlenecks:

- `filter_skymodel`
- WSClean image runs and resource/concurrency policy
- calibration plotting only if repeated real runs show it matters

Benchmark rule:

- keep the default automatic `ci-benchmark`
- use `ci-benchmark-image-products` when changing image products,
  `filter_skymodel`, WSClean image behavior, or image post-processing
- use `ci-benchmark-predict-chunks` only for prediction scheduling changes
- use `ci-benchmark-wsclean-predict` only for calibration prediction setup or
  WSClean-predict paths
- leave many-sector mosaic benchmarks out of automatic CI unless changing that
  path

### 7. Final Pre-Switch Verification

Before recommending the branch as the new default:

```bash
python3 -m ruff check --fix --select I <touched-python-files>
python3 -m ruff format <touched-python-files>
python3 -m pytest -m "not integration" tests
RAPTHOR_TEST_RUN_ROOT=/tmp/rapthor-integration-runs \
  python3 -m pytest -m integration -vv -ra --durations=0 \
  tests/integration tests/operations/integration
```

Then rerun or refresh:

- the saved-reference science gate if scientific products changed
- branch repeatability/equivalence for the main decision scenarios
- the current CI benchmark scenario set
- at least one real-user manual parset after the final code changes

## What Is Already Done

- Owner-package execution architecture is in place for image, calibrate,
  concatenate, predict, mosaic, and pipeline flows.
- Operation adapters are thin; command builders, payload validation, output
  discovery, migrated helper logic, and flow wiring live under
  `rapthor/execution/<owner>/`.
- Runtime bootstrap supports no-server local runs, explicit Prefect API runs,
  local Dask, external Dask, and run tags.
- No-server runs blank `PREFECT_API_URL`, disable Prefect analytics, and use an
  isolated temporary Prefect home for each process, which is the right interim
  production mode until a shared Prefect server has a Postgres backend.
- Resource validation understands MPI command requests and checks that MPI
  WSClean is exclusive and does not request more processes than the configured
  Slurm node allocation.
- A new Spack recipe is still required for this branch. The existing
  `py-rapthor` recipe in `../ska-sdp-spack/packages/` is the legacy package and
  still carries the old Toil/CWL dependency.
- Calibration solve order is strategy-driven through `calibration_strategy`.
- Legacy implicit solve-slot behavior has been replaced with explicit solve
  types and order.
- DI scalar phase, DI diagonal slow-gain, and DI full-Jones products are
  pre-applied for image-only workflows; DD products are applied on the fly when
  directions match.
- Task observability is much stronger: readable flow/task names, tool tags,
  task timing JSONL, command timing artifacts, and persistent postage-stamp
  preview PNGs.
- Performance-equivalence gates pass for phase-only core and DD/full-Jones
  scenarios.

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
  image resource/concurrency policy. Relevant evidence is in
  `docs/source/development/benchmark_baselines/`, especially the 2026-07-08 to
  2026-07-11 benchmark reports.
- **Calibration plotting:** optimize only if larger real runs keep showing it
  as a meaningful post-processing cost.
- **WSClean prediction parallelism:** investigate splitting the internal
  frequency/facet loop inside WSClean prediction tasks only with a targeted
  benchmark and explicit resource limits.
- **Multi-sector mosaic:** keep smoke/stored-reference coverage available, but
  treat this as lower priority than common single-sector paths.
- **Slurm/MPI production hardening:** adapt the prototype multi-node launch
  pattern, validate external Dask workers on each allocated node, and prove MPI
  WSClean imaging in staging.
- **Spack deployment:** add the new Prefect/Dask branch recipe and module-load
  smoke checks before production-style manual testing.
- **Deferred code tidying:** split or simplify modules such as
  `rapthor.execution.image.diagnostic_calculation`,
  `rapthor.execution.image.flux_normalization`,
  `rapthor.execution.calibrate.h5parm_combination`,
  `rapthor.operations.calibrate.base`, and `rapthor.operations.image.base`
  only when changing behavior or when profiling/maintenance pressure justifies
  the edit.
- **Testing suite polish:** keep architecture and regression guards focused on
  payload serializability, thin operation adapters, task-boundary visibility,
  calibration strategy semantics, image-only apply behavior, and branch
  equivalence reporting.

## Evidence Locations

- Stakeholder summary: `EQUIVALENCE_REPORT.md`
- Science contract: `docs/source/development/science_equivalence_contract.rst`
- Performance contract:
  `docs/source/development/performance_equivalence_contract.rst`
- Science reports: `docs/source/development/science_equivalence_runs/`
- Performance reports: `docs/source/development/performance_equivalence_runs/`
- Benchmark reports: `docs/source/development/benchmark_baselines/`
- Current DD/full-Jones manual inspection run:
  `runs/equivalence-gate-dd-phase-plus-di-fulljones-20260712/`

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
