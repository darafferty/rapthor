# Post-Merge Development Plan

Updated: 2026-10-06.

## Scope and Priorities

This roadmap starts after `gec-468-ai-migrate-to-prefect` is merged into master.
Manual testing of the migration is nearing completion; the work below develops
and improves the resulting Prefect/Dask pipeline.

Base each change on the merged master and keep it in a focused branch and merge
request. The normal command remains `rapthor input.parset`. Keep temporary
`rapthor3` comparison-test changes outside production branches.

The suggested order is:

1. Record the merged baseline and carry forward outstanding issues.
2. Enforce resource budgets before increasing concurrency.
3. Validate and improve cluster deployment and monitoring.
4. Profile real workloads and implement measured performance improvements.
5. Extend coverage and simplify maintenance.

Independent work, such as catalog caching, CI improvements and a persistent
Prefect service, can proceed alongside resource allocation. Resource budgets
and tool thread limits must precede changes that run more heavy tasks at once.

## 1. Establish the Baseline and Carry Forward Issues

Record the merge revision, dependency versions and final manual-test findings
in the merge request or a linked compact baseline report.
Use that revision and its products as the reference for subsequent changes.
Turn outstanding findings into focused follow-up issues with a reproducer and
expected behavior. Keep large run products in ignored directories or artifacts.

Consolidate the existing branches:

- Retain `gec-618-resource-allocation` as a source for the resource work below.
  Extract its relevant changes onto the merged master rather than merging its
  older migration snapshot.
- Retire temporary CLI overlays once comparison testing no longer needs them.
  `gec-535-rapthor3-cli` also contains resource changes, so preserve useful work
  before retiring it; do not merge the whole branch for its CLI change.
- The filtering and WSClean environment fixes from `gec-629-fix-chunking` have
  already been ported. Its chunking fix is covered by the newer development
  implementation. Check the merged tree before carrying over any old patch.

Port audit, 2026-10-06: remote master was confirmed at `a998d5f7` (2026-10-02)
and compared with migration revision `b5bd0d83`. The entries below describe
remaining behavior or explicit review decisions, rather than unmatched commits
alone. Recheck them after the merge and retain only outstanding work. Remove
completed items and document deliberate differences with a reason.

| Area | Follow-up if still applicable |
| --- | --- |
| Scientific defaults | Reconcile calibration/imaging `bda_frequencybase` (`5000` on the reference master, `20000` on the migration branch; `a998d5f7`). Check scientific and runtime effects before changing the default. Add `normalization_reference_frequencies` to `defaults.json` where needed (`6d4df857`) and test consistency with parset defaults. |
| Normalization catalogs | Preserve known-survey correction/error metadata for supplied catalogs where appropriate (`dfa7f11a`); the migration branch currently uses correction `1` and error `0`. Cover flux-scale behavior with supplied and downloaded references. |
| Prediction | Review array-beam application (`23ed80e9`) and rendering/reordered-data reuse (`d587eec5`). Port required behavior into the Prefect execution owners with command, product and scientific comparisons. Verify named multi-facet DS9 regions with DP3 and WSClean (`d90786e8`): master supplies separate point-label and polygon-label formats, while the migration shares one region. Adapt formats only if the supported tool stack requires it. |
| Observation layout | Restore support for mixed station sets from `4c00305d` where required. Cover concatenation and residual-MS behavior as well as station selection. |
| Logging and failures | Port the duplicate-handler and uncolored file-log fixes (`fda65c8a`) if still needed. Improve access to the relevant task log and failure cause; command exceptions already carry command and exit information. |
| Packaging and builds | Apply the specific metadata, container and test-environment follow-ups below. Keep source-tool compatibility and NumPy's compiled-extension ABI consistent across supported images. |
| CI and tests | Adapt the Astron/SKA CI and integration-sharding changes below. Add the missing real facet-RMS and gridding integration checks listed in Coverage and Maintenance; these have unit/command coverage already. Preserve existing image-only and calibration-memory integration coverage. |

Preserve behavior through the current execution owners. Retired CWL templates,
Toil/StreamFlow runners and their staging mechanics do not need to be restored.

### Build, Packaging and CI Ports

- **Ubuntu APT sources (`a39e517b`, `d1fd7249`):** update the mirror rewrite in
  `Docker/Dockerfile` and `ci/ubuntu_24_04-base` to use
  `/etc/apt/sources.list.d/ubuntu.sources`, including the runtime stages. Retain
  Ubuntu 24.04 as the default; the useful build fixes are separate from the
  reverted Ubuntu 26.04 default change.
- **NumPy build/runtime consistency (`7eb0b03f`):** reconcile the standalone
  Dockerfile's `numpy<2` build with the CI images' NumPy 2 policy. Preserve the
  migration's Boost.NumPy/PyBDSF compatibility work and import checks. Verify
  compiled extensions in the final image, not just the builder.
- **SageCal/libdirac compatibility:** master resolves SageCal to
  `33d21c45000bf13e5e29077ba3413405c42c503f` for its GLib-free build; the
  migration resolver uses upstream `HEAD`. Choose a verified compatible source
  tuple and reflect it in the resolver, image labels and build-cache key.
- **Package metadata and releases (`a39e517b`, `583dc808`, `70b561e5`):** align
  the supported/tested Python range with master's Python 3.10–3.14 policy;
  adopt the SPDX expression, license-file metadata and `setuptools>=77`.
  Evaluate `lsmtool>=1.9.0` in place of the old Git pin, verifying the APIs and
  scientific products used by the execution owners.
- **Compiled test dependencies and beam data (`a39e517b`):** pass
  `EVERYBEAM_DATADIR` through integration tox. Review master's
  `sitepackages = true` change so tests use the intended container-installed
  astronomy libraries without rebuilding an incompatible extension stack.
  Preserve the migration's isolated Prefect state and test run roots.
- **Astron/SKA CI (`be6642f8`):** port common/site-specific jobs, runner and
  finalizer setup. Bound unit-test workers to the runner's allocation; master
  uses eight rather than `-n auto` to avoid excessive workers and OOM failures.
  Keep field tests and Prefect-server tests serial. Reproduce the mirror's
  multiprocessing socket-path fix with a tested Python >=3.13.7 integration
  environment; master uses `tox-uv` with managed Python 3.13. Retain short
  temporary paths and verify the actual Prefect runs on the mirror.
- **Integration balancing (`3e352b73`, `c3fac822`):** regenerate duration data
  for the current Prefect tests and use `--durations-path` with
  `--splitting-algorithm least_duration`. Review four versus eight shards using
  measured job times and runner resources. Keep each shard's state separate.

## 2. Resource Allocation and Safe Concurrency

Use the GEC-618/GEC-619 changes as starting material: gridding/shared-facet
selection (`63403e6b`, `1ac71a94`), configurable WSClean/DP3 thread counts
(`ea2c08fd`) and resource allocation (`b7dfd010`). Review each against the
current execution architecture rather than assuming it applies unchanged.

Implement this work in dependency order:

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

Verify one-worker, multiple-worker and MPI configurations with real command
execution. Record total CPU usage, peak memory and elapsed time, and confirm
that products and failure/restart behavior remain consistent. Increase default
concurrency only when the combined workload stays within its allocation.

## 3. Cluster Deployment and Monitoring

- **Slurm, external Dask and MPI:** run a representative multi-node imaging
  workload. Exercise worker-local scratch, shared scratch, product promotion,
  interrupted runs and restart. Existing opt-in Slurm smoke tests check
  allocation/scheduler access; extend coverage to the actual imaging path.
- **Dashboard access:** demonstrate Prefect and Dask dashboards from a local
  browser while the run executes on a cluster. Have the launcher print or write
  usable SSH tunnel commands and document the scheduler/worker setup.
- **Persistent Prefect service:** provide a managed Postgres-backed service for
  users who need one dashboard and durable history for independent jobs.
  Document startup, configuration and recovery. Keep isolated ephemeral state
  available for standalone runs; shared SQLite is unsuitable for the intended
  concurrent production service.
- **Installation:** verify the supported container and site/Spack/module paths
  on representative production systems and update their instructions.

Document the tested allocation, tool versions, storage layout and recovery
results in [running](docs/source/running.rst) and
[architecture](docs/source/development/architecture.rst). Keep Ubuntu 24.04 as
the supported container baseline while Ubuntu 26.04's reported memory problems
are investigated separately.

## 4. Measured Performance Improvements

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

### Parallelization Investigations

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

## 5. Coverage and Maintenance

- **Facet-RMS integration (`eb1b6f2f`):** adapt master's
  `test_sector_diagnostics.py` scenario to verify actual `facets_rms` JSON from
  a pipeline run. Assert flat-noise and beam-corrected statistics, including
  mean, median, standard deviation, minimum and maximum. The calculation and
  focused unit tests are present; this checks the full product path.
- **Parallel-gridding integration (`38abdb92`):** adapt the matrix of task
  counts 1/2, full/single DDE modes and serial/MPI-wrapper routing. Check the
  executed WSClean commands and products rather than CWL log filenames.
  Master's MPI wrapper invokes serial WSClean; real MPI validation remains
  part of the cluster workstream.
- **Synthetic Measurement Sets (`da442dfc`):** port flag/weight initialization
  into `tests/integration/conftest.py:ms_for_normalisation`. After replacing
  UVW and DATA to describe a new synthetic observation, reset `FLAG` and
  `FLAG_ROW` to false and `WEIGHT`/`WEIGHT_SPECTRUM` to one. Verify the fixture
  starts with the intended signal and weighting rather than seed-MS state.
- **Integration CPU limits (`c3fac822`):** cap smoke-fixture CPU requests by
  available CPUs, as master's `min(cpu_limit, misc.nproc())` does, instead of
  always requesting six. Keep gridding/thread settings within that allocation.
- **Shared-facet I/O:** resolve the tool failure behind the
  `shared_facet_rw = True` case's `xfail(run=False)` and execute that path in a
  supported environment. Its `False` control already runs.
- **Broader scientific paths:** add representative SKA-Low and screens/IDGCal
  coverage as their target environments become available. Historical
  equivalence evidence covers selected LOFAR HBA paths.
- **Multi-sector mosaic:** maintain targeted smoke/product coverage. Give it
  lower priority than common imaging unless a failure affects shared contracts.
- **Legacy strategies:** document the transition to `calibration_strategy` and
  schedule removal of the `do_slowgain_solve`/`do_fulljones_solve` translation.
  Replace deprecated flags with an actionable error after the documented
  deprecation period, and retire the compatibility-only strategy helper.
- **Supported strategy combinations (`0ebc0690`):** review any required
  master workflows excluded by the migration's explicit solve-combination
  validation, such as DI `medium_phase` alone or DD `full_jones`. Treat this
  as a support decision; expand combinations only with execution and scientific
  product coverage, preserving explicit solve order.
- **Module and test cleanup:** simplify large diagnostics, normalization,
  h5parm-combination and operation modules when maintenance or behavior changes
  justify it. Consolidate shared test helpers where useful. Preserve guards for
  serializable payloads, thin operation adapters, calibration semantics,
  image-only application, restart and scientific products.
  Use the GEC-486 helper work (`5f59fe87`, `6c58212b`, `14377a61`, `4cc0b732`)
  where it removes duplication; scope integration-only fixtures to integration
  tests. Current pytest conversions and repository-wide formatting already
  cover the corresponding master changes.

For each follow-up, run focused tests and the relevant tool/integration case.
Scientific, default or scheduling changes need product comparisons against the
recorded baseline. Update defaults, user documentation, examples and command
contracts together when behavior changes. Keep the architecture diagrams
aligned with new flow/task boundaries.

## Tracking and Evidence

Create separate issues/merge requests for the workstreams and record their
priority, dependencies, scope and verification results. Update this roadmap as
work lands; remove completed tasks rather than accumulating another migration
status report.

Manual testing supersedes the historical migration equivalence reports. The
[upgrade guide](docs/source/upgrading.rst) records the intentional behavior and
output differences from the CWL/Toil implementation, with links to their
maintained documentation. Keep new verification results with the relevant
issue or merge request, without committing large Measurement Sets or raw
products.
