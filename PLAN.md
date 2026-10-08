# Post-Merge Development Plan

Updated: 2026-10-08.

## Scope and Review Baseline

This work starts after `gec-468-ai-migrate-to-prefect` is merged into master.
Implement each task on the merged master in a focused branch and merge request.
The normal command remains `rapthor input.parset`; temporary `rapthor3`
comparison overlays do not belong in production branches.

- **Part A: missing master functionality and bug fixes.** These are identified
  gaps, missing regression coverage, and explicit compatibility decisions.
- **Part B: future improvements.** These extend or improve the resulting
  Prefect/Dask pipeline; they are not missing ports from master.

The audit compared local `master` and `origin/master`, both at `ff1f29d9`
(2026-10-08), with migration HEAD `b5884ade` (2026-10-06). Their common ancestor
is `2e21be62`. There are 64 master-only commits and 632 branch-only commits.
The review used both histories, the final trees, the previous plan and the
[upgrade guide](docs/source/upgrading.rst). Commit ancestry alone does not show
which features are missing: many master changes were reimplemented during the
migration. This is a source review, not a new scientific-equivalence run.

After the merge, record its revision, dependency versions and final manual-test
findings in the merge request or a compact baseline report. Recheck Part A
against that tree and any newer master commits before implementation. Turn
remaining findings into issues with the task IDs below, a reproducer, expected
behavior and verification results. Keep large run products in ignored storage
or artifacts.

## Part A — Missing Functionality, Bug Fixes and Regression Coverage

Unchecked tasks below were outstanding at the reviewed revisions. Source hashes
refer to master unless explicitly labelled as migration commits; inspect them
with `git show <hash>`. Port behavior into the current execution owners, using
plain serializable payloads and the current output/finalizer/restart contracts.
Do not transplant old CWL workflows or merge an older migration snapshot.

Suggested priority: scientific behavior/defaults (P01, P03, P04, P06), build and
CI reliability (P09–P12), then the remaining ports. P02 follows P01 and the facet
format decision D02. P05 should use P04's catalog metadata behavior. The test
ports can proceed independently and should protect later changes.

### Scientific and User-Visible Behavior

#### P01. Apply the full beam during WSClean prediction

- [ ] Port `23ed80e9` into the calibration prediction path.
- **Gap:** the branch has no `wsclean_predict_beam_interval` option, predicts
  without `-apply-facet-beam`, and still sets DP3's `applybeam.beammode` and
  `solve1.beammode` to `array_factor`.
- **Change:** add the option with master's 120-second default; pass it through
  field/operation state, payloads and validation to WSClean's
  `-facet-beam-update`. Use the required `-model-fpb.fits` names and disable the
  corresponding DP3 beam application when WSClean has already applied it.
  Main implementation: `rapthor/execution/calibrate/{prediction,commands,builders}.py`
  and `rapthor/operations/calibrate/`.
- **Done when:** both defaults, parset docs and test templates include the
  option; command and real multi-facet tests demonstrate one beam application,
  correct model-column names and consistent fluxes for WSClean and DP3 paths.

#### P02. Port improved WSClean model rendering and reordered-data reuse

- [ ] Adapt `d587eec5` to the current prediction tasks.
- **Gap:** the branch uses the supplied image geometry and predicts each
  frequency group/facet separately with `-no-reorder`. Master derives rendering
  resolution from the sky extent, UVW coordinates and frequency, limits the
  image size, predicts the full band per facet, and reuses reordered data.
- **Change:** implement the rendering calculation and bounded fallback in
  `rapthor/execution/calibrate/prediction.py`; extend command options for
  `-channels-out`, `-parallel-reordering`, `-save-reordered` and
  `-reuse-reordered`. Keep reusable scratch alive through all facet consumers,
  isolate it by chunk, and preserve recovery behavior on failure.
- **Done when:** single-band, multi-band and single-channel cases cover every
  requested channel; model visibilities agree within documented tolerances;
  CPU, memory, scratch and elapsed time are measured. Preserve the migration's
  correction of WSClean's end-exclusive channel ranges.

#### P03. Accept observations with different station sets

- [ ] Port `4c00305d` across field, imaging and residual-product handling.
- **Gap:** `rapthor/lib/field.py` still rejects differing station lists, and
  `rapthor/execution/image/preparation.py` always concatenates prepared MSs in
  time. Master takes the station union and passes separate MSs to WSClean when
  their station sets differ.
- **Change:** build a deterministic station union, retain valid calibration
  station constraints, and allow the image payload/command path to carry a
  list of prepared MSs. Match master's disabling of residual-MS creation for
  this case, or implement a separately validated residual path. Adapt the MS
  used for normalization and all affected output/finalizer/restart records.
- **Done when:** same-station and mixed-station multi-epoch tests run through
  calibration and imaging; serial/MPI commands receive the correct inputs;
  residual requests cannot produce incomplete or misleading products. Document
  the supported observation layout and residual limitation.

#### P04. Preserve normalization metadata for supplied survey catalogs

- [ ] Port the normalization changes in `dfa7f11a` and the JSON default fix in
  `6d4df857`.
- **Gap:** `_get_survey_metadata()` in
  `rapthor/execution/image/flux_normalization.py` assigns every supplied catalog
  correction `1.0` and error `0.0`. Master recognizes known survey frequencies
  and retains their flux correction and uncertainty. The branch also omits
  `normalization_reference_frequencies` from `defaults.json`, although parsing
  and `defaults.parset` already support it.
- **Change:** preserve WENSS/VLSSr metadata for supplied survey catalogs and the
  neutral fallback for unrecognized frequencies; add the missing JSON entry.
- **Done when:** supplied and downloaded versions of the same surveys yield
  consistent normalization; tests cover known/unknown frequencies, invalid
  frequency lists and offline use. Keep the documented reference-frequency and
  already-corrected/custom-catalog semantics explicit.

#### P05. Restore a usable offline catalog-preparation workflow

- [ ] Adapt the catalog-prefetch functionality from `dfa7f11a`.
- **Gap:** master provides `fetch_skymodel.py` and an HBA preparation workflow
  which downloads TGSS, Pan-STARRS, VLSSr and WENSS catalogs before an offline
  run. The branch accepts supplied catalogs but has no equivalent preparation
  helper or workflow.
- **Change:** provide a maintained command/module that reads the MS pointing,
  downloads the requested survey/sky area on an internet-enabled machine, and
  documents how the files and normalization frequencies feed the existing
  parset options. Put domain/execution logic in the appropriate owners and any
  entry point in `pyproject.toml`; retain the Prefect architecture.
- **Done when:** a prepared example completes normalization and diagnostics
  with `allow_internet_access=False` and no survey queries from workers. Test
  pointing/radius, output paths and download failures. Per-run catalog caching
  in B3 is a separate improvement.

#### P06. Restore master's less aggressive frequency-BDA defaults

- [ ] Reconcile `a998d5f7`: `bda_frequencybase` is `5000.0` for calibration and
  imaging on master, but remains `20000.0` in both branch defaults.
- **Change:** adopt `5000.0` in `defaults.parset`, `defaults.json`, documentation
  and expected-default templates, unless scientific review explicitly chooses
  and documents a different value. Keep the time-BDA default unchanged.
- **Done when:** parsing and command tests agree on the default; representative
  calibration/imaging runs record smearing, flux/RMS, memory and runtime
  effects. Explicit user overrides and frequency-only BDA must still work.

### Logging and Failure Diagnosis

#### P07. Eliminate duplicate console records and ANSI codes in file logs

- [ ] Adapt `fda65c8a` in `rapthor/_logging.py`.
- **Gap:** the branch still adds a console handler alongside existing handlers
  and wraps file-handler emission in ANSI coloring. The coloring wrapper also
  mutates the shared log record.
- **Change:** make Rapthor's logging setup idempotent and keep color confined to
  console formatting. Account for Prefect's handlers when adapting master's
  root-handler cleanup.
- **Done when:** repeated setup and Prefect execution emit each Rapthor message
  once to its intended console/file destination; file logs contain no ANSI
  codes and retain the module logger names.

#### P08. Put the task log and failure cause in the surfaced exception

- [ ] Preserve the user-facing failure context introduced by `aabe35f2`.
- **Gap:** `rapthor/execution/shell.py` already records task output and raises
  `ShellCommandError` with the command and return code. The exception does not
  identify its output log or include the tool's diagnostic; master surfaces
  those diagnostics via its old workflow-log parser.
- **Change:** attach the relevant task-log path and a bounded useful output
  excerpt to failures in the current shell/flow path. Preserve the original
  exception and task/operation context across Prefect/Dask boundaries.
- **Done when:** a failing DP3/WSClean fixture surfaces its cause, exit code and
  readable log path through the CLI and leaves the full log available. Reuse
  the existing failed-command tests; CWL log parsing is no longer needed.

### Builds, Packaging and CI

#### P09. Fix Ubuntu 24.04 APT source rewriting in all image stages

- [ ] Port the applicable build fixes from `a39e517b` and `d1fd7249`.
- **Gap:** `Docker/Dockerfile` and `ci/ubuntu_24_04-base` still rewrite
  `/etc/apt/sources.list`; master uses
  `/etc/apt/sources.list.d/ubuntu.sources` and also handles runtime stages.
- **Done when:** builder and runtime stages build using the intended mirror;
  CI and dev-container targets still use Ubuntu 24.04. The reverted Ubuntu
  26.04 default is not part of this port.

#### P10. Align the standalone image's NumPy build and runtime

- [ ] Complete the ABI fix from `7eb0b03f` in `Docker/Dockerfile`.
- **Gap:** the standalone build still installs `numpy<2`; the branch's CI image
  uses NumPy 2 and has additional Boost.NumPy/PyBDSF compatibility work.
- **Change:** align the standalone build/runtime dependency policy with the
  supported CI image, retaining the migration's compiled-extension fixes.
- **Done when:** NumPy, PyBDSF and python-casacore import in the final standalone
  and CI images and a small image/source-finding run succeeds. Checking only
  the builder image is insufficient.

#### P11. Update package metadata and the compiled test environment

- [ ] Adapt `a39e517b` and `583dc808` in `pyproject.toml`.
- **Gap:** the branch advertises Python >=3.9 and tests 3.9–3.13; master declares
  >=3.10 and tests 3.10–3.14. The SPDX license expression, `license-files`,
  `setuptools>=77`, integration `EVERYBEAM_DATADIR` passthrough and master's
  `sitepackages = true` test configuration are also absent.
- **Change:** align the supported/tested Python policy after checking the
  Prefect dependency stack; adopt the metadata and beam-data environment fixes.
  Decide explicitly how tox obtains compatible compiled astronomy libraries
  before applying `sitepackages` to the migration's environments.
- **Done when:** package metadata builds, the declared Python environments can
  import/run the supported stack, and integration tests locate EveryBeam data.
  Preserve isolated Prefect state/run roots. `lsmtool>=1.9.0` is already present
  and needs no port.

#### P12. Restore Astron/SKA CI support and bounded, balanced test execution

- [ ] Adapt `be6642f8`, `3e352b73` and `c3fac822` to the Prefect test suites.
- **Gap:** master has common/Astron/SKA CI files and mirror runner/finalizer
  setup. The branch has a single CI file, uses `-n auto`, lacks stored duration
  balancing, and retains fixed six-CPU integration settings.
- **Change:** port the site configuration and cap unit workers to the runner's
  allocation (master uses eight to avoid OOM). Keep field and Prefect-server
  tests serial. Reproduce the mirror's multiprocessing socket-path fix with
  a tested Python >=3.13.7 integration environment; master uses `tox-uv` with
  managed Python 3.13. Keep short temporary paths and isolated state per shard.
  Regenerate durations for current tests, use `--durations-path` and
  `--splitting-algorithm least_duration`, and choose four/eight shards from
  measured times. Cap integration CPU requests with the equivalent of
  `min(cpu_limit, misc.nproc())`, including dependent thread/gridding settings.
- **Done when:** both site configurations validate and run unit, integration,
  documentation and finalizer jobs; collected tests are assigned once across
  shards, and small CPU allocations do not request more cores than available.

#### P13. Add the Docker dependency-version inspection helper

- [ ] Port `f75dfe53`: `Docker/extract_version_hashes.sh`, its tests and usage
  documentation. None is present on the branch.
- **Change:** expose the image's existing `nl.astron.rapthor.*.version` labels
  as `NAME_COMMIT=value` assignments. Place the tests in the current test layout
  rather than reviving the retired scripts package.
- **Done when:** a fake Docker executable tests matching/missing labels,
  hyphenated names, values containing `=`, and inspection failures (including
  partial output); the documented invocation works on a built image.

### Missing Regression Tests and Documentation Corrections

#### P14. Verify facet-RMS diagnostics through a real pipeline run

- [ ] Adapt `eb1b6f2f`'s `test_sector_diagnostics.py` scenario.
- **Gap:** facet-RMS calculation and unit coverage exist, but the branch lacks
  master's end-to-end assertion on the saved `facets_rms` JSON.
- **Done when:** an integration run verifies each expected facet's flat-noise
  and beam-corrected mean, median, standard deviation, minimum and maximum in
  the final diagnostics product. Preserve the existing out-of-image-facet fix.

#### P15. Restore the parallel-gridding integration matrix

- [ ] Adapt `38abdb92`'s `test_wsclean_parallel_gridding.py`.
- **Gap:** operation/command coverage exists, but the real-tool matrix for task
  counts 1/2, full/single DDE modes and serial/MPI-wrapper routing is absent.
- **Done when:** tests inspect executed commands via `commands.jsonl` and verify
  output products. Master's wrapper calls serial WSClean; this checks routing,
  while real multi-node MPI validation remains B2. Respect P12's CPU limits.

#### P16. Reset flags and weights in the synthetic normalization MS

- [ ] Port the fixture correction from `da442dfc` to
  `tests/integration/conftest.py:ms_for_normalisation`.
- **Gap:** the fixture replaces UVW and DATA but inherits flags and weights from
  the seed MS, although it now represents a different synthetic observation.
- **Change:** initialize `FLAG`/`FLAG_ROW` to false and
  `WEIGHT`/`WEIGHT_SPECTRUM` to one before predicting the synthetic sources.
- **Done when:** fixture checks and normalization integration tests establish
  the intended unflagged signal and unit weights without changing the seed MS.

#### P17. Port remaining applicable documentation fixes

- [ ] Apply the remaining relevant parts of `ff1f29d9`.
- **Gap:** `docs/source/preparation.rst` still says initial DI calibration must
  be in `DATA`, instead of the column selected by `data_colname`.
  `Sector.set_imaging_parameters()` in `rapthor/lib/sector.py` also lacks the
  literal-block markup added on master for its expected dictionary keys.
- **Done when:** these corrections are present and the Sphinx build verifies
  the affected pages. Most of master's API/glossary/strategy corrections are
  already covered by the rewritten migration docs; retain their Prefect paths,
  solution semantics and legacy-option rules when reconciling text.

### Compatibility Decisions and Existing Limitations

These items need an explicit result before they can be closed. A missing code
fragment alone does not establish a regression in these cases.

- [ ] **D01 — Supported solve combinations (`0ebc0690`).** Master validates
  mode/solve names more permissively; the branch's
  `rapthor/lib/strategy.py:_validate_calibrate_strategy` allows only specific
  sequences. Inventory required master workflows currently rejected (for
  example DI `medium_phase` alone and DD `full_jones`). Establish which actually
  execute correctly on master, then port each required sequence with ordered
  solve, application, seeding and product tests, or document the deliberate
  restriction in the upgrade guide. Do not infer support from parsing alone.
- [ ] **D02 — Named facet-region formats (`d90786e8`).** Master writes separate
  point-labelled and polygon-labelled DS9 files; the branch shares one helper
  in `rapthor/execution/regions.py`. Test multiple named facets with the
  supported DP3/WSClean/LSMTool versions. Add separate formats and payload paths
  only where needed, and verify selected facets match model columns and h5parm
  directions. Resolve alongside P01/P02.
- [ ] **D03 — SageCal/libdirac source compatibility.** Master's
  `Docker/fetch_commit_hashes.sh` pins
  `33d21c45000bf13e5e29077ba3413405c42c503f` for its GLib-free build; the branch
  resolves upstream `HEAD` after migration commit `310fbb04`. Build and test the
  intended source tuple before deciding whether to restore the pin. Keep the
  resolver, standalone build defaults, labels and build-cache key consistent.
- [ ] **L01 — Shared-facet I/O tool failure (existing limitation).**
  `tests/integration/test_shared_facet_rw.py` marks the enabled case
  `xfail(run=False)` because serial WSClean 3.7 aborts; the disabled control runs.
  Reproduce with the supported tool stack, resolve the failure or explicitly
  constrain support, and execute the enabled test. This is an unresolved
  functionality issue already in the plan, not a newly identified master port.

### Changes Already Covered or Superseded

Keep these out of the port backlog unless the merged tree loses the behavior.
This table records source-review findings, not fresh runtime validation.

| Master change | Evidence / disposition on the migration branch |
| --- | --- |
| Balanced observation chunking (`b8312075`) | Ported by `e74be379`. The chunking methods in `rapthor/lib/observation.py` match master; lifecycle wiring and real-MS boundary tests are present. Do not reapply older GEC-629 chunking patches. |
| LSMTool release dependency/API (`70b561e5`, `9fa90768`) | `2168c1c6` sets `lsmtool>=1.9.0`; field/region helpers use `read_from_skymodel`. Remove the old plan's dependency-upgrade task. |
| Example strategies and initial-model documentation (`302714f8`, `4d62d3a8`) | The four updated custom/default calibration/imaging example files already match master. The strategy docs explain the initial-model-dependent phase-only cycles. |
| Time-ordered concatenation (`bf4608ef`), relaxed station diameter (`56a43f84`) | Present through `e7d2727a` and `d201dded`. Mixed station *sets* still need P03. |
| Facet RMS, missing/invalid regions, off-image facets (`eb1b6f2f`, `908f83c9`, `da442dfc`) | Runtime behavior exists in the image execution owner (`ffa64f6d`, `f3617f41` and subsequent fixes). Only the test/fixture gaps in P14/P16 remain. |
| Astrometry corrections (`ebe35408`) | Ported by `c0127ac6`, with subsequent error handling and survey-fallback fixes including `b5884ade`. |
| Later-cycle normalization and empty-source handling (`fc79ef7f`, `3e4eca19`) | Implemented in strategy/normalization execution and tests; `5974f354` records the catch-up. Supplied-survey metadata still needs P04. |
| Basic WSClean prediction and configurable narrow-band models (`d90786e8`, `e8867abd`, `c4dfcfd8`) | Present through `5c3c9471` and `d201dded`; end-exclusive channels fixed by `706556f6`. Remaining changes are P01/P02/D02. |
| Model-data and residual-visibility products (`971a2b25`, `17448437`) | Ported by `d51e08a2`, including current output records/finalizers. |
| Parallel gridding/shared-facet selection (`38abdb92`, `01a81e11`) and per-sector input shape (`18cf2d72`) | Current `Image` builds one gridding value per sector; builders preserve that mapping. Selection behavior was ported by `5974f354`/`d201dded`; remaining integration coverage is P15/L01. |
| Flexible strategies and image-only application (`0ebc0690`) | Present with explicit execution validation and image-only integration cases. Only the support decision D01 remains; retain documented cycle/seeding differences. |
| Imaging frequency BDA and averaging limits (`c6f4375b`, `15d4ccb0`, `3fd9e69f`) | Implemented with operation, command and integration coverage. P06 addresses the later default reduction. |
| Calibration-memory preflight/per-cycle checks (`043c15d4`) | Ported by `4fd75842`; unit/integration coverage remains. Node-wide scheduling budgets in B1 are additional work. |
| Reset with missing directories (`e25b4a6a`) | `rapthor/modifystate.py` skips absent product directories. |
| Ubuntu 24.04, libdeflate, current measures-table URL (`37f6fc06`, `0e163498`, `bc2c65a7`, `d1fd7249`) | Present. Do not restore `488f5c00`'s temporary fallback URLs superseded on master. `b27ba6e6`'s streamed download is an implementation difference; the branch uses the same URL and removes its temporary archive. Outstanding build differences are P09/P10/D03. |
| Earlier CI fixes (`c853f707`) | Test sharding/thread-environment settings and separate supplied/downloaded normalization cases are present. The old LSMTool pin is superseded; remaining CI changes are P12. |
| CWL-only fixes (`b457f49c`, `c28c9ae2`, `e8873f19`) | Duplicate CWL fields, `pickValue`/nullable outputs and staging symlinks belong to retired workflows. Prefect owns outputs and direct input paths; no CWL restoration is needed. |
| Calibration `avg` → `bdaavg` rename (`dbb8993c`) | Both versions explicitly use `bdaaverager`; this is a step-label change, not missing averaging behavior. Keep any later naming cleanup separate from ports. |
| Formatting (`f826c262`, `13e3bd8e`, `6c74e20d`, `f379cf15`), pytest/helper refactors (including `0c422261`) and obsolete DP3 `writefullresflag` removal (`b307e769`) | Covered or superseded by the new modules/tests; no wholesale test-helper or formatting port. Relevant remaining fixture/CI changes are P12/P16. |

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
  concurrent production service.

Keep Ubuntu 24.04 as the supported container baseline while Ubuntu 26.04's
reported memory problems are investigated separately.

### B3. Measured Performance Improvements

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
