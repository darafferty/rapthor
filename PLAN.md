# Rapthor Architecture Refactor Plan

Status snapshot: 2026-07-04.

## Goal

Make the Prefect/Dask Rapthor pipeline easy to run, understand, test, debug,
benchmark, and scale while preserving the user-facing CLI workflow:

```bash
rapthor input.parset
```

This Prefect/Dask implementation has not been released, so prefer clean
production architecture over unreleased Python API compatibility, migration
aliases, compatibility shims, or test-only production surfaces.

## Current State

The main architecture cleanup is complete enough to move into benchmarking and
Dask scalability work.

- Execution code is organized by owner package: `image`, `calibrate`,
  `concatenate`, `predict`, `mosaic`, and `pipeline`.
- Image and calibration operations are package-based adapters, with payloads,
  builders, validation, commands, outputs, and flow wiring living near the
  operation-specific execution code.
- Migrated helper-script logic lives in importable execution modules. Production
  flows call these modules directly except where shell isolation is still useful
  for external tools or third-party multiprocessing.
- The installed `rapthor` command is exposed through `rapthor.cli:main`.
- `concat_linc_files` remains a supported installed utility through
  `rapthor.execution.concatenate.linc_cli:main`.
- Broad execution facades, normalized command wrappers, migration shims, unused
  runtime abstractions, and retired helper-script entry points have been removed
  or guarded by architecture tests.
- Command builders are deterministic and tested. Stable argument groups use
  option dataclasses where that improves readability.
- Scheduler-independent work units are separated from Prefect flow wiring for
  complex image and calibration paths.
- Prefect flow and task run names include operation, calibration mode where
  relevant, cycle, and coarse task identifiers.
- The development architecture docs include the current Prefect/Dask
  orchestration diagram.
- `calibration_strategy` is the production interface for solve type and solve
  order. Legacy `do_fulljones_solve` and `do_slowgain_solve` flags are retired
  from production configuration.
- Image-only cycles can carry forward prior calibration products when
  `do_image = True` and `do_calibrate = False`. DI calibration products are
  pre-applied during imaging preparation, while DD calibration products are
  passed to imaging for on-the-fly application.
- Initial CLI runtime bootstrap is in place for Prefect API mode, external
  Prefect health checks, external Dask scheduler checks, local Dask setup, local
  Dask worker sizing, and startup logging.
- A lightweight CLI smoke test starts from `rapthor input.parset` and covers
  parset path materialization, execution-config extraction, runtime bootstrap
  handoff, and top-level flow invocation without external astronomy tools.
- Test isolation closes temporary Rapthor run-log file handlers between tests,
  so parset reads that install `rapthor.log` handlers do not leak into later
  tests after temporary working directories are removed.
- Agent-facing repository guides live under `.agents/`, with `AGENTS.md`
  serving as the top-level routing and guardrail document.
- Benchmark baseline scaffolding is in place: a committed single CI benchmark
  scenario definition, generated benchmark parset/strategy inputs, a developer
  runner, command-log parsing, JSON/Markdown report generation, and focused
  report tests.
- The generated local demo parset and CI benchmark parset now share the same
  benchmark strategy, including the legacy DD default solve order, while using
  different resource defaults for local and CI runs.
- The CI base image installs runtime Python dependencies from `pyproject.toml`,
  and `pyproject.toml` participates in the base-image hash so dependency
  changes rebuild the image.
- Dev containers install docs dependencies by default.

Recent verification has covered linting, non-integration tests, integration
tests, saved CWL equivalence, runtime bootstrap slices, and generated
Prefect/Dask demo runs. Keep future verification notes in commit messages, CI
artifacts, or reports rather than growing this plan.

Known caveats:

- Use the prepared dev-container Python environment for local full-suite runs.
  Local tox-created environments may fail to build radio astronomy dependencies
  without system headers.
- Keep large integration, equivalence, and demo run roots on `/tmp` or another
  spacious filesystem.
- Pydantic remains a future option for configuration and payload validation.
  Keep contracts and builders clean enough that adopting it later would be
  incremental rather than a rewrite.

## Immediate Next Steps

Do these in order:

1. Rebuild or refresh the CI base/full images and run the manual or scheduled
   `run_benchmark_baseline` job.
2. Triage the three-repetition `ci-benchmark` artifact bundle:
   - confirm all repetitions finish under the two-hour CI limit
   - confirm the runtime parsets use the intended worker/thread settings
   - identify top wall-clock contributors from `commands.jsonl`, Dask HTML, and
     command resource profiles
3. If the benchmark is valid, commit a compact curated baseline report under
   `docs/source/development/benchmark_baselines/`. If it fails or times out,
   adjust the benchmark inputs or CI resource policy before starting Dask
   scalability work.
4. Tighten equivalence comparison next: improve FITS/image metrics, expose
   product statistics in reports, and add the branch-vs-master runner.
5. Only after a benchmark baseline and stronger equivalence signal are in
   place, start the Dask scalability guardrails and first task-boundary slices.

## Work Queue

### 1. Stabilization Gate

Status: complete enough to maintain while moving into benchmarking. Before
changing task granularity or performance-sensitive code, keep protecting the
current user-facing behavior.

Tasks:

- Keep parset/default contracts synchronized whenever runtime, cluster, or
  strategy options change:
  - `rapthor/settings/defaults.parset`
  - `rapthor/settings/defaults.json`
  - `tests/resources/*parset_dict.template`
  - `docs/source/parset.rst`
  - `docs/source/running.rst`
- Keep the tiny user-facing parset smoke lane that starts from
  `rapthor input.parset` and uses mocked external tools aligned with CLI
  startup, parset materialization, path handling, and runtime bootstrap.
- Keep `.agents/scientific_glossary.md` linked from `AGENTS.md` and the
  development docs, and use it as the naming reference for future refactors,
  especially for strategy tokens such as `slow_gains`.
- Keep test isolation for Rapthor run-log handlers intact when changing parset
  setup, working-directory handling, or logging.
- Keep image-only calibration-application coverage explicit:
  - supplied DD h5parm plus supplied sky model images on the fly
  - previous-cycle DI full-Jones pre-apply plus DD facet application
  - previous-cycle DD slow diagonal gains applied on the fly
  - previous-cycle DI slow gains pre-applied while DD phase solutions remain
    on the fly
  - user-supplied DI scalar phase or diagonal slow-gain h5parm pre-applied in
    image-only runs without requiring an input sky model
  - carried-forward DD h5parms use the calibration skymodel/facet directions
    from the cycle that produced the h5parm
  - image-only cycles with no detected sources keep empty sky models writable
    and continue with zero calibration patches rather than inventing directions
- Preserve the payload distinction between `prepare_data_h5parm` for DP3
  pre-apply and `h5parm` for imaging-time facet/DD application.
- Re-run the fast branch-health lane before scalability changes:
  - `tests/lib/test_parset.py`
  - `tests/execution/test_config.py`
  - runtime bootstrap tests
  - CLI tests
  - lint/import checks for touched files

Done when:

- CI and local dev runs agree on parset/default snapshots.
- A developer can change a parset option and know which tests and docs must move
  with it.
- One lightweight test continues to start from `rapthor input.parset`, not only
  from internal bootstrap helpers.
- The fast branch-health lane passes in the prepared dev-container environment.

### 2. Benchmark Baseline

Status: ready for a new CI benchmark run.

Current contract:

- CI runs one generated scenario, `ci-benchmark`, for three repetitions.
- The scenario uses two local Dask workers and a 30-thread
  `cpus_per_task`/`max_threads` budget per worker on the larger GitLab runner.
- The generated local demo parset and CI benchmark parset share the same
  strategy so behavior is comparable; only resource sizing differs.
- The strategy exercises DI phase calibration, DD phase/faceting, the legacy DD
  default solve order, full-Jones calibration, imaging, mosaicking, source
  filtering, and Dask/Python orchestration overhead.
- The CI artifact bundle includes summary/report JSON and Markdown, raw
  repetition results, command logs, Rapthor logs, generated parsets, Prefect
  server logs, and Dask performance reports.

Benchmark before changing Dask task boundaries, scheduler behavior, or
performance-sensitive execution code. The benchmark should identify what to
optimise next, not just produce one wall-clock number.

Tasks:

- Maintain the benchmark scenario definition, generated demo data, benchmark
  runner, report parsing, summarization tests, and CI artifact list together.
- Trigger the CI job after changes to dependencies, generated benchmark inputs,
  execution resources, task boundaries, or external command behavior.
- For a local smoke benchmark on a smaller workstation, override with
  `--scenario ci-benchmark --repetitions 1 --local-dask-workers 1 --cpus-per-task 4 --max-threads 4`.
- Treat first-run cache effects separately when interpreting the three
  repetitions.
- Keep bulky generated products and full run directories in CI artifacts or a
  local path outside the repository, for example
  `/tmp/rapthor-benchmark-artifacts/<pipeline-id>/`.
- If the run is a valid baseline, commit only a compact curated report under
  `docs/source/development/benchmark_baselines/<YYYY-MM-DD>-gitlab-60core.md`.
  Include the commit SHA, CI pipeline/job URL, container image tag, runner
  CPU/memory description, benchmark command, worker/thread profile, summary
  table, top wall-clock contributors, Dask idle/scheduler observations, and any
  caveats such as first-run cache effects.
- Commit a small machine-readable companion only if it is useful for future
  comparisons, for example
  `docs/source/development/benchmark_baselines/<YYYY-MM-DD>-gitlab-60core.summary.json`.
- Do not commit raw run directories, FITS/MS products, full Dask HTML reports,
  command logs, Prefect/Rapthor logs, or other bulky generated products.

Done when:

- The benchmark harness and report-generation tests are committed.
- CI produces a successful three-repetition Markdown/JSON benchmark report
  artifact for `ci-benchmark`.
- A reproducible generated benchmark baseline has been captured in GitLab
  artifacts and summarized in a compact curated report.
- The committed report identifies the top wall-clock contributors and the
  biggest Dask idle or scheduler gaps.

### 3. Saved Equivalence Renewal And Image Robustness

Status: saved-reference comparison is useful as a migration regression check,
but image evaluation should become stronger before relying on it as a long-term
scientific confidence gate.

The current saved CWL comparison checks product presence, operation state,
output-record shapes, h5parm dataset structure and values, sky-model summaries,
beam tables, exact text/region products, and FITS image aggregate statistics.
That is enough to catch many regressions, but aggregate FITS statistics alone
cannot prove that image structure is unchanged.

Tasks:

- Extend saved-reference comparison reporting so it emits the product-statistic
  tables directly to JSON and Markdown instead of requiring one-off analysis of
  products on disk.
- Keep the existing FITS finite-count, `mean`, `std`, `rms`, `min`, and `max`
  checks, but add image-difference metrics that are robust to tiny
  external-tool numerical drift:
  - same shape, finite mask, and selected science-relevant FITS header/WCS keys
  - residual `mean`, `std`, `rms`, `min`, `max`, and robust MAD-based noise
  - max absolute delta and max relative delta, with safeguards around zero
  - percentile residuals such as p50, p95, p99, and p99.9
  - residual RMS normalized by reference image RMS and robust noise
  - per-plane metrics for cubes and Stokes products rather than only whole-file
    summaries
- Add optional source- or peak-aware checks for final science images where that
  gives better confidence than pixel statistics alone:
  - brightest-pixel location and value within tolerance
  - compact source catalog summaries when PyBDSF or an existing catalog product
    is available
  - flux-scale checks for normalization scenarios
- Use product-class-specific tolerances where needed, for example dirty images,
  restored images, cubes, beam tables, and final mosaics, while keeping strict
  defaults for products expected to be deterministic.
- Ensure failed comparisons print enough context to decide whether the change is
  likely numerical jitter, a WCS/header change, a spatial image change, or a
  contract change.
- Keep intentionally changed scenarios explicit. Stale references should remain
  runnable by name or with `--include-stale-references`, and the replacement
  integration coverage for the new contract must be named in the report or
  script output.
- Add a reference-refresh path that can generate newer artifacts from the
  repository's `master` branch:
  - run `rapthor` from a clean `master` checkout or installed wheel inside the
    prepared dev container or CI image, not from the current feature branch
  - use frozen equivalence inputs, parsets, strategies, random seeds where
    applicable, and fixed runtime settings
  - record the `master` commit SHA, container image digest or tag, Python and
    external-tool versions, command line, parset, strategy files, and run root
  - store bulky reference artifacts outside the source tree or as CI artifacts,
    with only compact curated reports committed when useful
  - compare the current branch against those refreshed `master` artifacts with
    the same saved-reference harness
- Add an ad hoc branch-equivalence runner, for example
  `scripts/dev/run_branch_equivalence.py`, that accepts a parset or named
  scenario and runs the same input through `rapthor` from `master` and from the
  current branch:
  - create isolated `master` and current-branch run roots under a caller-provided
    output directory, defaulting to `/tmp`
  - support a single `--parset path/to/input.parset` for exploratory checks
    against arbitrary local datasets
  - support committed scenario manifests for repeatable checks, with parsets,
    strategies, expected data paths or data-preparation notes, tolerances, and
    scenario metadata kept under an organized tree such as
    `tests/equivalence/scenarios/<name>/`
  - keep Measurement Sets, FITS products, h5parms, logs, and bulky run products
    outside git; scenario manifests may point to external data locations, CI
    artifacts, or developer-supplied paths
  - run both branches with the same container image, runtime settings,
    environment overrides, parset materialization rules, and external-tool
    versions where practical
  - emit the same JSON and Markdown comparison reports as the saved-reference
    runner, including provenance for both branches and the exact command lines
  - allow developers to promote useful exploratory parsets into named scenarios
    once they are stable and documented
- Add focused tests for the image-comparison helper functions using synthetic
  FITS files that cover exact matches, small floating drift, localized spatial
  changes, cube-plane changes, WCS/header changes, NaN-mask changes, and true
  failures.

Done when:

- The saved-reference runner can regenerate reference artifacts from `master`
  and compare the current branch against them without manual checkout surgery.
- A developer can run one command with a parset or named scenario to compare
  `master` and the current branch on a chosen dataset.
- Repeatable equivalence parsets and scenario manifests are organized in the
  repository while large input and output data remain external.
- The equivalence report includes reproducible provenance for the reference
  artifacts and product-level image statistics.
- FITS comparisons would fail for meaningful spatial image differences even
  when global aggregate statistics are similar.
- The default saved-reference matrix remains robust to expected external-tool
  floating-point jitter.

### 4. Dask Scalability Guardrails

Make distributed boundaries explicit before making them finer grained.

Tasks:

- Add payload-size and serialization guard tests for image, calibration,
  predict, mosaic, and concatenate task payloads.
- Assert that worker payloads are plain serializable data, not `Field`,
  `Observation`, `Sector`, or operation instances.
- Add tests that assert each flow submits the intended task units.
- Extend resource-request coverage beyond image WSClean MPI paths.
- Add small representative fixture payloads for each operation.
- Make task-boundary tests assert stable task names where names carry useful
  domain identifiers such as mode, sector, chunk, observation, image type, or
  epoch.

Done when:

- Tests and docs make the current Dask task boundaries visible.
- A developer can see what data each boundary receives.
- Tests fail if a future refactor sends rich domain objects or oversized
  payloads to workers.

### 5. First Scalability Slices

Only split work where it improves dashboard clarity, scheduling, or restart
behavior without fighting external tools.

Tasks:

- Let predict post-processing for an observation start as soon as that
  observation's model-data outputs are ready.
- Split image-sector orchestration into clearer task boundaries:
  - prepare one imaging Measurement Set per observation
  - concatenate prepared Measurement Sets
  - run or reuse WSClean
  - filter source and skymodel products
  - run diagnostics
  - build image cubes and normalization products
  - compress final images when requested
- Split mosaic orchestration into template, per-sector regrid, final mosaic,
  and optional compression tasks.
- Keep calibration solve chunks as the primary calibration parallelism for now.
  Split collect, plot, and combine only if benchmarks show a bottleneck or a
  restart benefit.
- Keep DP3, WSClean, IDG, and PyBDSF as coarse external commands unless a proven
  library-level integration exists.
- Document which steps are distributed by Dask and which still run as coarse
  external commands or execution-owned module adapters.

Done when:

- A representative demo run shows useful task-stream activity in the Dask
  dashboard without oversubscribing threaded or MPI external tools.
- Restart and output-record behavior remains unchanged.
- Scientific equivalence checks still pass.

### 6. Runtime UX And Contributor Docs

Make Rapthor easier to run and easier to improve.

Tasks:

- Expand dry-run or preflight output to show planned operation order, task
  groups, resource hints, expected outputs, external tools, execution-owned
  module adapters, and unsupported multi-node features.
- Improve preflight messages for missing tools, unsupported container
  configuration, Slurm/external-Dask mismatch, missing Dask scheduler, and MPI
  WSClean assumptions.
- Show resolved Prefect API mode, Dask scheduler mode, local worker count, and
  dashboard URLs where known.
- Add short contributor docs/checklists for:
  - adding a parset option
  - modifying an operation
  - adding an external command helper
  - adding an execution-owned module adapter
  - adding a new flow task boundary
  - converting a legacy utility to an importable module
- Add a short "how to debug a failed flow" page covering logs, `.done` markers,
  `.outputs.json`, command records, Prefect run names, Dask dashboard views, and
  external-command stderr.

Done when:

- A user can preflight a run and understand likely failures without reading flow
  code.
- A contributor can find the owner module, expected tests, and docs updates for
  common changes.

### 7. Deferred Targeted Refactors

Do not split these modules just for tidiness. Split them when changing behavior
or when a smaller extraction clearly reduces risk:

- `rapthor.execution.image.diagnostic_calculation`
  - later split into photometry, astrometry, plotting, and orchestration helpers
- `rapthor.execution.image.flux_normalization`
  - later split into catalog loading, source matching, SED fitting, and h5parm
    writing helpers
- `rapthor.execution.calibrate.h5parm_combination`
  - later move toward named combination strategies
- `rapthor.operations.calibrate.base` and `rapthor.operations.image.base`
  - keep adapter size under review, but split only with behavior changes or
    testable extractions

Keep generated local noise (`__pycache__`, `.tox`, `.ruff_cache`, `runs`,
`htmlcov`, build outputs, temporary integration/equivalence/demo roots) out of
repo decisions.
