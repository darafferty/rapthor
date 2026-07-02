# Rapthor Architecture Refactor Plan

Status snapshot: 2026-07-02.

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
- Benchmark baseline scaffolding is in place: committed quick-demo/rich-demo
  scenario definitions, a developer runner, command-log parsing, JSON/Markdown
  report generation, and focused report tests.
- Dev containers install docs dependencies by default.

Recent verification has covered linting, non-integration tests, integration
tests, saved CWL equivalence, runtime bootstrap slices, and the rich
Prefect/Dask demo. Keep future verification notes in commit messages, CI
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

Status: started. The first committed slice defines benchmark scenarios and
report-generation code; the remaining work is to run repeated baselines and wire
the reports into CI artifacts.

Benchmark before changing Dask task boundaries, scheduler behavior, or
performance-sensitive execution code. The benchmark should identify what to
optimise next, not just produce one wall-clock number.

Tasks:

- Maintain committed benchmark scenario definitions, runner code, report
  parsing, and summarization tests.
- Use the quick demo for startup overhead, the generated rich demo for the
  representative Prefect/Dask graph, and later an optional larger science
  fixture outside the repo for realistic external-tool scaling.
- Run each scenario from a clean working directory with fixed runtime settings,
  including local Dask workers, command profiling, dashboard/report options, and
  external scheduler settings.
- Repeat each benchmark at least three times on the same machine/container
  image and report median plus min/max. Treat first-run cache effects
  separately.
- Capture:
  - total wall-clock time
  - operation and Prefect task durations
  - `logs/commands.jsonl` command timings
  - command resource profiles from `prefect_command_profile = time`
  - Dask performance report HTML
  - task count, task concurrency, worker idle time, and scheduler gaps
  - peak memory and disk footprint
  - output equivalence or checksum status for scientific products
- Add a CI benchmark job that can run manually or on schedule and publishes:
  - a Markdown benchmark report artifact
  - a JSON summary artifact
  - optionally Dask performance HTML, command logs, and selected run logs
- Keep bulky generated products and run directories out of git.

Done when:

- The benchmark harness and report-generation tests are committed.
- CI can produce a Markdown benchmark report artifact.
- A reproducible rich-demo baseline exists before Dask scalability changes.
- The report identifies the top wall-clock contributors and the biggest Dask
  idle or scheduler gaps.

### 3. Dask Scalability Guardrails

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

### 4. First Scalability Slices

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

### 5. Runtime UX And Contributor Docs

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

### 6. Deferred Targeted Refactors

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
