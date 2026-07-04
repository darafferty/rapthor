# Rapthor Architecture Refactor Plan

Status snapshot: 2026-07-04.

## Goal

Make the Prefect/Dask Rapthor pipeline easy to run, understand, test, debug,
benchmark, and scale while preserving the user-facing CLI workflow:

```bash
rapthor input.parset
```

This Prefect/Dask implementation has not been released, so prefer clean
production architecture over unreleased compatibility shims.

## Current Position

The architecture cleanup is complete enough to focus on scientific confidence,
benchmark-guided scalability, and test maintainability.

Done:

- Execution code is organized by owner package:
  `image`, `calibrate`, `concatenate`, `predict`, `mosaic`, and `pipeline`.
- Operation adapters are thin. Payload builders, validation, command builders,
  output discovery, and flow wiring live in execution owner packages.
- Prefect/Dask runtime bootstrap, local/external Dask setup, CLI smoke coverage,
  and dashboard/report logging are in place.
- Benchmark scaffolding is in place for the single `ci-benchmark` scenario,
  including timestamped CI artifacts, GitLab metadata, Dask report parsing, and
  operation-boundary timing.
- Dask task-boundary guardrails now cover owner-flow submissions, plain
  serializable worker payloads, readable task names, and representative payload
  fixtures.
- Resource-propagation guardrails now cover CI-style demo/benchmark overrides
  into local Dask startup, effective `ExecutionConfig`, runtime parsets, and
  representative operation payload thread fields.
- The first FITS image-equivalence robustness pass is implemented in
  `scripts/dev/run_saved_cwl_equivalence.py`: residual metrics, finite-mask
  checks, WCS/header checks, pixel comparison, per-plane cube/Stokes metrics,
  JSON product statistics, and Markdown report output.

Known caveats:

- Use the prepared dev-container Python environment for tests, formatting,
  integration checks, equivalence checks, and demo/benchmark runs. Local tox
  environments may fail to build astronomy dependencies.
- Keep large integration, equivalence, benchmark, and demo run roots outside
  git, preferably under `/tmp` or CI artifacts.
- Do not commit raw run directories, FITS/MS products, command logs, full Dask
  HTML reports, Prefect logs, `.tox`, `.ruff_cache`, `htmlcov`, or build
  products.

## Next Work, In Order

1. **Run strengthened saved-reference equivalence.**
   Run `scripts/dev/run_saved_cwl_equivalence.py` in the prepared dev container
   with the new FITS residual checks. Save the generated JSON and Markdown
   reports outside git, then update `EQUIVALENCE_REPORT.md` with the new report
   path and FITS residual summary.

2. **Decide whether branch-vs-master equivalence is needed before scalability.**
   If the saved-reference run is green and the references still represent the
   scenarios we care about, proceed to the first scalability slice. If the
   references are stale, ambiguous, or fail for reasons that need a fresher
   baseline, build the branch-vs-master runner first.

3. **Take one low-risk image-cycle scalability slice.**
   Start with one natural boundary inside image-sector execution, such as
   source/model filtering or diagnostics after WSClean. Preserve output records,
   restart behavior, run names, worker payload serializability, and scientific
   products.

4. **Re-run scientific and performance gates after the slice.**
   Run focused tests, saved-reference equivalence, then the three-repetition
   `ci-benchmark` job. Compare against the 2026-07-04 baseline before taking a
   second slice.

5. **Refresh benchmark baseline documentation if the CI run is valid.**
   Commit only compact curated reports under
   `docs/source/development/benchmark_baselines/`. Keep bulky artifacts in CI
   artifacts or external storage.

6. **Resume test-suite maintainability cleanup.**
   Continue after the first benchmark-led scalability slice is guarded and
   measured. Keep `TESTING.md`, `.agents/testing_playbook.md`, and this plan in
   sync.

## Current Benchmarks

Use `docs/source/development/benchmark_baselines/2026-07-04-gitlab-60core.md`
and its companion summary JSON as the current compact baseline.

The regenerated 2026-07-04 report shows:

- Median wall time: `482.894 s`
- Dask report duration: `469.550 s`
- Dask task compute time: `247.350 s`
- Dask duration-minus-compute gap: `220.940 s`
- Dask task count: `12`
- `image_sector_task` compute dominates at about `239 s` total median
- Command profiling accounts for about `231 s` median
- Operation elapsed median is `281.881 s`
- Operation-minus-command gap is concentrated in image operations:
  about `11 s` for `image_1`, `9-10 s` for `image_2` and `image_3`, and
  `8.6 s` for `image_4`

Benchmark interpretation:

- The configured `2` worker / `60` thread shape is reaching Dask.
- The main opportunity is task granularity and orchestration visibility, not
  simply adding more CPU.
- Benchmark before and after any task-boundary, scheduler, dependency, or
  performance-sensitive execution change.

Local smoke benchmark command for smaller workstations:

```bash
scripts/dev/run_benchmark_baseline.py \
  --scenario ci-benchmark \
  --repetitions 1 \
  --local-dask-workers 1 \
  --cpus-per-task 4 \
  --max-threads 4
```

## Scientific Equivalence Track

Immediate task:

- Run the saved-reference matrix with strengthened FITS checks and update
  `EQUIVALENCE_REPORT.md`.

Keep:

- Product presence checks
- Operation order and `.done` marker checks
- Output-record shape checks
- h5parm dataset name, shape, and value checks
- Sky-model summaries
- Beam-table checks
- Exact text/region checks
- FITS aggregate checks
- FITS residual, finite-mask, WCS/header, pixel, and per-plane checks

Later:

- Add product-class-specific tolerances for dirty images, restored images,
  cubes, beam tables, and mosaics where needed.
- Add optional peak/source-aware checks for final science images.
- Add branch-vs-master equivalence runner if saved references are stale or
  insufficient:
  `scripts/dev/run_branch_equivalence.py --parset path/to/input.parset`.
- Keep bulky reference/current artifacts outside git; commit only compact
  curated reports and scenario manifests.

## First Scalability Slice

Start only after the strengthened saved-reference equivalence run is understood.

Preferred first slice:

- Split one image-sector post-WSClean step into a separate task boundary,
  likely source/model filtering or diagnostics.

Guardrails:

- Worker inputs stay plain serializable data, not `Field`, `Observation`,
  `Sector`, operation instances, open file handles, or subprocess state.
- Task names remain readable and include useful operation/sector identifiers.
- Output records and restart behavior remain unchanged.
- Existing image/calibrate product handoff semantics remain unchanged:
  `prepare_data_h5parm` is for DP3 pre-apply; `h5parm` is for imaging-time
  facet/DD application.
- External tools such as DP3, WSClean, IDG, and PyBDSF stay coarse commands
  unless a proven library-level integration exists.

Do not split a second image boundary until the first one has focused tests,
saved-equivalence evidence, and a benchmark comparison.

## Test Suite Track

Paused while equivalence and the first scalability slice are handled.

Keep doing as files are touched:

- Run `ruff check --fix --select I` and `ruff format` for Python code changes.
- Use pytest fixtures, fixture factories, `tmp_path`, `monkeypatch`,
  `pytest.raises`, `caplog`, and table-driven parametrization with readable ids.
- Keep tests clean code and living documentation.
- Avoid duplicated setup and broad helpers that hide the behavior under test.
- Add docstrings when the test name does not explain the scientific or runtime
  reason for the case.
- Replace placeholder `pass` tests with real assertions or delete redundant
  tests. Keep `pass` only for deliberate no-op fake methods.
- Keep slow, external-tool, internet/data-download, Prefect-server, and
  integration behavior behind clear marks or fixtures.

Near-term cleanup backlog:

- Continue deduplicating large execution/operation test setup.
- Split very large flow/operation test modules when touching them:
  `tests/execution/test_image_flow.py`,
  `tests/execution/test_calibrate_flow.py`,
  `tests/execution/test_pipeline_flow.py`,
  `tests/operations/test_image.py`, and
  `tests/operations/test_calibrate.py`.
- Keep `TESTING.md` and `.agents/testing_playbook.md` aligned with test layout,
  marks, integration requirements, and Prefect harness usage.
- Use collection and durations checks during cleanup:

```bash
python -m pytest -m "not integration" tests --collect-only -q
python -m pytest -m "not integration" tests --durations=30 --durations-min=0.25
```

## Runtime UX And Docs

Do after the first scalability slice has settled.

Next useful improvements:

- Expand preflight/dry-run output to show operation order, task groups,
  external tools, resource hints, expected outputs, and unsupported features.
- Improve messages for missing tools, unsupported container setup,
  Slurm/external-Dask mismatch, missing Dask scheduler, and MPI WSClean
  assumptions.
- Show resolved Prefect API mode, Dask scheduler mode, local worker count, and
  dashboard URLs.
- Add concise contributor docs for adding parset options, operation changes,
  external command helpers, execution-owned module adapters, and new flow task
  boundaries.
- Add a short debugging guide covering logs, `.done`, `.outputs.json`,
  command records, Prefect run names, Dask dashboards, and external-command
  stderr.

## Deferred Refactors

Do not split these modules for tidiness alone. Split them when changing
behavior or when a small extraction clearly reduces risk:

- `rapthor.execution.image.diagnostic_calculation`
- `rapthor.execution.image.flux_normalization`
- `rapthor.execution.calibrate.h5parm_combination`
- `rapthor.operations.calibrate.base`
- `rapthor.operations.image.base`
