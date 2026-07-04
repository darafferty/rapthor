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
  sparse-outlier residual gating, JSON product statistics, and Markdown report
  output.
- The strengthened saved-reference matrix passed on 2026-07-04, with the
  current report recorded in `EQUIVALENCE_REPORT.md`.
- Branch-vs-master equivalence runner scaffolding is in place at
  `scripts/dev/run_branch_equivalence.py`: it accepts explicitly prepared
  base/current parsets, can create a base-ref worktree plus virtual environment
  for `master` or a chosen commit, runs each branch, records command logs and
  manifests, and reuses the strengthened product comparison checks.

Known caveats:

- Use the prepared dev-container Python environment for tests, formatting,
  integration checks, equivalence checks, and demo/benchmark runs. Local tox
  environments may fail to build astronomy dependencies.
- Keep large integration, equivalence, benchmark, and demo run roots outside
  git, preferably under `/tmp` or CI artifacts.
- Do not commit raw run directories, FITS/MS products, full Dask HTML reports,
  Prefect logs, `.tox`, `.ruff_cache`, `htmlcov`, or build products. Compact
  curated reports, manifests, and short command logs may be tracked under
  `docs/source/development/` when they explain an important result.

## Next Work, In Order

1. **Resolve the multi-cycle branch-vs-master equivalence findings.**
   Treat this as the next gate before performance-sensitive execution work.
   Two compact report bundles are now tracked under
   `docs/source/development/equivalence_runs/`: the default-like run that
   exposes the master slow-gain/amplitude issue, and the phase-only run that
   completes both branches with small restored-image residuals. Use these
   reports to decide the intended product contract for missing
   `field-MFS-image-pb-ast` products, output-record differences, sparse model
   image differences, and phase h5parm tolerances.

2. **Lock down the branch-equivalence comparison contract.**
   Update the branch-equivalence runner/tests so expected legacy-vs-current
   differences are explicit. Product-presence changes should be either restored
   or documented as intentional; tolerances should distinguish restored image
   residuals, sparse model-image outliers, h5parm numeric drift, text/region
   ordering, and output-record path-only differences.

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

Current status:

- The saved-reference matrix passed with strengthened FITS checks on
  2026-07-04. The current report path and residual summary are recorded in
  `EQUIVALENCE_REPORT.md`.
- `scripts/dev/run_branch_equivalence.py` now compares arbitrary explicitly
  prepared base/current parsets across `master` and the current branch. It does
  not translate parsets or strategies; the input files are the scientific
  contract. Use `--prepare-only` to validate/report the chosen files before
  launching expensive runs. Prefer absolute paths inside these parsets for
  shared data and strategy files, because each side is run from its own branch
  checkout. Use `--setup-base-env` to create or reuse a virtualenv for the base
  ref so `master` or a chosen commit runs with its own Python dependencies
  while the current branch runs normally.
- The first manual `benchmark-default-like` branch-vs-master run used
  `master` commit `17448437b78583f1eaf38112a524b2dbe5f34bb8` as the reference.
  Both branches returned `0`, but the strengthened product comparison failed.
  The compact report, manifest, and command logs are tracked under
  `docs/source/development/equivalence_runs/2026-07-04-default-like-master-ref/`.
  The key finding is that master logs a `combine_h5parms.py ...
  p1p2a2_diagonal` broadcasting error during the slow-gain path but still
  completes, leaving the active facet h5parm phase-only.
- The manual `benchmark-phase-only` branch-vs-master run also used `master` as
  the reference. Both branches returned `0`; strict comparison failed, but the
  restored dirty/image/residual FITS deltas are small, the h5parm phase delta is
  about `1.54e-05`, and the remaining failures are mostly missing
  `field-MFS-image-pb-ast` products, sparse model-image differences,
  source-catalog/facet-region text differences, and output-record summaries.
  The compact report, manifest, and command logs are tracked under
  `docs/source/development/equivalence_runs/2026-07-04-phase-only-master-ref/`.

Immediate task:

- Turn the two 2026-07-04 branch-equivalence reports into actionable contract
  decisions:

  - compare calibrate/image output records and classify differences as
    path-only, product-presence, or semantic
  - decide whether missing `field-MFS-image-pb-ast` products are intentional in
    the current branch or should be restored/configured
  - decide whether sparse `field-MFS-model-pb` differences should be compared
    with model-specific sparse-outlier tolerances or investigated as a real
    deconvolution/model-selection change
  - add h5parm phase tolerances appropriate for the observed phase-only delta,
    while keeping shape, axes, soltab names, and finite values strict
  - handle source-catalog/facet-region text differences with deterministic
    ordering or semantic comparison where possible
  - decide whether the slow-gain default-like reference should remain a
    documented master bug/legacy limitation, or whether the master checkout
    should be patched only for an intended-amplitude reference run
  - update `scripts/dev/run_branch_equivalence.py`,
    `tests/execution/test_branch_equivalence.py`, `EQUIVALENCE_REPORT.md`, and
    the tracked report README with the agreed comparison contract

For a workstation smoke check or user-supplied data, use `--prepare-only` first
to confirm the exact base/current parsets and work directories that will be
reported. The current benchmark strategy intentionally exercises DI-only phase
and full-Jones cycles that `master` cannot represent exactly; use explicit
DD-only/default-like branch-equivalence strategies for scientific
branch-vs-master gates.

For a specific base commit, pass `--base-ref <commit-ish>`. By default the base
checkout is placed under the run root at `checkouts/base` and the venv is
created at `checkouts/base/.venv`. Use `--base-venv` to reuse a different
environment, `--base-install-spec '.[dev]'` if development extras are needed,
`--base-system-site-packages` in prepared containers that provide compiled
astronomy packages globally, and `--reinstall-base-env` when dependencies should
be refreshed.

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
- Keep bulky reference/current artifacts outside git; commit only compact
  curated reports and scenario manifests.

## First Scalability Slice

Start only after the branch-vs-master multi-cycle equivalence runner exists and
the benchmark/default-like scenario is understood.

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
