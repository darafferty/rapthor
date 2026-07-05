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
- WSClean-based DD calibration prediction is ported into the Prefect/Dask
  calibration owner package: `use_wsclean_predict`, WSClean predict command
  construction, generated region/readpatches support, narrow-band model drawing,
  copied-MS model-column prediction, payload validation, defaults, docs, and
  focused tests are in place.

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

2. **Port master-only feature work that changes runtime behavior or products.**
   Compare against fetched `origin/master` at
   `17448437b78583f1eaf38112a524b2dbe5f34bb8` (`Generate residual
   visibilities`, 2026-07-01). Port the missing behavior into the current
   Prefect/Dask owner packages, preserving payload/output-record contracts and
   avoiding a return to CWL scripts. The remaining identified runtime feature
   slices have now been ported and the skipped/already-covered commit audit is
   recorded below.

3. **Lock down the branch-equivalence comparison contract.**
   Update the branch-equivalence runner/tests so expected legacy-vs-current
   differences are explicit. Product-presence changes should be either restored
   or documented as intentional; tolerances should distinguish restored image
   residuals, sparse model-image outliers, h5parm numeric drift, text/region
   ordering, and output-record path-only differences.

4. **Take one low-risk image-cycle scalability slice.**
   Start with one natural boundary inside image-sector execution, such as
   source/model filtering or diagnostics after WSClean. Preserve output records,
   restart behavior, run names, worker payload serializability, and scientific
   products.

5. **Re-run scientific and performance gates after the slice.**
   Run focused tests, saved-reference equivalence, then the three-repetition
   `ci-benchmark` job. Compare against the 2026-07-04 baseline before taking a
   second slice.

6. **Refresh benchmark baseline documentation if the CI run is valid.**
   Commit only compact curated reports under
   `docs/source/development/benchmark_baselines/`. Keep bulky artifacts in CI
   artifacts or external storage.

7. **Resume test-suite maintainability cleanup.**
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

## Master Feature Catch-Up Track

Comparison basis: `git fetch origin master` on 2026-07-04 confirmed
`origin/master` and local `master` at
`17448437b78583f1eaf38112a524b2dbe5f34bb8` (`Generate residual visibilities`,
2026-07-01). `git log HEAD..origin/master` shows 23 master commits since this
branch diverged. The items below are the master runtime/product features that
are missing or only partially represented in the current Prefect/Dask branch.

Porting standard: preserve the intended behavior from master, but adapt each
feature to this branch's current ownership boundaries and Prefect/Dask execution
contracts. Do not copy legacy script or CWL-era structure when a cleaner
execution-module implementation is available. While porting, clean up local
code paths when that reduces complexity without changing behavior, and improve
tests so they are maintainable, human-readable, and cover the branch-specific
contracts as well as the master behavior being restored.

Port these in order:

1. **Apply small low-risk master fixes first.** Done on 2026-07-04.
   The Measurement Set time-concatenation ordering fix from `bf4608ef` is
   ported into `rapthor/execution/concatenate/measurement_sets.py`, using
   Measurement Set `TIME` values rather than caller order for TAQL
   concatenation. The `modifystate.py` missing-directory guard from `e25b4a6a`
   is also ported so reset does not fail when optional output directories such
   as `visibilities/` do not exist. Focused pytest coverage exists for both.

2. **Restore astrometry-corrected image products.** Implemented on 2026-07-04.
   The `ebe35408` astrometry-correction behavior is now represented under
   `rapthor/execution/image/`: image-sector execution creates the Stokes-I
   `image-pb-ast.fits(.fz)` product after diagnostics, compression preserves it,
   image finalization records `I_image_file_true_sky_astcorr`, and mosaic inputs
   include the product when all sectors provide it. Focused unit, flow,
   finalizer, mosaic, Dask-boundary, and reference-fixture tests pass. Re-run the
   branch-vs-master equivalence scenario to confirm the previously missing
   `field-MFS-image-pb-ast` products are now present in full pipeline outputs.

3. **Add per-facet RMS diagnostics.** Implemented on 2026-07-04.
   The facet-selection helper from `eb1b6f2f` is ported into
   `rapthor/lib/fitsimage.py`, and `calculate_image_diagnostics` now adds
   `facets_rms` for valid facet-region files while omitting it for missing,
   `none`, or unreadable region inputs. Focused coverage exists for FITS facet
   selection, facet RMS statistics, missing/invalid region handling, and the
   diagnostics JSON contract.

4. **Port imaging model-data and residual-visibility products.** Implemented on 2026-07-04.
   The `save_residual_visibilities` parset/default/docs path is ported. Final
   image cycles now ask WSClean to retain/update `MODEL_DATA` when residual
   visibilities are requested, DP3 writes `*_resid.ms` as DATA minus
   MODEL_DATA, residual products are returned by the image flow, and image
   finalization publishes them under `visibilities/image_X/sector_Y`. Focused
   command-builder, payload, flow, finalizer, pipeline-flag, parset, and
   observation tests pass.

5. **Port WSClean-based prediction.** Implemented on 2026-07-04.
   The `use_wsclean_predict` parset/default/docs path is ported. DD calibration
   preparation now creates a prediction region, reads patch names from it, copies
   each input MS before adding model columns, draws narrow-band WSClean model
   images, runs WSClean `-predict` per patch/frequency slice, and wires the
   resulting model columns into DP3 solve commands without restoring the legacy
   `rapthor/scripts/wsclean_predict.py` glue. Focused command-builder, payload,
   flow, operation, parset, and field tests pass.

6. **Align normalization and parallel-gridding semantics with master.** Implemented on 2026-07-05.
   The `fc79ef7f`, `3e4eca19`, `38abdb92`, and `01a81e11` behavior is now
   represented in the Prefect/Dask implementation. `do_normalize` outside the
   first cycle logs a warning instead of raising, flux-density normalization
   handles empty/no-match products more robustly, WSClean parallel gridding is
   modelled as task groups via `parallel_gridding_tasks`, no-DDE and MPI facet
   WSClean commands keep `-parallel-gridding`, and shared facet read/write is
   enabled only when the actual patch/facet count supports it. The LSMTool
   dependency was advanced to `176ef008534bdd929e58c57b00c0a60e3445ad68`; Rapthor
   call sites were updated for the current `read_skymodel(..., wcs=...)` API.

7. **Record what is intentionally already covered or not relevant.** Done on 2026-07-05.
   The remaining commits in `HEAD..origin/master` were audited against
   `origin/master` at `17448437b78583f1eaf38112a524b2dbe5f34bb8`.

   | Commit | Decision |
   | --- | --- |
   | `bf4608ef`, `e25b4a6a`, `eb1b6f2f`, `ebe35408`, `d90786e8`, `971a2b25`, `e8867abd`, `17448437` | Runtime/product behavior is ported in tasks 1-5 above. |
   | `fc79ef7f`, `3e4eca19`, `38abdb92`, `01a81e11` | Normalization, parallel-gridding, and shared-facet semantics are ported in task 6. |
   | `908f83c9` | Superseded by the migrated diagnostic module. Missing/`none` facet-region handling is covered by `compute_facet_rms_noise` tests, so the legacy script patch does not need a separate port. |
   | `37f6fc06`, `0e163498` | Already covered by this branch's Ubuntu 24.04 CI images and libdeflate runtime/build dependencies. The dev container may remain independently tuned for local development. |
   | `c853f707` | Covered or superseded. CI integration splitting/thread caps are represented by this branch's current CI/tox setup, `test_do_normalize` coverage exists here, and the old LSMTool pin is superseded by updating to LSMTool `176ef008534bdd929e58c57b00c0a60e3445ad68` plus current `read_skymodel(..., wcs=...)` call sites. |
   | `f826c262`, `13e3bd8e` | Formatting-only churn. No behavioral port required; formatting is enforced by the repository ruff workflow. |
   | `5f59fe87`, `6c58212b`, `0c422261`, `14377a61`, `4cc0b732` | Test-structure cleanup from master overlaps with this branch's ongoing test maintainability track. Do not copy it wholesale while larger Prefect/Dask-specific fixtures and helpers are still stabilizing; harvest useful patterns as part of the paused test-suite cleanup. |

   No further master-only runtime feature slice is currently known. Re-run the
   phase-only branch-equivalence scenario after these feature ports and update
   `EQUIVALENCE_REPORT.md` with the remaining intentional differences.

After each feature slice, run the focused unit tests for the touched owner
package, then `ruff check --fix --select I`, `ruff format`, and the relevant
saved/branch equivalence checks in the prepared dev container. Re-run the
phase-only branch-equivalence scenario after tasks 2-6 because those tasks
change the exact product set compared with master.

## First Scalability Slice

Start only after the branch-vs-master multi-cycle equivalence runner exists,
the benchmark/default-like scenario is understood, and the master feature
catch-up track above has either been implemented or explicitly deferred.

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
