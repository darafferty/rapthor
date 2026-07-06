# Rapthor Architecture Refactor Plan

Status snapshot: 2026-07-05.

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
  manifests, reuses the strengthened product comparison checks, records image
  diagnostic deltas, and generates compact side-by-side image/solution visual
  comparisons.
- The full integration suite passed in the prepared dev container on
  2026-07-05: `28 passed, 1 skipped, 1 xfailed`.
- WSClean-based DD calibration prediction is ported into the Prefect/Dask
  calibration owner package: `use_wsclean_predict`, WSClean predict command
  construction, generated region/readpatches support, narrow-band model drawing,
  copied-MS model-column prediction, payload validation, defaults, docs, and
  focused tests are in place.
- The flexible calibration initial-solution contract is implemented and
  documented: matching previous-cycle same-mode/same-solve products may seed
  later solves only as optimizer seeds, DD seeds require direction
  compatibility, and preapply/imaging-time h5parm use remains current-cycle
  guarded.
- DI full-Jones collection now matches the legacy master post-processing step:
  collected full-Jones gains are amplitude-normalized before plotting/finalizer
  handoff, without slow-gain flagging or smoothing.

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

1. **Turn classified DD plus DI full-Jones deltas into comparison rules.**
   The focused 2026-07-06 normalized rerun removed the strict h5parm failure
   and shrank the restored-image residual from about `1.025e-02` max absolute
   delta to about `2.486e-05`. The branch-equivalence report now classifies the
   remaining items as 4 small image residuals, 1 sparse model-image residual,
   23 PyBDSF diagnostic catalog columns, 1 DS9 region text-formatting
   difference, and 2 legacy output-record metadata warnings. Keep h5parm
   structure, product presence, operation order, source count, and primary
   catalog values strict. Next, implement semantic DS9 region comparison and
   decide whether to run a fresh three-repeat normalized full-Jones envelope to
   derive numeric tolerances for the image/PyBDSF repeatability candidates.

2. **Use the repeatability envelope to classify remaining scientific deltas.**
   The fixed-`facet_layout` and DD phase plus DI full-Jones repeatability
   envelopes are now tracked under `docs/source/development/equivalence_runs/`.
   Use these data before changing tolerances: fixed-facet image differences are
   repeatability-bounded, while the existing DD-plus-DI full-Jones envelope is
   pre-normalization evidence and should only be refreshed if the normalized
   focused rerun is not enough to classify the remaining residuals.
   Run a phase-only DD repeatability envelope only if comparison-rule changes
   need additional multi-cycle phase-only evidence.

3. **Keep flexible-strategy carry-forward explicit.**
   The current policy is no silent carry-over after a new calibration step:
   imaging and preapply use current-cycle products, while previous-cycle
   products may only seed matching solves or be reused by an explicit image-only
   cycle. Keep tests and docs aligned with this policy.

4. **Add a risk-based option equivalence matrix.**
   After the core repeatability envelope is available, add a small set of
   option-specific equivalence scenarios rather than a full combinatorial
   sweep. Prioritize normalization, WSClean predict versus image-based predict,
   BDA/averaging behavior, and screens where the target environment supports
   them. Keep each scenario to one meaningful option family so failures remain
   attributable.

5. **Tighten the branch-equivalence comparison contract.**
   Update `scripts/dev/run_branch_equivalence.py`,
   `tests/execution/test_branch_equivalence.py`, `EQUIVALENCE_REPORT.md`, and
   tracked report docs so expected legacy-vs-current differences are explicit.
   Tolerances and semantic comparisons should distinguish output-record
   metadata shape, h5parm phase drift, sparse model-image outliers, text/region
   ordering, and restored-image residuals while keeping operation order,
   product presence, shapes, axes, finite values, and soltab names strict.

6. **Re-run the full scientific gate.**
   Run focused tests, the strengthened saved-reference matrix, the essential
   branch-vs-master matrix, and the full integration suite in the prepared dev
   container. Update `EQUIVALENCE_REPORT.md` and compact report bundles with
   the final interpretation before taking performance-sensitive work.

7. **Take one low-risk image-cycle scalability slice.**
   Start with one natural boundary inside image-sector execution, such as
   source/model filtering or diagnostics after WSClean. Preserve output records,
   restart behavior, run names, worker payload serializability, and scientific
   products.

8. **Re-run scientific and performance gates after the slice.**
   Run focused tests, saved-reference equivalence, then the three-repetition
   `ci-benchmark` job. Compare against the 2026-07-04 baseline before taking a
   second slice.

9. **Refresh benchmark baseline documentation if the CI run is valid.**
   Commit only compact curated reports under
   `docs/source/development/benchmark_baselines/`. Keep bulky artifacts in CI
   artifacts or external storage.

10. **Resume test-suite maintainability cleanup.**
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
- The 2026-07-05 phase-only rerun used the same `master` reference after the
  current branch had caught up with the missing image-product behavior. Both
  branches returned `0`; strict comparison still failed. The rerun includes 32
  image diagnostic deltas and 20 side-by-side visual comparisons under
  `docs/source/development/equivalence_runs/2026-07-05-phase-only-master-ref/`.
  `field-MFS-image-pb-ast` products are now present on both branches. Top-level
  diagnostics are close, with matching source counts and largest relative
  deltas around `0.23%`, but h5 phase/source differences grow by cycle:
  exact in cycle 1, about `2.18e-07` in cycle 2, about `1.60e-03` in cycle 3,
  and about `6.12` plus source-coordinate differences in cycle 4.
- Investigation on 2026-07-05 found the first actionable semantic difference:
  master passes previous-cycle DD fast-phase h5parms into later calibration
  cycles as `fast_initialsolutions_h5parm`, while this branch's migrated
  calibration adapter treats those files as stale because their recorded cycle
  number is lower than the current calibration index. The current logs show
  explicit "Ignoring DD fast-phase/medium-phase h5parm ... produced in cycle
  N" warnings in cycles 2-4. Saved pipeline inputs confirm master provides
  previous-cycle fast-phase initial solutions for cycles 2-4, while current
  inputs set `solve1_initialsolutions_h5parm = None`. The phase h5parms are
  exact in cycle 1, nearly exact in cycle 2, then diverge cumulatively, which
  is consistent with different solve initialization rather than a parallel
  gridding/resource issue.

Initial-solution alignment status:

- Updated on 2026-07-05: calibration solve initialization now has a separate
  resolver from preapply products. Current-cycle same-solve products may seed
  matching solves. Previous-cycle same-mode/same-solve products may seed
  matching solves only as optimizer seeds when cycle and product role are
  valid; DD products additionally require compatible directions via fixed
  `facet_layout`, matching h5parm directions, or a future explicit remap/filter
  step. Future-cycle products are rejected. The strict current-cycle guard
  remains for calibration preapply products and imaging-time h5parm application.
- Focused operation and execution tests cover current-cycle, previous-cycle,
  future-cycle, phase-only DD, slow-gain DD, DI, wrong-mode, DD direction
  matching, fixed-`facet_layout` carry-over, and payload/command propagation
  cases.
- The rerun tracked under
  `docs/source/development/equivalence_runs/2026-07-05-phase-only-initial-solutions-master-ref/`
  confirms both branches pass previous-cycle DD fast-phase initial solutions in
  cycles 2-4 and leave phase-only medium/slow initial-solution slots unset.
  Re-run this scenario after the flexible carry-over contract is fully
  validated, because the current branch now intentionally allows matching
  phase-only medium seeds when directions are compatible.
- The first essential branch-vs-master mixed-mode scenario is tracked under
  `docs/source/development/equivalence_runs/2026-07-05-dd-phase-plus-di-fulljones-master-ref/`.
  Both branches return `0` for a single DD fast+medium phase-only cycle followed
  by DI full-Jones. Strict comparison still fails, but operation order, product
  presence, source counts, and high-level diagnostics are close; the largest
  image-diagnostic relative deltas are about `0.24%`, and the full-Jones
  amplitude h5parm delta is about `1.14e-03`.
- Follow-up investigation found that the DP3 solve commands, WSClean imaging
  commands, image preapply payloads, and full-Jones applycal settings match
  between branches. The systematic amplitude offset comes from post-processing:
  legacy master runs `process_gains.py` on the collected full-Jones h5parm with
  normalization enabled and flagging/smoothing disabled, while the current
  branch was previously handing the raw collected full-Jones gains to imaging.
  The current branch now ports that normalization behavior and focused tests
  cover the exact processing options. Rerun the focused full-Jones equivalence
  check before interpreting the old DD-plus-DI full-Jones repeatability envelope.
- The normalized focused rerun is tracked under
  `docs/source/development/equivalence_runs/2026-07-06-dd-phase-plus-di-fulljones-normalized-master-ref/`.
  Both branches return `0`; strict h5parm comparison now passes for all three
  h5parm products. The restored-image max absolute residual dropped from about
  `1.025e-02` to about `2.486e-05`, and the largest diagnostic relative delta
  is about `0.021%`. Remaining failures are small FITS image residuals, PyBDSF
  source-catalog uncertainty/shape columns, DS9 region text formatting, and
  legacy output-record metadata shape.
- The DI multi-cycle carry-over scenario is tracked under
  `docs/source/development/equivalence_runs/2026-07-05-di-multicycle-carryover-master-ref/`.
  It uses two selfcal/image cycles of master-compatible DD fast+medium
  phase-only calibration followed by DI full-Jones. This exposed a
  current-branch bug where a previous full-Jones h5parm was passed as an
  optimizer seed with `solve1.initialsolutions.soltab=[phase000]`; DP3 rejected
  the cycle-2 DI solve. The current branch now uses
  `[amplitude000,phase000]` for full-Jones initial solutions, with focused
  command and operation tests. The rerun returns `0` on both branches, reaches
  `calibrate_2`, `calibrate_di_2`, and `image_2`, and confirms image-cycle
  application of the cycle-2 full-Jones h5parm. Strict comparison still fails:
  cycle-1 diagnostics are close, while cycle-2 image RMS diagnostics differ by
  about `9-10%`, consistent with compounded differences after master carries
  unsafe DD seeds across regrouped directions and current skips them.
- The mode-boundary scenarios are tracked under
  `docs/source/development/equivalence_runs/2026-07-05-di-then-dd-mode-boundary-master-ref/`
  and
  `docs/source/development/equivalence_runs/2026-07-05-dd-then-di-mode-boundary-master-ref/`.
  Both use fixed facets and both branches return `0`. The DI-to-DD scenario
  confirms a policy difference: master carries the cycle-1 DI full-Jones
  product into cycle-2 imaging even though cycle 2 has no DI solve, while the
  current branch keeps imaging-time full-Jones application current-cycle
  guarded. The paired DD-to-DI scenario confirms that both branches apply the
  cycle-2 full-Jones product when cycle 2 runs DI; the remaining differences
  are dominated by DD seed policy and full-Jones numeric drift.
- The fixed-`facet_layout` carry-over scenario is tracked under
  `docs/source/development/equivalence_runs/2026-07-05-fixed-facet-carryover-master-ref/`.
  Both branches return `0` and run `calibrate_2`. Master passes only the
  previous fast-phase seed into cycle 2, while the current branch passes
  compatible fast and medium phase seeds because the fixed facet layout proves
  DD direction compatibility. Strict comparison still fails, as expected for
  this intentional flexible-strategy difference.
- The changing-facet carry-over scenario is tracked under
  `docs/source/development/equivalence_runs/2026-07-05-changing-facet-carryover-master-ref/`.
  Both branches return `0` and run `calibrate_2` after cycle 2 changes from
  five DD directions to three. Master still passes the previous fast-phase
  seed, while the current branch skips all previous-cycle DD initial-solution
  h5parms and logs the direction mismatch. Strict comparison still fails, but
  the safety behavior is the intended current-branch flexible-strategy
  contract.
- The slow-gain/default-like scenario is tracked under
  `docs/source/development/equivalence_runs/2026-07-05-slow-gain-default-like-master-ref/`.
  Both branches return `0` for a one-cycle calibration-only DD solve order
  equivalent to fast phase, medium phase, slow gains, and post-slow medium
  phase. Both produce standalone slow-gain and medium2 h5parms. Master still
  logs the `combine_h5parms` broadcast error and its final `field-solutions.h5`
  lacks `sol000/amplitude000`, while the current branch preserves the slow-gain
  amplitude soltab in the final combined h5parm.

Possible bugs on the master branch to investigate:

- Slow-gain h5parm combination can fail while the overall run still returns
  success. In the 2026-07-04 default-like branch-vs-master reference run,
  master logged a `combine_h5parms.py ... p1p2a2_diagonal` broadcasting error
  during the slow-gain path, but the CWL step/run completed and the active
  facet h5parm remained phase-only. The 2026-07-05 one-cycle slow-gain
  default-like run reproduced the issue in a tighter calibration-only case:
  master produced standalone slow-gain products but its final
  `field-solutions.h5` lacked `sol000/amplitude000`, while the current branch
  preserved it. Verify whether master should fail loudly, whether
  `p1p2a2_diagonal` should handle the product shapes, or whether this should
  remain a documented master reference limitation.
- Previous-cycle DD initial-solution h5parms are carried across calibration
  patch/facet changes without direction compatibility checks. In the 2026-07-05
  phase-only master reference run, cycle 1 used `[Patch_rich_*]`, cycle 2 used
  `[Patch_0..4]`, and cycle 4 used ten sector-specific patch names, but master
  still passed the previous fast-phase h5parm as `fast_initialsolutions_h5parm`.
  The 2026-07-05 changing-facet carry-over run reproduced the issue in a
  tighter two-cycle case: cycle 2 used only `Patch_2..4`, but master still
  passed the previous five-direction fast h5parm. Verify whether DP3 safely
  ignores/remaps unmatched directions or whether this is a master bug; the
  current branch now rejects this unsafe carry-over unless direction
  compatibility is proven.
- Master carries a previous-cycle DI full-Jones product into later imaging when
  the later cycle does not run a DI solve. The 2026-07-05 DI-to-DD
  mode-boundary run shows master `image_2` applying the cycle-1
  `fulljones-solutions.h5`, while the current branch intentionally leaves
  `fulljones_h5parm` unset for `image_2` because the active cycle is DD-only.
  Treat this as legacy implicit-state behavior that should remain documented
  but should not be copied silently into the flexible strategy.

Remaining equivalence tasks, in order:

1. **Use branch repeatability controls before tuning tolerances.**
   The essential branch-vs-master scenario matrix now has compact tracked
   reports for phase-only DD, DD plus DI full-Jones, fixed-facet DD carry-over,
   changing-facet DD carry-over, slow-gain/default-like behavior, DI
   multi-cycle carry-over, and both DI/DD mode-boundary directions.
   `scripts/dev/run_branch_equivalence.py` now supports
   `--repeatability-repetitions N`, which creates generated per-repetition
   parsets with unique clean work directories, writes pair reports, and writes
   aggregate `repeatability-summary.json` and `repeatability-summary.md` files.
   Two successful envelopes are tracked so far:
   `docs/source/development/equivalence_runs/2026-07-05-fixed-facet-repeatability-master-ref/`
   and
   `docs/source/development/equivalence_runs/2026-07-05-dd-phase-plus-di-fulljones-repeatability-master-ref/`.
   The fixed-facet envelope shows aggregate image differences inside
   same-branch scatter. The DD plus DI full-Jones envelope shows stable
   same-branch master repeats but was generated before current-branch
   full-Jones gain normalization was ported. The normalized focused rerun
   removes the h5parm failure and leaves small image/catalog residuals; refresh
   the repeatability envelope only if those remaining residuals need final
   tolerance evidence.
   Use short `/tmp` paths for `--run-root`, `--repeatability-work-root`, and
   any master checkout/venv paths when the master branch will run imaging. The
   legacy master CWL/Toil image filter path runs PyBDSF multiprocessing from
   scratch directories and can fail with `OSError: AF_UNIX path too long` when
   the generated paths are too deep.
   Set up a repeatability run directory that contains frozen input snapshots
   and unique clean work directories for each repetition, for example
   `/tmp/r<scenario>` for summaries and `/tmp/w<scenario>` for work products.
   Do not reuse `.done` markers or previous pipeline products. Keep resource settings,
   thread counts, parsets, strategy files, input data, external-tool versions,
   base commit, and current commit fixed across repetitions. Use the production
   thread/resource settings that reviewers will care about; optionally add a
   separate single-thread smoke repeat only when isolating nondeterminism.

   Recommended comparison layout:

   - Run three `master` repeats and compare all three master-master pairs.
   - Run three current-branch repeats and compare all three current-current
     pairs.
   - Compare all nine master-current pairs, not only matched repeat numbers.
   - Store compact per-pair JSON/Markdown summaries and one aggregate summary
     under `docs/source/development/equivalence_runs/`; keep raw FITS/MS/log
     products outside git.
   - Aggregate by product class: h5parm datasets, restored/dirty/model FITS
     residual metrics, image diagnostics, source-catalog columns, sky-model
     summaries, and text/region products.

   Use the data to classify each difference:

   - **Strict contract:** product presence, operation order, h5parm axes/shapes,
     soltab names, finite masks, and source counts must not vary within branch
     or across equivalent branches.
   - **Repeatability-bounded:** image residuals, h5parm numeric values,
     source-catalog flux/error/rms columns, and image diagnostics may use
     product-specific tolerances derived from the maximum or high percentile of
     same-branch scatter, with a small documented safety factor.
   - **Systematic branch difference:** a branch-vs-master delta that is
     consistently larger than same-branch scatter needs a scientific
     explanation, a bug fix, or an explicit report label as intentional
     flexible-strategy behavior.

2. **Document and enforce carry-forward policy for flexible strategy products.**
   The policy decision is to avoid silent carry-over after a new calibration
   step. A previous-cycle product may seed a matching solve when the product
   role and, for DD, directions are compatible. It may be reused by an explicit
   image-only cycle. It must not be applied during imaging after a new
   calibration step unless that product was part of the current cycle's
   calibration state. Current-cycle full-Jones imaging application is guarded by
   focused image-operation tests; keep DD direction compatibility checks strict.

3. **Add option-specific equivalence scenarios.**
   After the repeatability envelope exists, build a small risk-based option
   matrix. Do not combine many options in one run; each scenario should toggle
   one meaningful behavior against a stable baseline, preferably with fixed
   facets and short demo data.

   Recommended priority:

   - **Normalization:** high scientific impact and already covered by saved
     references; add a branch-vs-master scenario if master can represent the
     configuration cleanly.
   - **Prediction path:** run separate scenarios for image-based predict and
     WSClean predict, because WSClean predict was recently ported and failures
     should clearly point to model-column/predict behavior.
   - **BDA/averaging:** add one focused scenario for BDA or averaging-related
     options because they change calibration/imaging inputs even when solve
     logic is unchanged.
   - **Screens/IDGCal/hybrid screens:** keep as target-environment checks unless
     the prepared dev container can run them reliably; record skipped/blocked
     status explicitly.
   - **Other imaging options:** keep cubes, QUV/full-Stokes, astrometry,
     photometry, and peeling primarily in saved-reference or integration
     coverage unless a reviewer needs branch-vs-master parity for one of them.

   Use the option matrix after the core tolerance/repeatability work, not as a
   replacement for it. When an option scenario fails, decide whether the delta
   is within the repeatability envelope, a master limitation, or a current
   branch regression.

4. **Comparison-rule cleanup.**
   Classify output-record summary differences as metadata-shape differences or
   real product-record differences. Add h5parm numeric statistics/tolerances
   for accepted phase-only drift while keeping solset/soltab names, axes,
   shapes, finite values, and source tables strict. Decide whether sparse
   `field-MFS-model-pb` differences need model-specific sparse-outlier
   tolerances or deeper deconvolution/model-selection investigation. Handle
   source-catalog and facet-region text differences with deterministic ordering
   or semantic comparison where possible.

5. **Master-reference decisions.**
   Decide whether the slow-gain default-like reference should remain a
   documented master bug/legacy limitation, whether the master checkout should
   be patched for an intended-amplitude reference run, and how any intentional
   current-vs-master divergence should be labelled in reports.

6. **Tooling and reviewer smoke check.**
   Update `scripts/dev/run_branch_equivalence.py`,
   `tests/execution/test_branch_equivalence.py`, `EQUIVALENCE_REPORT.md`, and
   tracked report docs as the comparison contract is tightened. Add a
   user-supplied parset smoke scenario: run `--prepare-only` to verify reported
   base/current parsets, strategy files, and work directories, then run one
   documented real comparison so reviewers can reuse the workflow with their
   own datasets.

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
