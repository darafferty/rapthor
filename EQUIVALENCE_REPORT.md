# Rapthor Equivalence Decision Report

Latest status scan: 2026-08-20

Science contract:
`docs/source/development/science_equivalence_contract.rst`

Performance contract:
`docs/source/development/performance_equivalence_contract.rst`

Detailed science evidence:
`docs/source/development/science_equivalence_runs/`

Detailed performance evidence:
`docs/source/development/performance_equivalence_runs/`

## Executive Decision

**Recommendation: accept the refactored Prefect/Dask pipeline as
scientifically sound for the tested contract, and accept the current
repeatability-aware performance evidence for the tested phase-only and
DD/full-Jones scenarios. Continue targeted scalability optimisation and
operational readiness checks before making the final switch from `master`.**

The current evidence answers two related but separate questions:

- **Science equivalence:** the refactored pipeline preserves the tested
  self-calibration product contract.
- **Performance equivalence:** repeatability-aware phase-only and DD/full-Jones
  performance gates pass and show the current branch faster than `master` for
  both tested scenarios.

This report is the reviewer-facing summary. The dated folders under
`docs/source/development/science_equivalence_runs/` and
`docs/source/development/performance_equivalence_runs/` are the audit trail for
individual runs.

## Current Gate Status

| Gate | Latest Result | Evidence | Decision |
| --- | --- | --- | --- |
| Science equivalence | **Pass / accepted with classified repeatability differences** | `runs/equivalence-gate-20260820-august-sync/`, `2026-08-04-august-master-sync`, and earlier DD/full-Jones evidence | Exact `043c15d4` master/current repeatability gates pass for generated initial sky-model grouping with and without imaging BDA, and for the default frequency-BDA path. Earlier old-reference normalization and peeling warnings remain documented external-tool baseline differences. |
| Performance equivalence | **Pass for phase-only core and DD/full-Jones** | `2026-07-11-phase-only-core-repeatability-gate`, `2026-07-12-dd-phase-plus-di-fulljones-repeatability-gate` | Performance equivalence is established for the current optimisation phase; continue targeted benchmarking for new scalability changes. |

The manual-testing discrepancy around generated initial sky models is now
covered. Paired unaveraged and production-BDA scenarios generate the initial
image and catalog, filter and regroup to the requested one-direction/1 Jy
contract, and continue through calibration. Each branch ran three times. The
no-imaging-BDA control passes all 9/9 cross-branch comparisons as
repeatability-bounded; the production-BDA case has two strict passes and seven
repeatability-bounded comparisons. Source identities, patch membership,
positions, fluxes, spectral terms, shapes, initial image diagnostics, FITS
products, catalogs, and h5parm solutions are included in the decision.
All six unaveraged runs raise the requested 1.00 Jy grouping threshold to
3.17 Jy on both branches; all six production-BDA runs raise it to 4.41 Jy on
both branches. The original branch-dependent target-flux symptom is therefore
not reproduced when configuration and BDA settings are matched explicitly.

## What Was Being Decided

Rapthor was migrated from a CWL/Toil execution model to a Python
Prefect/Dask execution model. That migration changed the orchestration layer,
operation boundaries, internal code structure, runtime visibility, and task
scheduling.

The equivalence programme asks two decision questions:

1. **Science:** does the refactored pipeline still produce scientifically
   equivalent self-calibration outputs?
2. **Performance:** does the refactored pipeline run at least as well as
   `master`, once normal run-to-run scatter is measured?

For Rapthor, the science question means preserving the iterative loop:

- predict model visibilities from the sky model
- solve DI and/or DD calibration terms
- apply or use those calibration products during imaging
- make FITS images, h5parm solution files, sky models, catalogs, regions, and
  diagnostics
- repeat the cycle without silently reusing stale or incompatible solutions

The performance question means comparing `master` and the current branch with
repeated runs, same-branch repeatability envelopes, cross-branch product
comparisons, wall-clock timing, and operation-level timing.

## Latest Science Gate

Status: **accepted for the covered scientific contract**.

Confidence: **high for the tested LOFAR HBA self-calibration paths**.

Latest tracked science evidence:

- `runs/equivalence-gate-20260820-august-sync/` (local compact report archive;
  rerunnable inputs are committed under `tests/resources/equivalence/`)

- `docs/source/development/science_equivalence_runs/2026-08-04-august-master-sync/`
- `docs/source/development/science_equivalence_runs/2026-07-16-post-master-sync-saved-reference/`
- `docs/source/development/science_equivalence_runs/2026-07-16-post-master-sync-option-matrix/`
- `docs/source/development/science_equivalence_runs/2026-07-16-frequency-only-imaging-bda-current/`
- `docs/source/development/science_equivalence_runs/2026-07-11-post-task-split-saved-reference/`
- `docs/source/development/science_equivalence_runs/2026-07-11-post-task-split-option-matrix/`
- `docs/source/development/science_equivalence_runs/2026-07-06-dd-phase-plus-di-fulljones-normalized-repeatability-master-ref/`
- `docs/source/development/science_equivalence_runs/2026-07-06-dd-phase-plus-di-fulljones-normalized-master-ref/`
- `docs/source/development/science_equivalence_runs/2026-07-06-saved-reference-final-gate/`

The current branch passes the science gate because:

- the exact `043c15d4` generated-initial-sky-model control and production-BDA
  scenarios pass across three master and three current repetitions each,
  closing the target-flux grouping gap found during manual ICAL testing
- the exact `043c15d4` frequency-BDA scenario passes all 9/9 cross-branch
  pairs; current median runtime is `125.944 s` versus `298.744 s` for master
  in the verification container (`-57.842%`)
- strict OOM preflight integration coverage exits before any calibration
  operation starts, and the focused August-sync suites pass

- the August active saved-reference matrix has five strict passes; its
  normalization and peeling warnings are limited to three old-reference FITS
  image comparisons, while operation, product, structure, h5parm, and text
  checks pass
- controlled same-stack comparisons against exact August `master` commit
  `b307e769` pass strictly for the newly ported calibration-aware BDA frequency
  limits and for normalization, including the normalization frequency cube
- the old-reference failures and original tolerances remain preserved; no
  comparator was weakened and no saved product was refreshed to force a pass
- the old saved normalization cube difference is stable, concentrated at beam
  boundary pixels, and absent from a controlled same-stack `master`/current
  normalization comparison
- the post-sync WSClean prediction difference is explained by a master bug:
  master leaves two of eight channels unpredicted, while current uses complete
  end-exclusive channel ranges with direct unit and integration coverage
- frequency-only imaging BDA completes on current with a two-SPW DP3 output,
  WSClean reorder/facet-beam application, and a finite primary-beam image;
  documented master defects prevent a valid reference run for this path
- the DD phase plus DI full-Jones repeatability run passes for every
  base-base, current-current, and base-current pair after full-Jones gain
  normalization was aligned
- h5parm calibration solution structure, axes, directions, solution names, and
  metadata remain strict where they define scientific state
- remaining differences are either within measured repeatability, are
  non-scientific metadata/artifact differences, or are intentional improvements
  to unsafe implicit state handling in `master`

## Latest Performance Gate

Status: **phase-only core and DD/full-Jones performance equivalence pass**.

Latest tracked performance evidence:

- `docs/source/development/performance_equivalence_runs/2026-07-12-dd-phase-plus-di-fulljones-repeatability-gate.md`
- `docs/source/development/performance_equivalence_runs/2026-07-12-dd-phase-plus-di-fulljones-repeatability-gate.summary.json`
- `docs/source/development/performance_equivalence_runs/2026-07-11-phase-only-core-repeatability-gate.md`
- `docs/source/development/performance_equivalence_runs/2026-07-11-phase-only-core-repeatability-gate.summary.json`

DD phase plus DI full-Jones result:

- all six branch runs completed with return code `0`
- science/product validity passed; 4 of 9 `master`/current pairs were
  repeatability-bounded and the remaining cross-branch pairs passed directly
- current median runtime was `94.004 s`
- `master` median runtime was `151.183 s`
- current branch median runtime was `37.821%` faster than `master`
- parsed operation medians were faster on the current branch for calibration,
  prediction, imaging, and mosaic operations

Phase-only result:

- all six branch runs completed with return code `0`
- 9 of 9 `master`/current pairs were repeatability-bounded
- current median runtime was `303.160 s`
- `master` median runtime was `429.557 s`
- current branch median runtime was `29.425%` faster than `master`
- all parsed operation medians were faster on the current branch

Together, these results are strong evidence that the Prefect/Dask refactor does
not impose a runtime penalty for the tested core and mixed-calibration paths.
Future optimisation batches should continue to produce targeted benchmark
reports, but the performance-equivalence gate no longer blocks the current
scalability phase.

## Evidence Summary

| Evidence | Result | Why it matters |
| --- | --- | --- |
| August master-sync verification | Accepted with classified saved-reference warnings; two same-stack branch comparisons pass strictly | Verifies the newly ported BDA frequency limits and separates rebuilt-tool image drift from current-vs-master behavior without changing tolerances. |
| Post-master-sync saved-reference gate | Six strict passes; normalization dependency shift classified | Confirms the refactor remains stable after LSMTool/EveryBeam updates without hiding an old beam-boundary baseline change behind broader tolerances. |
| Post-master-sync option checks | Normalization passes; WSClean prediction divergence accepted as a master bug fix | Separates dependency effects from branch effects and proves current covers every requested prediction channel. |
| Frequency-only imaging BDA | Current-only pass | Closes a recently ported path for which master cannot provide a scientifically valid reference. |
| Post-task-split saved-reference gate | Pass | Confirms task-boundary changes preserved the saved scientific product contract for active scenarios. |
| Post-task-split option matrix | Pass for active rows | Confirms high-risk options still work after task splitting: normalization, DP3 prediction, WSClean prediction, and BDA/averaging. |
| DD phase plus DI full-Jones repeatability | Pass | Confirms the most important mixed calibration path is stable across repeated runs and branch comparisons. |
| Focused normalized full-Jones branch-vs-master run | Accepted with classified non-blocking differences | Confirms calibration solutions match after full-Jones gain normalization alignment. |
| Phase-only performance repeatability gate | Pass | Confirms current branch is faster than `master` for the phase-only core scenario while products remain repeatability-bounded. |
| DD/full-Jones performance repeatability gate | Pass | Confirms current branch is faster than `master` for the broader mixed DD phase plus DI full-Jones scenario while product differences remain valid. |
| Flexible carry-forward and mode-boundary scenarios | Accepted intentional differences | Confirms the refactor uses explicit, safer calibration-state rules rather than copying implicit master behavior. |

## What Is Strict

The gates keep strictness where strictness matters:

- operation order and required operation presence
- required product presence and product basenames
- FITS image/table structure, finite masks, and key WCS/header information
- h5parm solset and soltab names, axes, shapes, finite masks, source tables,
  and non-numeric datasets
- sky-model source identity, patch membership, positions, fluxes, spectral
  terms, shapes, patch positions, and source/patch counts
- region/facet products where they affect DD solution application
- source-catalog and image-diagnostic products
- finalizer-visible output records where downstream behavior depends on them
- run return codes, required logs, and required diagnostics for performance
  gates

Numeric FITS, h5parm, catalog, diagnostic, and runtime differences are accepted
only when they are inside measured same-branch repeatability envelopes or have a
specific scientific or operational explanation.

## Why Remaining Differences Are Accepted

The accepted differences fall into three categories.

**1. Repeatability-bounded numerical differences**

Small image residuals, source-catalog diagnostic columns, image diagnostics,
and sparse model-image statistics vary within measured same-branch
repeatability envelopes. These are not interpreted as scientific regressions.

**2. Non-scientific metadata or review artifacts**

Legacy CWL output-record checksums, file-size metadata, diagnostic plot
artifact names, generated preview PNGs, and minor text formatting differences do
not change the scientific state of the pipeline. They are useful for review,
but they are not the science contract.

**3. Intentional improvements to calibration-state handling**

The refactored pipeline is stricter about when calibration products may be
reused. Previous-cycle products may seed compatible later solves as optimizer
seeds, but they are not silently applied during imaging after a new calibration
step unless they belong to the active cycle's calibration state. For DD
solutions, previous-cycle products must also have compatible directions before
they can seed a later solve.

This is scientifically safer than blindly carrying solutions across changed
facets, regrouped directions, or mode boundaries.

## Important Current-Vs-Master Differences

These differences are accepted and should not be treated as regressions:

- The current branch blocks unsafe previous-cycle DD seed reuse when directions
  are not proven compatible.
- The current branch does not silently carry an old DI full-Jones solution into
  a later DD-only imaging step after a new DD calibration step.
- The current branch preserves slow-gain amplitude solutions in the final
  `field-solutions.h5`, rather than copying the legacy master behavior where a
  slow-gain h5parm-combination error can still produce a successful run with
  phase-only active solutions.
- The current branch interprets WSClean `-channel-range` endpoints correctly
  and covers every prediction channel. Master can leave the last channel of
  each legacy frequency chunk unpredicted.
- The current branch has leaner output records than the legacy CWL path; this
  is a metadata shape difference, not a change in the scientific products.
- The current branch exposes more Prefect task boundaries and runtime metrics;
  performance decisions therefore use repeatability-aware wall-clock and
  operation-level timing rather than task-count comparisons alone.

These changes make calibration state more explicit and reduce the risk of
using stale or incompatible solutions.

## Known Caveats

- Performance equivalence has been tested for phase-only core and DD/full-Jones
  paths. Continue to add targeted benchmark evidence when changing task
  granularity, resource policy, WSClean/PyBDSF behavior, or hidden scaling
  paths.
- The `screens` option-matrix row remains skipped until reliable IDGCal/screen
  support is available in the target environment.
- MPI WSClean, Slurm, and external-Dask behavior remain deployment checks
  rather than local science-gate blockers.
- Historical default-like and slow-gain branch-vs-master runs expose probable
  `master` limitations. They are preserved in the historical run log, but they
  should not be used as the desired scientific target without a separate
  decision.
- The historical normalization frequency-cube baseline predates EveryBeam
  0.8.3. Its small edge-concentrated shift is retained in the strict report and
  classified through repeatability plus a same-stack branch comparison; the
  global FITS tolerances were not widened.

## Decision Implication

On scientific grounds, the refactored pipeline is ready for benchmarking,
scalability work, and operational readiness checks.

On performance grounds, the first formal repeatability-aware gate is positive:
the current branch is faster than `master` for the phase-only core scenario
while branch-vs-branch products remain repeatability-bounded. Decision makers
should require the same style of evidence for DD/full-Jones before making a
final production replacement decision.

## Ongoing Governance

Documentation-only, report-only, preview-artifact-only, and refactor-only
changes may rely on this accepted evidence plus focused tests.

Rerun the relevant saved-reference, branch-vs-master, repeatability,
option-matrix, or performance gate after changes to:

- calibration strategy semantics
- DI/DD solve order, solution seeding, h5parm collection, or solution
  application
- prediction and subtraction behavior
- imaging preparation, WSClean commands, FITS products, cubes, mosaics, or
  source catalogs
- sky-model filtering, normalization, region/facet products, or diagnostics
- finalizer-visible output records or product locations
- task boundaries, scheduling, resource allocation, Dask worker shape, or
  performance-sensitive command execution

Every decision-relevant rerun should update this root report with the current
verdict and link to a dated detailed report. Keep raw Measurement Sets, FITS
products, h5parm files, full logs, and temporary run directories out of git.
