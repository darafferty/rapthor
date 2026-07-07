# Rapthor Science Equivalence Decision Report

Latest status scan: 2026-07-06

Method contract: `docs/source/development/science_equivalence_contract.rst`

Detailed historical run log:
`docs/source/development/equivalence_runs/science_gate_history.md`

## Executive Decision

**Recommendation: accept the refactored Prefect/Dask pipeline as
scientifically sound for the tested contract, and continue with guarded
scalability and performance work.**

The evidence supports switching scientific development attention from
"is the refactor scientifically valid?" to "does the refactor scale and perform
well enough to replace the current production branch?". Future changes that
touch scientific products must continue to pass the relevant science-equivalence
checks.

This report does not claim that every possible Rapthor configuration has been
exhaustively tested. It says that the core self-calibration behavior, key image
and solution products, and the highest-risk refactor paths have enough evidence
to accept the current branch for the covered scientific contract.

## What Was Being Decided

Rapthor was migrated from a CWL/Toil execution model to a Python
Prefect/Dask execution model. That migration changed the orchestration layer,
operation boundaries, and internal code structure. The science gate asks a
separate question from code quality or runtime performance:

**Does the refactored pipeline still produce scientifically equivalent
self-calibration outputs?**

For Rapthor, that means preserving the scientific behavior of the iterative
loop:

- predict model visibilities from the sky model
- solve DI and/or DD calibration terms
- apply or use those calibration products during imaging
- make FITS images, h5parm solution files, sky models, catalogs, regions, and
  diagnostics
- repeat the cycle without silently reusing stale or incompatible solutions

## Gate Verdict

Status: **accepted for the covered scientific contract**.

Confidence: **high for the tested LOFAR HBA self-calibration paths**.

The current branch passes the science gate because:

- the saved-reference matrix passes for all active non-stale reference
  scenarios
- the strongest DD phase plus DI full-Jones repeatability run passes for every
  base-base, current-current, and base-current pair
- all h5parm calibration solution products pass in the focused normalized
  full-Jones branch-vs-master check
- the first risk-based option matrix passes for flux-scale normalization,
  DP3 image-based prediction, WSClean prediction, and BDA/averaging
- the remaining differences are either within measured repeatability, are
  non-scientific metadata/artifact differences, or are intentional improvements
  to unsafe implicit state handling in `master`

## Evidence Summary

| Evidence | Result | Why it matters |
| --- | --- | --- |
| Saved-reference final gate | Pass | Confirms the refactored pipeline preserves the legacy product contract for active saved scenarios, including FITS, h5parm, sky-model, region, and text products. |
| DD phase plus DI full-Jones repeatability | Pass | Confirms the most important mixed calibration path is stable across repeated runs and branch comparisons. |
| Focused normalized full-Jones branch-vs-master run | Accepted with classified strict differences | Confirms calibration solutions match after the full-Jones gain normalization behavior was aligned; remaining differences are small image/catalog/text/metadata differences already explained by repeatability or non-scientific formatting. |
| Risk-based option matrix | Pass for active rows | Confirms high-risk user options still work: provided normalization sky models, DP3 image-based prediction, WSClean prediction, and BDA/averaging. |
| Flexible carry-forward and mode-boundary scenarios | Accepted intentional differences | Confirms the refactor uses explicit, safer calibration-state rules rather than copying implicit master behavior. |

Latest tracked evidence:

- `docs/source/development/equivalence_runs/2026-07-06-saved-reference-final-gate/`
- `docs/source/development/equivalence_runs/2026-07-06-dd-phase-plus-di-fulljones-normalized-repeatability-master-ref/`
- `docs/source/development/equivalence_runs/2026-07-06-dd-phase-plus-di-fulljones-normalized-master-ref/`
- `docs/source/development/equivalence_runs/2026-07-06-option-matrix/`
- `docs/source/development/equivalence_runs/2026-07-05-*carryover*`
- `docs/source/development/equivalence_runs/2026-07-05-*mode-boundary*`

## What Was Checked Strictly

The gate kept the scientific product contract strict where strictness matters:

- operation order and required operation presence
- required product presence and product basenames
- FITS image/table structure, finite masks, and key WCS/header information
- h5parm solset and soltab names, axes, shapes, finite masks, source tables,
  and non-numeric datasets
- sky-model source and patch counts
- region/facet products where they affect DD solution application
- source-catalog and image-diagnostic products
- finalizer-visible output records where downstream behavior depends on them

Numeric FITS, h5parm, catalog, and diagnostic differences are accepted only
when they are inside the measured same-branch repeatability envelope or have a
specific scientific explanation.

## Why Remaining Differences Are Accepted

The accepted differences fall into three categories.

**1. Repeatability-bounded numerical differences**

Small image residuals, source-catalog diagnostic columns, and image diagnostics
vary within the measured same-branch repeatability envelope. These are not
interpreted as scientific regressions.

**2. Non-scientific metadata or review artifacts**

Legacy CWL output-record checksums, file-size metadata, diagnostic plot artifact
names, generated preview PNGs, and minor text formatting differences do not
change the scientific state of the pipeline. They are useful for review, but
they are not the science contract.

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
- The current branch has leaner output records than the legacy CWL path; this is
  a metadata shape difference, not a change in the scientific products.

These changes make calibration state more explicit and reduce the risk of
using stale or incompatible solutions.

## Known Caveats

- The `screens` option-matrix row remains skipped until reliable IDGCal/screen
  support is available in the target environment.
- MPI WSClean, Slurm, and external-Dask behavior remain deployment checks
  rather than local science-gate blockers.
- Historical default-like and slow-gain branch-vs-master runs expose probable
  `master` limitations. They are preserved in the historical run log, but they
  should not be used as the desired scientific target without a separate
  decision.
- This report assesses scientific equivalence. Runtime speed and scalability
  are covered by the separate performance/scalability equivalence gate.

## Decision Implication

On scientific grounds, the refactored pipeline is ready for the next phase:
benchmarking, scalability work, and operational readiness checks.

Decision makers should not require a return to the `master` implementation on
the basis of the science-equivalence evidence currently available. The stronger
decision still needed is the performance/scalability decision: whether the
current branch runs and scales well enough to replace `master` operationally.

## Ongoing Governance

Documentation-only, report-only, preview-artifact-only, and refactor-only
changes may rely on this accepted evidence plus focused tests.

Rerun the relevant saved-reference, branch-vs-master, repeatability, or
option-matrix gate after changes to:

- calibration strategy semantics
- DI/DD solve order, solution seeding, h5parm collection, or solution
  application
- prediction and subtraction behavior
- imaging preparation, WSClean commands, FITS products, cubes, mosaics, or
  source catalogs
- sky-model filtering, normalization, region/facet products, or diagnostics
- finalizer-visible output records or product locations

Future evidence should remain compact and reviewable: commit Markdown reports,
JSON summaries, scenario manifests, input snapshots, and short command logs
when they explain a decision. Keep raw Measurement Sets, FITS products, h5parm
files, full logs, and temporary run directories out of git.
