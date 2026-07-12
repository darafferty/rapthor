# Rapthor Equivalence Decision Report

Latest status scan: 2026-07-12

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
scientifically sound for the tested contract, and continue guarded performance
and scalability validation before making the final operational switch from
`master`.**

The current evidence answers two related but separate questions:

- **Science equivalence:** the refactored pipeline preserves the tested
  self-calibration product contract.
- **Performance equivalence:** the first repeatability-aware phase-only
  performance gate passes and shows the current branch faster than `master`,
  but broader performance evidence is still needed for DD/full-Jones paths.

This report is the reviewer-facing summary. The dated folders under
`docs/source/development/science_equivalence_runs/` and
`docs/source/development/performance_equivalence_runs/` are the audit trail for
individual runs.

## Current Gate Status

| Gate | Latest Result | Evidence | Decision |
| --- | --- | --- | --- |
| Science equivalence | **Pass / accepted** | `2026-07-11-post-task-split-saved-reference`, `2026-07-11-post-task-split-option-matrix`, and earlier DD/full-Jones repeatability evidence | Scientific contract is accepted for the covered LOFAR HBA self-calibration paths. |
| Performance equivalence | **Pass for phase-only core** | `2026-07-11-phase-only-core-repeatability-gate` | Initial performance gate passes; run DD/full-Jones before treating performance equivalence as broadly established. |

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

- `docs/source/development/science_equivalence_runs/2026-07-11-post-task-split-saved-reference/`
- `docs/source/development/science_equivalence_runs/2026-07-11-post-task-split-option-matrix/`
- `docs/source/development/science_equivalence_runs/2026-07-06-dd-phase-plus-di-fulljones-normalized-repeatability-master-ref/`
- `docs/source/development/science_equivalence_runs/2026-07-06-dd-phase-plus-di-fulljones-normalized-master-ref/`
- `docs/source/development/science_equivalence_runs/2026-07-06-saved-reference-final-gate/`

The current branch passes the science gate because:

- the post-task-split saved-reference gate passes for all active non-stale
  reference scenarios
- the post-task-split option matrix passes for normalization, DP3 image-based
  prediction, WSClean prediction, and BDA/averaging
- the DD phase plus DI full-Jones repeatability run passes for every
  base-base, current-current, and base-current pair after full-Jones gain
  normalization was aligned
- h5parm calibration solution structure, axes, directions, solution names, and
  metadata remain strict where they define scientific state
- remaining differences are either within measured repeatability, are
  non-scientific metadata/artifact differences, or are intentional improvements
  to unsafe implicit state handling in `master`

## Latest Performance Gate

Status: **phase-only core performance equivalence passes**.

Latest tracked performance evidence:

- `docs/source/development/performance_equivalence_runs/2026-07-11-phase-only-core-repeatability-gate.md`
- `docs/source/development/performance_equivalence_runs/2026-07-11-phase-only-core-repeatability-gate.summary.json`

Phase-only result:

- all six branch runs completed with return code `0`
- 9 of 9 `master`/current pairs were repeatability-bounded
- current median runtime was `303.160 s`
- `master` median runtime was `429.557 s`
- current branch median runtime was `29.425%` faster than `master`
- all parsed operation medians were faster on the current branch

This result is strong evidence that the Prefect/Dask refactor does not impose a
phase-only runtime penalty. It is not yet the final performance decision for all
scientific modes. The next performance gate should run the
`dd-phase-plus-di-fulljones` scenario with three repetitions per branch.

## Evidence Summary

| Evidence | Result | Why it matters |
| --- | --- | --- |
| Post-task-split saved-reference gate | Pass | Confirms task-boundary changes preserved the saved scientific product contract for active scenarios. |
| Post-task-split option matrix | Pass for active rows | Confirms high-risk options still work after task splitting: normalization, DP3 prediction, WSClean prediction, and BDA/averaging. |
| DD phase plus DI full-Jones repeatability | Pass | Confirms the most important mixed calibration path is stable across repeated runs and branch comparisons. |
| Focused normalized full-Jones branch-vs-master run | Accepted with classified non-blocking differences | Confirms calibration solutions match after full-Jones gain normalization alignment. |
| Phase-only performance repeatability gate | Pass | Confirms current branch is faster than `master` for the phase-only core scenario while products remain repeatability-bounded. |
| Flexible carry-forward and mode-boundary scenarios | Accepted intentional differences | Confirms the refactor uses explicit, safer calibration-state rules rather than copying implicit master behavior. |

## What Is Strict

The gates keep strictness where strictness matters:

- operation order and required operation presence
- required product presence and product basenames
- FITS image/table structure, finite masks, and key WCS/header information
- h5parm solset and soltab names, axes, shapes, finite masks, source tables,
  and non-numeric datasets
- sky-model source and patch counts
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
- The current branch has leaner output records than the legacy CWL path; this
  is a metadata shape difference, not a change in the scientific products.
- The current branch exposes more Prefect task boundaries and runtime metrics;
  performance decisions therefore use repeatability-aware wall-clock and
  operation-level timing rather than task-count comparisons alone.

These changes make calibration state more explicit and reduce the risk of
using stale or incompatible solutions.

## Known Caveats

- The performance gate has passed for the phase-only core scenario only. Run
  DD/full-Jones repeatability before claiming broad performance equivalence.
- The `screens` option-matrix row remains skipped until reliable IDGCal/screen
  support is available in the target environment.
- MPI WSClean, Slurm, and external-Dask behavior remain deployment checks
  rather than local science-gate blockers.
- Historical default-like and slow-gain branch-vs-master runs expose probable
  `master` limitations. They are preserved in the historical run log, but they
  should not be used as the desired scientific target without a separate
  decision.

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
