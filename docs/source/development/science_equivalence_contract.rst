Science Equivalence Contract
============================

Purpose
-------

This contract defines how Rapthor decides whether a refactor, runtime
migration, or branch comparison preserves the scientific product contract. It
is the durable method behind the root-level ``EQUIVALENCE_REPORT.md`` status
report.

Science equivalence does not require byte-for-byte equality for every generated
file. It does require that the same scientific workflow is executed, that the
same required products are present, and that scientifically meaningful numeric
differences are either strict passes or bounded by measured repeatability.

Comparison Surfaces
-------------------

The science gate compares products that scientists and downstream workflows use
or inspect:

* operation order and operation presence
* ``.done`` markers and restart/output-record state
* finalizer-visible output records and product locations
* FITS images, cubes, mosaics, beams, and tables
* h5parm solution files, solsets, soltabs, axes, directions, and values
* sky models, patch/source counts, and text products
* region files and facet geometry products
* source catalogs and image diagnostics
* command records and compact logs when they explain product differences

Generated preview PNGs, dashboard artifacts, legacy metadata spelling, and
auxiliary diagnostic plot names are review aids. They should not block the
science gate unless a scenario explicitly treats them as part of the user-facing
contract.

Gate Types
----------

Use the smallest gate that answers the question:

* **Saved-reference gate** compares the current branch against curated saved
  legacy/reference products. Use it for broad regression confidence after
  product-affecting changes.
* **Branch-vs-master gate** compares explicitly prepared base/current parsets
  against ``master`` or a chosen base ref. Use it when the decision is whether
  the current branch can replace the existing production/reference branch.
* **Repeatability gate** runs multiple repetitions per branch and compares all
  same-branch and cross-branch pairs. Use it before accepting numeric
  tolerances or classifying branch differences as repeatability-bounded.
* **Risk-based option matrix** checks focused option families, such as
  normalization, prediction path, BDA/averaging, screens, cubes, peeling, or
  Stokes products. Use it after the core gate to avoid one giant,
  hard-to-debug scenario.

Run Protocol
------------

For every comparison:

* record the base/current git refs, command lines, parsets, strategy files,
  work directories, and output roots
* use explicit input paths and strategy files rather than relying on implicit
  branch-local defaults
* run ``--prepare-only`` first for branch comparisons so reviewers can inspect
  the exact inputs before launching expensive runs
* use short run roots for legacy master/CWL/Toil paths when PyBDSF or
  multiprocessing may hit AF_UNIX path-length limits
* keep large FITS, Measurement Sets, h5parm products, and full logs outside git
* store compact reports, manifests, and selected command logs under
  ``docs/source/development/equivalence_runs/``

The comparison is invalid if either run fails, required products are missing, or
the scenario does not actually exercise comparable scientific work on both
branches.

Strict Checks
-------------

Keep these checks strict unless a documented repeatability result justifies a
specific tolerance:

* operation order and required operation presence
* required product presence and product basenames
* FITS HDU count, shape, finite mask, key WCS/header metadata, and table
  structure
* h5parm solset/soltab names, axes, shapes, directions/source tables, finite
  masks, and non-numeric datasets
* sky-model source and patch counts
* region-file structure and facet geometry when it affects DD solution
  application
* output-record shape for products used by finalizers or downstream operations

Numeric Checks
--------------

FITS, h5parm, catalog, and diagnostic numeric comparisons may use tolerances,
but only when the tolerances are product-class specific and justified by
repeatability evidence. Reports should show enough statistics to explain the
decision, for example:

* max absolute delta
* percentile absolute delta
* residual RMS
* relative residual against a robust image scale
* per-plane statistics for cubes and Stokes products
* h5parm dataset min/max/finite checks and value deltas
* source count and key source-parameter deltas
* image-diagnostic deltas such as RMS, peak, and source-count changes

When a difference is accepted because it is inside the same-branch
repeatability envelope, the report must say so explicitly.

Intentional Branch Differences
------------------------------

Some current-vs-master differences are acceptable when they make the current
branch more explicit or scientifically safer. Label them clearly rather than
hiding them in broad tolerances. Current examples include:

* rejecting previous-cycle DD solution seeds when DD directions are not proven
  compatible
* keeping previous-cycle products as optimizer seeds rather than silently
  applying them during later imaging after a new calibration step
* preserving slow-gain amplitude solutions where legacy master logs a
  combination error and finishes with phase-only active solutions
* treating generated preview images and diagnostic plot artifact names as
  non-scientific review aids

Decision Bands
--------------

Use these bands for reports:

* **pass**: required products are present and strict/numeric checks pass
* **accepted with warnings**: required products are present and remaining
  differences are repeatability-bounded, non-scientific, or explicitly
  intentional
* **fail**: a run fails, a required product is missing, strict structure checks
  fail, numeric differences exceed the accepted envelope, or the scenario does
  less comparable scientific work on one branch

Do not treat a failed legacy/master behavior as the desired target without a
scientific reason. If the base branch has a known bug or implicit-state
behavior, document it as a legacy limitation and decide explicitly whether the
current branch should copy, reject, or improve it.

Rerun Policy
------------

Documentation-only, report-only, preview-artifact-only, or refactor-only
changes may rely on the current accepted evidence plus focused tests. Rerun the
relevant saved-reference, branch-vs-master, or option-matrix gate after changes
to:

* calibration strategy semantics
* DI/DD solve order, solution seeding, h5parm collection, or solution
  application
* prediction and subtraction behavior
* imaging preparation, WSClean commands, FITS products, cubes, mosaics, or
  source catalogs
* sky-model filtering, normalization, region/facet products, or diagnostics
* finalizer-visible output records or product locations

Artifacts
---------

Commit compact, reviewable evidence:

* Markdown reports
* JSON summaries
* scenario manifests
* input parset and strategy snapshots
* short command logs when they explain a result

Keep bulky raw artifacts out of git: Measurement Sets, FITS products, h5parm
files, full logs, Dask/PyBDSF reports, generated visual-comparison PNGs, and
temporary run directories. Store those in ignored run roots, ``/tmp``, or CI
artifacts.

