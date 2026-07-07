Scalability and Performance Equivalence Contract
================================================

Purpose
-------

This contract defines how Rapthor should compare the current Prefect/Dask
branch against ``master`` for runtime behavior. It complements the science
equivalence gate: science equivalence answers whether the products are
acceptable, while performance equivalence answers whether the current branch is
at least runtime-neutral, and preferably faster or more scalable, on the same
inputs and hardware.

The first implementation should be advisory. It should fail only for
infrastructure errors, failed runs, missing products, or failed science
equivalence. Performance deltas should be reported as evidence until enough
repeatability data exists to choose hard thresholds.

Comparison Rule
---------------

Compare one current-branch run set with one base-branch run set. A comparison is
valid only when all of the following are recorded and intentionally matched:

* input Measurement Sets and external data products
* parset intent, strategy intent, and expected product contract
* preview-artifact settings
* container image or prepared software environment
* git refs for base and current branches
* CPU count, worker/thread shape, and resource limits
* thread-limiting environment variables
* run root, output root, and materialized parset path for each repetition
* command-profile and report-generation settings

If the branches cannot express exactly the same parset or strategy, the report
must describe the intended equivalence and label any known legacy limitation.
Do not interpret a performance win from a scenario where one branch silently
does less science work.

Scenario Matrix
---------------

Start with a small matrix:

* a fast CI-sized scenario that both branches can represent
* one richer scenario that exercises DI/DD calibration, imaging, mosaicking,
  sky-model filtering, h5parm handling, and output finalization

Add larger datasets, multi-node runs, and cluster-specific scenarios only after
the fast and rich scenarios are repeatable. Avoid scenarios where ``master`` has
a known scientific bug unless the report explicitly treats the base result as a
legacy limitation rather than a current-branch regression.

Run Protocol
------------

For each scenario:

* run ``--prepare-only`` first, or the equivalent future command, to verify the
  exact base/current parsets, strategies, work directories, and output roots
* run at least three repetitions per branch for decision-quality evidence
* alternate base/current repetitions when possible to reduce runner-load,
  filesystem-cache, and thermal bias
* use fresh run directories for each repetition
* keep generated FITS/postage-stamp previews disabled unless preview overhead is
  the explicit subject of the comparison
* keep command timing enabled where supported
* record Dask performance reports for the Prefect/Dask branch

The comparison is invalid if either branch fails, required products are missing,
or the paired science-equivalence check fails.

Metrics
-------

Reports should include at least:

* return code for every repetition
* wall-clock min, median, and max per branch
* current-vs-base wall-time delta
* command-profile totals and dominant command groups
* operation timing where available
* operation-minus-command gap where available
* Dask duration, compute time, duration-minus-compute gap, task count, task
  groups, worker count, thread count, and memory summary for the current branch
* output-product and science-equivalence status
* notes for known branch-specific behavior

Do not use absolute wall-clock thresholds in unit or architecture tests. Tests
should protect the runtime shape that makes these metrics meaningful.

Decision Bands
--------------

Use advisory bands at first:

* **pass**: current branch passes science equivalence and is faster, or
  runtime-neutral with better scalability, observability, or maintainability
* **warn**: current branch passes science equivalence but is slower within the
  observed repeatability envelope, or the result is too noisy to judge
* **fail**: current branch fails science equivalence, misses required products,
  fails to run, or shows a stable performance regression that is not explained
  by safer or more complete scientific behavior

State the decision in words as well as numbers. Useful outcomes include:

* current branch is faster by a stable margin
* current branch is neutral but cleaner or more scalable
* current branch is slower but scientifically safer or more complete
* current branch needs more work before it should replace ``master``

Artifacts
---------

Keep bulky run products out of git. CI should store raw benchmark and
equivalence artifacts for inspection. Commit only compact milestone evidence,
for example:

* a scenario manifest
* a Markdown report
* a JSON summary
* short command logs when they explain an important result

Accepted milestone reports should live under
``docs/source/development/performance_equivalence_runs/``. Raw run
directories, Measurement Sets, FITS products, full Dask HTML reports, Prefect
logs, and temporary working directories should remain in CI artifacts, ``runs/``
for local inspection, or ``/tmp``.

Architecture Guards
-------------------

Add tests for runtime shape, not elapsed time. Useful guards include:

* benchmark profiles remain configured
* benchmark scenarios keep preview artifacts disabled when measuring pipeline
  runtime
* command profiling and compact report generation stay enabled
* resource profiles override generic worker/thread arguments
* expected Dask task groups remain visible
* worker payloads remain plain and serializable
* command logs keep fields required by performance reports

