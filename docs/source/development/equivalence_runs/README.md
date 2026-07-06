# Branch Equivalence Runs

This directory stores compact branch-vs-master equivalence reports that are
useful for reviewing calibration-strategy migration behavior. Generated data
products remain under ignored run directories; only reports, manifests, Rapthor
command logs, and input parset/strategy snapshots are tracked here.

Use `scripts/dev/run_branch_equivalence.py --repeatability-repetitions 3` when
deciding whether branch differences are scientifically meaningful. This writes
unique generated parsets and work directories for each base/current repetition,
then compares all same-branch pairs and all base-current pairs. Track the
compact `repeatability-summary.*` files and selected per-pair reports here; keep
raw FITS, MS, h5parm, and full log products in ignored run directories.
Generated visual-comparison PNGs are ignored by default because the numeric
summaries are usually more informative and much lighter for git history.
Force-add only a tiny curated PNG set when a visual difference is genuinely
useful for review.

Use `scripts/dev/run_branch_option_matrix.py` for focused option-equivalence
checks after the core science gate. The matrix file should list explicitly
prepared base/current parsets; the wrapper only orchestrates scenarios and
summarizes the existing branch-equivalence reports. Keep each scenario scoped to
one option family, for example normalization, prediction path, BDA/averaging, or
screens. Mark target-environment scenarios as skipped until the required
external tools are available.

Example matrix shape:

```json
{
  "description": "Risk-based option equivalence checks.",
  "scenarios": [
    {
      "id": "normalization",
      "base_parset": "inputs/base/normalization.parset",
      "current_parset": "inputs/current/normalization.parset"
    },
    {
      "id": "screens",
      "skip_reason": "requires reliable IDGCal/screen support"
    }
  ]
}
```

When running repeatability or branch-vs-master checks that execute the legacy
master CWL path, keep `--run-root`, `--repeatability-work-root`, and any base
checkout/venv paths short, preferably under `/tmp` with compact names. The
master image/filter path runs PyBDSF through Toil scratch directories, and long
paths can make Python multiprocessing fail with `OSError: AF_UNIX path too
long`.

- `2026-07-04-saved-reference-strengthened/`: compact strengthened
  saved-reference report copied out of the raw run directory before cleanup.
- `2026-07-06-saved-reference-final-gate/`: refreshed saved-reference gate
  after output-record semantic cleanup and relative image-jitter handling. All
  default non-stale saved scenarios pass; the old DI full-Jones CWL fixture is
  excluded from the default matrix because it predates same-cycle full-Jones
  imaging application.
- `2026-07-04-default-like-master-ref/`: default-like four-cycle comparison
  that exposes the legacy master slow-gain/amplitude-solution behavior.
- `2026-07-04-phase-only-master-ref/`: four-cycle phase-only comparison using
  fast and medium phase solves, avoiding the slow-gain path.
- `2026-07-05-phase-only-master-ref/`: rerun after the master feature ports,
  including image diagnostic deltas and compact side-by-side image/solution
  visual comparisons.
- `2026-07-05-phase-only-initial-solutions-master-ref/`: rerun after aligning
  current-branch DD previous-cycle initial-solution handling with the master
  phase-only behavior; includes input snapshots, command logs, diagnostics, and
  compact image/solution visual comparisons.
- `2026-07-05-dd-phase-plus-di-fulljones-master-ref/`: single-cycle DD
  fast+medium phase-only calibration followed by DI full-Jones, using explicit
  master/current strategies; both branches complete, with compact diagnostics
  and visual comparisons for the remaining strict product differences.
- `2026-07-05-dd-phase-plus-di-fulljones-repeatability-master-ref/`:
  three-repeat branch-vs-master repeatability envelope for the DD phase plus
  DI full-Jones scenario. Master is stable within current strict tolerances;
  one current repetition drifts beyond strict tolerances, and all cross-branch
  pairs remain systematically larger than same-branch master scatter. This is
  pre-fix evidence: later investigation traced the systematic split to missing
  current-branch full-Jones gain normalization after h5parm collection, so the
  focused scenario should be rerun before using this envelope to set
  tolerances.
- `2026-07-06-dd-phase-plus-di-fulljones-normalized-master-ref/`: focused
  rerun after porting the legacy full-Jones gain normalization step. Both
  branches complete, strict h5parm comparison now passes, and the compact
  report classifies the remaining items as small image residuals, a sparse
  model-image residual, PyBDSF diagnostic catalog columns, DS9 region text
  formatting, and legacy output-record metadata shape.
- `2026-07-06-dd-phase-plus-di-fulljones-normalized-repeatability-master-ref/`:
  refreshed three-repeat envelope after the full-Jones normalization fix and
  comparison-rule cleanup. All base-base, current-current, and base-current
  pairs pass; cross-branch pairs retain only non-blocking auxiliary
  output-record artifact-name warnings for diagnostic plot names and the local
  full-Jones h5 alias.
- `2026-07-05-di-multicycle-carryover-master-ref/`: two selfcal/image cycles
  of master-compatible DD fast+medium phase-only calibration followed by DI
  full-Jones. This exposed and fixed a current-branch full-Jones
  initial-solution soltab bug; both branches now complete, and the report
  captures second-cycle image deltas after DI full-Jones carry-over.
- `2026-07-05-di-then-dd-mode-boundary-master-ref/`: two-cycle fixed-facet
  mode-boundary scenario. Cycle 1 runs DD plus DI full-Jones; cycle 2 returns
  to DD-only calibration and imaging. Master carries the cycle-1 full-Jones
  product into cycle-2 imaging, while the current branch intentionally keeps
  imaging-time full-Jones application current-cycle guarded after a new
  calibration step.
- `2026-07-05-dd-then-di-mode-boundary-master-ref/`: paired two-cycle
  fixed-facet mode-boundary scenario. Cycle 1 is DD-only; cycle 2 adds DI
  full-Jones after DD. Both branches apply the cycle-2 full-Jones product in
  cycle-2 imaging; the remaining split is mainly compatible DD initial-solution
  seeding, where the current branch also carries the medium-phase seed.
- `2026-07-05-fixed-facet-carryover-master-ref/`: two-cycle fixed-facet DD
  phase-only carry-over scenario; cycle 2 is calibration-only so the report
  isolates previous-cycle solution seeding. Master carries only the fast-phase
  seed, while the current branch carries compatible fast and medium seeds.
- `2026-07-05-fixed-facet-repeatability-master-ref/`: three-repeat
  branch-vs-master repeatability envelope for the fixed-facet DD carry-over
  scenario. The run used short `/tmp` paths to avoid the legacy master
  PyBDSF/Toil AF_UNIX path-length failure, and stores all compact pair reports.
- `2026-07-05-changing-facet-carryover-master-ref/`: two-cycle DD phase-only
  carry-over scenario with no fixed facet layout and a five-to-three direction
  change in cycle 2. Master carries the previous fast-phase seed despite the
  changed directions, while the current branch skips previous DD seeds and logs
  the direction mismatch.
- `2026-07-05-slow-gain-default-like-master-ref/`: one-cycle calibration-only
  DD default-like slow-gain scenario. Both branches produce standalone
  slow-gain and post-slow medium-phase products, but master logs the known
  h5parm-combination broadcast error and its final `field-solutions.h5` lacks
  the amplitude soltab that the current branch preserves.
