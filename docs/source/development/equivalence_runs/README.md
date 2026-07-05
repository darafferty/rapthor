# Branch Equivalence Runs

This directory stores compact branch-vs-master equivalence reports that are
useful for reviewing calibration-strategy migration behavior. Generated data
products remain under ignored run directories; only reports, manifests, Rapthor
command logs, compact visual comparisons, and input parset/strategy snapshots
are tracked here.

- `2026-07-04-saved-reference-strengthened/`: compact strengthened
  saved-reference report copied out of the raw run directory before cleanup.
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
- `2026-07-05-fixed-facet-carryover-master-ref/`: two-cycle fixed-facet DD
  phase-only carry-over scenario; cycle 2 is calibration-only so the report
  isolates previous-cycle solution seeding. Master carries only the fast-phase
  seed, while the current branch carries compatible fast and medium seeds.
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
