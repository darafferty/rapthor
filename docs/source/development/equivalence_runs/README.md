# Branch Equivalence Runs

This directory stores compact branch-vs-master equivalence reports that are
useful for reviewing calibration-strategy migration behavior. Generated data
products remain under ignored run directories; only reports, manifests, Rapthor
command logs, compact visual comparisons, and input parset/strategy snapshots
are tracked here.

- `2026-07-04-default-like-master-ref/`: default-like four-cycle comparison
  that exposes the legacy master slow-gain/amplitude-solution behavior.
- `2026-07-04-phase-only-master-ref/`: four-cycle phase-only comparison using
  fast and medium phase solves, avoiding the slow-gain path.
- `2026-07-05-phase-only-master-ref/`: rerun after the master feature ports,
  including image diagnostic deltas and compact side-by-side image/solution
  visual comparisons.
