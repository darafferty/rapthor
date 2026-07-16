# Post-Master-Sync Saved-Reference Gate

Status: **accepted with one dependency-baseline difference**

This gate reran every active saved-reference scenario after the LSMTool API
update and the move to EveryBeam 0.8.3. Six scenarios pass the strict saved
reference directly:

- `dd_only_calibration`
- `di_only_calibration`
- `full_stokes_clean_disabled`
- `image_cube`
- `peeling`
- `restart`

The main run stopped after `normalization`, so `peeling` and `restart` were
completed in a second invocation. The two unmodified reports are retained as
`equivalence-report.*` and `equivalence-report-tail.*`.

## Normalization Classification

`normalization` differs from the older saved frequency-cube reference. The
largest absolute residual is `3.4368e-4`, the 99th percentile absolute
residual is `2.3842e-6`, and the residual RMS is `2.1727e-6`. The residual RMS
is `3.61e-6` of the cube RMS.

This is classified as dependency-baseline drift, not current-branch
divergence, because:

- an independent current rerun reproduces the result closely, with
  current-to-current residual RMS `3.0061e-7` and no pixel differing by more
  than `1e-5`
- 121 of the 128 pixels that differ from the old reference by more than
  `1e-4` lie within ten pixels of an image edge
- the controlled `normalization-rich-demo` comparison in the paired
  post-master-sync option matrix passes between `master` and current when both
  use the same EveryBeam 0.8.3 stack

The strict saved report remains unchanged. Its failure is retained so that the
dependency transition is visible rather than hidden by a broader tolerance.
Compact repeatability statistics are in
`normalization-repeatability-evidence.json`.

## Decision

The gate supports retaining the accepted science decision for the covered
LOFAR HBA contract. The saved normalization baseline should only be replaced
as part of an explicit dependency-baseline refresh, not by weakening the
global image comparator.
