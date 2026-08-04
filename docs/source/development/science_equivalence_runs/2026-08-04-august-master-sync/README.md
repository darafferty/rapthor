# August 2026 Master-Sync Verification

Decision: **accepted with classified saved-reference warnings**.

This gate verifies the August port of three commits added to `master` after the
July sync: calibration-aware imaging/BDA frequency limits, clearer built-in
strategy documentation, and removal of DP3's unsupported
`writefullresflag` option.

The compared revisions were:

- `master`: `b307e769216a0c1315806cacde49167b2cb54f26`
- current: `59be6d9425a40b9cffd2b2db96cde497471fc73c`

## Evidence

The active saved-reference matrix is archived under `saved-reference/`. Five
of seven scenarios pass strictly. `normalization` and `peeling` retain three
strict FITS pixel failures against the older saved images:

- normalization primary-beam image: residual RMS `3.2012e-6`
- normalization frequency cube: residual RMS `1.2374e-5`
- peeling primary-beam image: residual RMS `3.5360e-6`

All operation, output-record, product-presence, FITS structure, h5parm, and
text-product checks pass in those scenarios. The strict report and original
tolerances are preserved.

Two controlled comparisons then ran `master` and current with the same
external-tool stack:

- `bda-frequency-limits/` exercises the newly ported calibration-aware
  frequency limits. It passes strictly across three operations, seven FITS
  products, two h5parm products, ten text products, and image diagnostics.
  The worst FITS residual RMS is `3.062e-8`.
- `normalization-rich-demo/` repeats the saved-reference warning path against
  current `master`. It passes strictly across four operations, eight FITS
  products, three h5parm products, twelve text products, and image diagnostics.
  The normalization-cube residual RMS is `4.187e-8`.

The July and August saved-reference inputs use the same relevant imaging and
frequency-averaging settings. The larger old-reference residuals therefore
reflect a rebuilt external-tool/dependency baseline, not a current-vs-master
regression. No comparison tolerance or saved reference was changed to obtain
this decision.

The branch reports contain single-run timings, but these are diagnostic only.
They are not a repeatability-aware performance-gate result.

Raw run products are retained under:

```text
runs/science-gate-20260804-august-master-sync-saved-reference/
runs/branch-equivalence-20260804-august-master-sync-bda-frequency-limits/
runs/branch-equivalence-20260804-august-master-sync-normalization/
```
