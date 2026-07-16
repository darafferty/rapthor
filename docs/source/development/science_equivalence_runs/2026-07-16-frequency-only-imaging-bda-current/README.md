# Frequency-Only Imaging BDA Validation

Status: **pass**

This focused current-branch validation closes the frequency-only imaging BDA
gap found during the July `master` sync. It ran Rapthor calibration and imaging
through DP3 frequency BDA, WSClean 3.7, EveryBeam 0.8.3, and primary-beam FITS
product generation.

The production-shaped integration scenario used:

- `imaging.bda_timebase = 0`
- `imaging.bda_frequencybase = 1000`
- calibration frequency steps that derive `bdaavg.minchannels = 4`
- DD calibration followed by WSClean imaging with facet-beam application

The run passed at current-branch commit
`2903ad141e8815e724bd57227b9c2b30f3bfaccb`. DP3 produced a two-row
`SPECTRAL_WINDOW` layout with `NUM_CHAN = [4, 8]`; WSClean received `-reorder`
and `-apply-facet-beam`; and the final primary-beam image contained 360,000
finite pixels.

This is current-branch product evidence rather than a branch-equivalence
comparison. Legacy `master` cannot supply a valid reference for this option
combination: it omits WSClean's required `-reorder` option, and its supported
EveryBeam generation rejects DP3's multi-SPW frequency-BDA layout.

Compact evidence:

- `command-evidence.json`: selected DP3 and WSClean arguments and outcomes
- `product-evidence.json`: final imaging-MS frequency layout and primary-beam
  FITS statistics

The raw run is retained under the ignored path
`runs/frequency-only-imaging-bda-20260716-current/ical-or9_p7bv/` for local
inspection. It is not part of the tracked evidence archive.
