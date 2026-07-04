# Rapthor Branch Equivalence

Scenario: `benchmark-phase-only`
Run root: `/app/runs/rbe-phase-only-master-ref-rerun-20260704-codex`

## Branch Runs

| Side | Ref | Return Code | Parset | Work Dir | Log |
| --- | --- | ---: | --- | --- | --- |
| base | `master` | 0 | `/app/runs/master-benchmark-phase-only-manual/inputs/master_benchmark_phase_only.parset` | `/tmp/rbe-master-phase-only-work` | `/app/runs/rbe-phase-only-master-ref-rerun-20260704-codex/base/rapthor-command.log` |
| current | `current` | 0 | `/app/runs/current-benchmark-phase-only-manual/inputs/current_benchmark_phase_only.parset` | `/tmp/rbe-current-phase-only-work` | `/app/runs/rbe-phase-only-master-ref-rerun-20260704-codex/current/rapthor-command.log` |

## Comparison Summary

| Result | Ops | Records | FITS | Image HDUs | Table HDUs | H5 | Text |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fail | 12 | 12 | 28 | 20 | 4 | 8 | 37 |

## FITS Residual Metrics

| Product | Max Abs Delta | P99 Abs Delta | Residual RMS | RMS / Ref RMS | RMS / Ref MAD |
| --- | ---: | ---: | ---: | ---: | ---: |
| `field-MFS-model-pb.fits.fz` | 4.326e-01 | 0.000e+00 | 4.619e-03 | 1.206e+00 | n/a |
| `field-MFS-model-pb.fits.fz` | 2.327e-01 | 0.000e+00 | 3.454e-03 | 1.141e+00 | n/a |
| `field-MFS-model-pb.fits.fz` | 1.645e-01 | 0.000e+00 | 2.695e-03 | 9.672e-01 | n/a |
| `field-MFS-dirty.fits.fz` | 1.485e-05 | 9.939e-06 | 4.154e-06 | 2.404e-05 | 2.682e-05 |
| `field-MFS-dirty.fits.fz` | 1.406e-05 | 9.898e-06 | 4.159e-06 | 2.420e-05 | 2.688e-05 |
| `field-MFS-dirty.fits.fz` | 1.398e-05 | 9.898e-06 | 4.156e-06 | 2.418e-05 | 2.685e-05 |
| `field-MFS-image.fits.fz` | 5.484e-06 | 3.584e-06 | 1.482e-06 | 1.731e-05 | 3.268e-05 |
| `field-MFS-image-pb.fits.fz` | 5.292e-06 | 3.640e-06 | 1.512e-06 | 1.739e-05 | 3.265e-05 |
| `field-MFS-residual.fits.fz` | 5.153e-06 | 3.555e-06 | 1.474e-06 | 3.200e-05 | 3.259e-05 |
| `field-MFS-image-pb.fits.fz` | 4.593e-06 | 3.219e-06 | 1.356e-06 | 1.565e-05 | 2.964e-05 |
| `field-MFS-image.fits.fz` | 4.500e-06 | 3.159e-06 | 1.329e-06 | 1.557e-05 | 2.965e-05 |
| `field-MFS-residual.fits.fz` | 4.472e-06 | 3.137e-06 | 1.321e-06 | 2.900e-05 | 2.956e-05 |
| `field-MFS-image-pb.fits.fz` | 4.292e-06 | 2.850e-06 | 1.193e-06 | 1.417e-05 | 2.861e-05 |
| `field-MFS-residual.fits.fz` | 4.232e-06 | 2.775e-06 | 1.162e-06 | 2.787e-05 | 2.852e-05 |
| `field-MFS-image.fits.fz` | 4.224e-06 | 2.792e-06 | 1.169e-06 | 1.409e-05 | 2.861e-05 |
| `field-MFS-image-pb.fits` | 1.431e-06 | 1.136e-07 | 3.826e-08 | 4.849e-07 | 1.356e-06 |
| `field-MFS-image.fits` | 1.431e-06 | 1.118e-07 | 3.732e-08 | 4.796e-07 | 1.350e-06 |
| `field-MFS-dirty.fits` | 1.229e-06 | 4.917e-07 | 1.343e-07 | 7.822e-07 | 8.720e-07 |
| `field-MFS-residual.fits` | 2.533e-07 | 1.006e-07 | 3.365e-08 | 1.160e-06 | 1.219e-06 |
| `field-MFS-model-pb.fits` | 1.788e-07 | 0.000e+00 | 1.450e-10 | 7.353e-08 | n/a |

## Warnings

- output-record summary differs for calibrate_1
- output-record summary differs for calibrate_2
- output-record summary differs for calibrate_3
- output-record summary differs for calibrate_4
- output-record summary differs for image_1
- output-record summary differs for image_2
- output-record summary differs for image_3
- output-record summary differs for image_4

## Failures

- FITS image pixels differ for field-MFS-dirty.fits.fz: max_abs_delta=1.484900712966919e-05, p99_abs_delta=9.939074516296387e-06, residual_rms=4.154122105138485e-06
- missing FITS product: /tmp/rbe-current-phase-only-work/images/image_1/field-MFS-image-pb-ast.fits.fz
- FITS image pixels differ for field-MFS-image-pb.fits.fz: max_abs_delta=4.291534423828125e-06, p99_abs_delta=2.8498470783233643e-06, residual_rms=1.1934925305742516e-06
- FITS image pixels differ for field-MFS-image.fits.fz: max_abs_delta=4.2244791984558105e-06, p99_abs_delta=2.7921050786972046e-06, residual_rms=1.1692712559533138e-06
- FITS std differs for field-MFS-model-pb.fits.fz: 0.0030267024880673904 != 0.0030308064512846013
- FITS rms differs for field-MFS-model-pb.fits.fz: 0.0030267066139305454 != 0.0030308105746002954
- FITS max differs for field-MFS-model-pb.fits.fz: 1.673178791999817 != 1.6863727569580078
- FITS image pixels differ for field-MFS-model-pb.fits.fz: max_abs_delta=0.23271916061639786, p99_abs_delta=0.0, residual_rms=0.003454141789100341
- FITS image pixels differ for field-MFS-residual.fits.fz: max_abs_delta=4.231929779052734e-06, p99_abs_delta=2.775341272354126e-06, residual_rms=1.1620359984165903e-06
- FITS image pixels differ for field-MFS-dirty.fits.fz: max_abs_delta=1.3977289199829102e-05, p99_abs_delta=9.898096323013306e-06, residual_rms=4.15577024539905e-06
- missing FITS product: /tmp/rbe-current-phase-only-work/images/image_2/field-MFS-image-pb-ast.fits.fz
- FITS image pixels differ for field-MFS-image-pb.fits.fz: max_abs_delta=4.59328293800354e-06, p99_abs_delta=3.2186508178710938e-06, residual_rms=1.3559155307134995e-06
- FITS image pixels differ for field-MFS-image.fits.fz: max_abs_delta=4.500150680541992e-06, p99_abs_delta=3.159046173095703e-06, residual_rms=1.3289755888202452e-06
- FITS std differs for field-MFS-model-pb.fits.fz: 0.0027858795579493467 != 0.0027914490377553675
- FITS rms differs for field-MFS-model-pb.fits.fz: 0.00278588318855657 != 0.0027914527824504523
- FITS image pixels differ for field-MFS-model-pb.fits.fz: max_abs_delta=0.16446129977703094, p99_abs_delta=0.0, residual_rms=0.0026945848267478094
- FITS image pixels differ for field-MFS-residual.fits.fz: max_abs_delta=4.472211003303528e-06, p99_abs_delta=3.1366944313049316e-06, residual_rms=1.3205511290182239e-06
- FITS image pixels differ for field-MFS-dirty.fits.fz: max_abs_delta=1.4059245586395264e-05, p99_abs_delta=9.898096323013306e-06, residual_rms=4.158604798657736e-06
- missing FITS product: /tmp/rbe-current-phase-only-work/images/image_3/field-MFS-image-pb-ast.fits.fz
- FITS image pixels differ for field-MFS-image-pb.fits.fz: max_abs_delta=5.291774868965149e-06, p99_abs_delta=3.63960862159729e-06, residual_rms=1.5116283555111625e-06
- FITS image pixels differ for field-MFS-image.fits.fz: max_abs_delta=5.4836273193359375e-06, p99_abs_delta=3.5837292671203613e-06, residual_rms=1.4823235441110664e-06
- FITS std differs for field-MFS-model-pb.fits.fz: 0.0038303014173447667 != 0.0037789568617443117
- FITS rms differs for field-MFS-model-pb.fits.fz: 0.003830304906959494 != 0.0037789605357193613
- FITS max differs for field-MFS-model-pb.fits.fz: 3.1229660511016846 != 2.7426722049713135
- FITS image pixels differ for field-MFS-model-pb.fits.fz: max_abs_delta=0.4326436072587967, p99_abs_delta=0.0, residual_rms=0.004619238236461788
- FITS image pixels differ for field-MFS-residual.fits.fz: max_abs_delta=5.153007805347443e-06, p99_abs_delta=3.5548582673072815e-06, residual_rms=1.4735693861751413e-06
- missing FITS product: /tmp/rbe-current-phase-only-work/images/image_4/field-MFS-image-pb-ast.fits
- HDF5 numeric dataset differs for field-solutions-fast-phase.h5:sol000/phase000/val (max_abs=1.53752e-05)
- HDF5 numeric dataset differs for field-solutions.h5:sol000/phase000/val (max_abs=1.53752e-05)
- FITS table column differs for sector_1.source_catalog.fits:PA_img_plane
- text product differs for sector_1_facets_ds9.reg
- text product differs for sector_1_facets_ds9.reg
- text product differs for sector_1_facets_ds9.reg
