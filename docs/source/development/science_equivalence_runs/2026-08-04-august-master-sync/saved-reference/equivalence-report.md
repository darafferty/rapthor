# Rapthor Saved-Reference Equivalence

Run root: `runs/science-gate-20260804-august-master-sync-saved-reference`

## Scenario Summary

| Scenario | Result | Ops | Records | FITS | Image HDUs | Table HDUs | H5 | Text |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `dd_only_calibration` | pass | 4 | 4 | 6 | 5 | 1 | 3 | 10 |
| `di_only_calibration` | pass | 5 | 5 | 6 | 5 | 1 | 2 | 9 |
| `full_stokes_clean_disabled` | pass | 4 | 4 | 9 | 8 | 1 | 3 | 8 |
| `image_cube` | pass | 4 | 4 | 7 | 6 | 1 | 3 | 12 |
| `normalization` | fail | 5 | 5 | 7 | 6 | 1 | 4 | 12 |
| `peeling` | fail | 4 | 4 | 6 | 5 | 1 | 3 | 11 |
| `restart` | pass | 4 | 4 | 6 | 5 | 1 | 3 | 10 |

## FITS Residual Metrics

| Scenario | Product | Max Abs Delta | P99 Abs Delta | Residual RMS | RMS / Ref RMS |
| --- | --- | ---: | ---: | ---: | ---: |
| `normalization` | `sector_1_I_freq_cube.fits` | 1.016e-03 | 2.617e-05 | 1.237e-05 | 2.058e-05 |
| `peeling` | `field-MFS-image-pb.fits` | 3.974e-04 | 6.825e-06 | 3.536e-06 | 7.496e-06 |
| `normalization` | `field-MFS-image-pb.fits` | 3.344e-04 | 6.497e-06 | 3.201e-06 | 6.400e-06 |
| `normalization` | `field-MFS-dirty.fits` | 5.722e-06 | 7.153e-07 | 2.822e-07 | 1.691e-07 |
| `normalization` | `field-MFS-image.fits` | 5.722e-06 | 5.364e-07 | 2.044e-07 | 4.152e-07 |
| `peeling` | `field-MFS-image.fits` | 2.801e-06 | 3.278e-07 | 1.095e-07 | 2.368e-07 |
| `peeling` | `field-MFS-residual.fits` | 2.801e-06 | 3.278e-07 | 1.095e-07 | 2.367e-07 |
| `peeling` | `field-MFS-dirty.fits` | 2.027e-06 | 2.980e-07 | 9.903e-08 | 2.140e-07 |
| `normalization` | `field-MFS-residual.fits` | 1.743e-06 | 4.768e-07 | 1.853e-07 | 4.366e-07 |
| `normalization` | `field-MFS-model-pb.fits` | 1.550e-06 | 0.000e+00 | 1.016e-09 | 1.424e-07 |
| `image_cube` | `sector_1_I_freq_cube.fits` | 7.957e-08 | 9.459e-10 | 5.079e-10 | 1.074e-05 |
| `dd_only_calibration` | `field-MFS-image-pb.fits` | 4.815e-08 | 6.694e-10 | 3.598e-10 | 7.733e-06 |
| `full_stokes_clean_disabled` | `field-MFS-I-image-pb.fits` | 4.812e-08 | 6.694e-10 | 3.598e-10 | 7.731e-06 |
| `image_cube` | `field-MFS-image-pb.fits` | 4.804e-08 | 6.694e-10 | 3.592e-10 | 7.719e-06 |
| `restart` | `field-MFS-image-pb.fits` | 4.801e-08 | 6.694e-10 | 3.591e-10 | 7.718e-06 |
| `di_only_calibration` | `field-MFS-image-pb.fits` | 3.875e-08 | 5.239e-10 | 3.069e-10 | 6.332e-06 |
| `full_stokes_clean_disabled` | `field-MFS-U-image-pb.fits` | 6.768e-09 | 1.173e-10 | 6.042e-11 | 8.168e-06 |
| `full_stokes_clean_disabled` | `field-MFS-Q-image-pb.fits` | 3.722e-09 | 7.549e-11 | 3.565e-11 | 8.519e-06 |
| `full_stokes_clean_disabled` | `field-MFS-V-image-pb.fits` | 2.626e-09 | 5.502e-11 | 2.593e-11 | 7.372e-06 |
| `full_stokes_clean_disabled` | `field-MFS-I-image.fits` | 3.856e-10 | 3.229e-11 | 1.108e-11 | 2.427e-07 |
| `dd_only_calibration` | `field-MFS-dirty.fits` | 3.311e-10 | 4.002e-11 | 1.295e-11 | 2.837e-07 |
| `image_cube` | `field-MFS-dirty.fits` | 3.129e-10 | 3.274e-11 | 1.127e-11 | 2.469e-07 |
| `restart` | `field-MFS-image.fits` | 2.947e-10 | 3.274e-11 | 1.103e-11 | 2.416e-07 |
| `restart` | `field-MFS-residual.fits` | 2.947e-10 | 3.274e-11 | 1.103e-11 | 2.416e-07 |
| `restart` | `field-MFS-dirty.fits` | 2.874e-10 | 3.274e-11 | 1.098e-11 | 2.404e-07 |

## Strict Failures

- `normalization`: FITS image pixels differ for field-MFS-image-pb.fits: max_abs_delta=0.0003343820571899414, p99_abs_delta=6.496906280517578e-06, residual_rms=3.2012326246535144e-06
- `normalization`: FITS image pixels differ for sector_1_I_freq_cube.fits: max_abs_delta=0.0010160207748413086, p99_abs_delta=2.6166439056396484e-05, residual_rms=1.2373903953782075e-05
- `peeling`: FITS image pixels differ for field-MFS-image-pb.fits: max_abs_delta=0.0003974437713623047, p99_abs_delta=6.8247318267822266e-06, residual_rms=3.536039226365652e-06
