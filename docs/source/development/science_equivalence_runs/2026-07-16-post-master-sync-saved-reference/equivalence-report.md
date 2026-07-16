# Rapthor Saved-Reference Equivalence

Run root: `/app/runs/science-gate-20260716-post-master-sync-saved-reference`

## Scenario Summary

| Scenario | Result | Ops | Records | FITS | Image HDUs | Table HDUs | H5 | Text |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `dd_only_calibration` | pass | 4 | 4 | 6 | 5 | 1 | 3 | 10 |
| `di_only_calibration` | pass | 5 | 5 | 6 | 5 | 1 | 2 | 9 |
| `full_stokes_clean_disabled` | pass | 4 | 4 | 9 | 8 | 1 | 3 | 8 |
| `image_cube` | pass | 4 | 4 | 7 | 6 | 1 | 3 | 12 |
| `normalization` | fail | 5 | 5 | 7 | 6 | 1 | 4 | 12 |

## FITS Residual Metrics

| Scenario | Product | Max Abs Delta | P99 Abs Delta | Residual RMS | RMS / Ref RMS | RMS / Ref MAD |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `normalization` | `sector_1_I_freq_cube.fits` | 3.437e-04 | 2.384e-06 | 2.173e-06 | 3.613e-06 | 4.255e-06 |
| `normalization` | `field-MFS-image-pb.fits` | 4.113e-05 | 6.348e-06 | 2.434e-06 | 4.867e-06 | 5.574e-06 |
| `normalization` | `field-MFS-residual.fits` | 2.885e-05 | 6.288e-06 | 2.416e-06 | 5.694e-06 | 5.692e-06 |
| `normalization` | `field-MFS-model-pb.fits` | 2.652e-05 | 0.000e+00 | 1.434e-08 | 2.010e-06 | n/a |
| `normalization` | `field-MFS-image.fits` | 1.907e-05 | 6.199e-06 | 2.377e-06 | 4.830e-06 | 5.560e-06 |
| `normalization` | `field-MFS-dirty.fits` | 5.722e-06 | 7.153e-07 | 2.636e-07 | 1.579e-07 | 1.610e-07 |
| `full_stokes_clean_disabled` | `field-MFS-I-image-pb.fits` | 4.715e-09 | 4.765e-11 | 2.914e-11 | 6.261e-07 | 6.883e-07 |
| `image_cube` | `sector_1_I_freq_cube.fits` | 1.673e-09 | 7.276e-11 | 2.476e-11 | 5.236e-07 | 5.718e-07 |
| `image_cube` | `field-MFS-image-pb.fits` | 5.966e-10 | 3.092e-11 | 1.113e-11 | 2.391e-07 | 2.629e-07 |
| `full_stokes_clean_disabled` | `field-MFS-U-image-pb.fits` | 5.948e-10 | 6.639e-11 | 2.265e-11 | 3.062e-06 | 2.956e-06 |
| `dd_only_calibration` | `field-MFS-image-pb.fits` | 5.675e-10 | 3.047e-11 | 1.108e-11 | 2.380e-07 | 2.617e-07 |
| `dd_only_calibration` | `field-MFS-image.fits` | 3.129e-10 | 2.910e-11 | 1.009e-11 | 2.210e-07 | 2.432e-07 |
| `dd_only_calibration` | `field-MFS-residual.fits` | 3.129e-10 | 2.910e-11 | 1.009e-11 | 2.210e-07 | 2.432e-07 |
| `full_stokes_clean_disabled` | `field-MFS-Q-image-pb.fits` | 3.120e-10 | 4.638e-11 | 1.571e-11 | 3.754e-06 | 3.328e-06 |
| `image_cube` | `field-MFS-image.fits` | 3.056e-10 | 2.910e-11 | 1.028e-11 | 2.252e-07 | 2.479e-07 |
| `image_cube` | `field-MFS-residual.fits` | 3.056e-10 | 2.910e-11 | 1.028e-11 | 2.252e-07 | 2.479e-07 |
| `full_stokes_clean_disabled` | `field-MFS-V-image-pb.fits` | 3.020e-10 | 4.093e-12 | 2.466e-12 | 7.012e-07 | 6.515e-07 |
| `full_stokes_clean_disabled` | `field-MFS-I-image.fits` | 2.838e-10 | 2.910e-11 | 1.055e-11 | 2.311e-07 | 2.543e-07 |
| `image_cube` | `field-MFS-dirty.fits` | 2.583e-10 | 2.910e-11 | 1.046e-11 | 2.291e-07 | 2.522e-07 |
| `dd_only_calibration` | `field-MFS-dirty.fits` | 2.437e-10 | 2.910e-11 | 9.866e-12 | 2.161e-07 | 2.378e-07 |
| `di_only_calibration` | `field-MFS-image-pb.fits` | 2.328e-10 | 2.638e-11 | 8.854e-12 | 1.827e-07 | 2.037e-07 |
| `di_only_calibration` | `field-MFS-dirty.fits` | 1.965e-10 | 2.547e-11 | 8.502e-12 | 1.789e-07 | 1.996e-07 |
| `di_only_calibration` | `field-MFS-image.fits` | 1.892e-10 | 2.547e-11 | 8.607e-12 | 1.811e-07 | 2.021e-07 |
| `di_only_calibration` | `field-MFS-residual.fits` | 1.892e-10 | 2.547e-11 | 8.607e-12 | 1.811e-07 | 2.021e-07 |
| `full_stokes_clean_disabled` | `field-MFS-U-image.fits` | 9.527e-11 | 6.457e-11 | 2.179e-11 | 3.003e-06 | 2.903e-06 |

## Failures

- `normalization`: FITS image pixels differ for sector_1_I_freq_cube.fits: max_abs_delta=0.00034368038177490234, p99_abs_delta=2.384185791015625e-06, residual_rms=2.172743359320716e-06

## Warnings

- `dd_only_calibration`: output-record optional artifact basenames differ for image_1
- `di_only_calibration`: output-record optional artifact basenames differ for image_1
- `full_stokes_clean_disabled`: output-record optional artifact basenames differ for image_1
- `image_cube`: output-record optional artifact basenames differ for image_1
- `normalization`: output-record optional artifact basenames differ for image_1
- `normalization`: output-record optional artifact basenames differ for normalize_1
