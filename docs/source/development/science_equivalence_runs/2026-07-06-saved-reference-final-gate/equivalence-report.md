# Rapthor Saved-Reference Equivalence

Run root: `/tmp/rsg4`

## Scenario Summary

| Scenario | Result | Ops | Records | FITS | Image HDUs | Table HDUs | H5 | Text |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `dd_only_calibration` | pass | 4 | 4 | 6 | 5 | 1 | 3 | 10 |
| `di_only_calibration` | pass | 5 | 5 | 6 | 5 | 1 | 2 | 9 |
| `full_stokes_clean_disabled` | pass | 4 | 4 | 9 | 8 | 1 | 3 | 8 |
| `image_cube` | pass | 4 | 4 | 7 | 6 | 1 | 3 | 12 |
| `normalization` | pass | 5 | 5 | 7 | 6 | 1 | 4 | 12 |
| `peeling` | pass | 4 | 4 | 6 | 5 | 1 | 3 | 11 |
| `restart` | pass | 4 | 4 | 6 | 5 | 1 | 3 | 10 |

## FITS Residual Metrics

| Scenario | Product | Max Abs Delta | P99 Abs Delta | Residual RMS | RMS / Ref RMS | RMS / Ref MAD |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `normalization` | `field-MFS-residual.fits` | 2.833e-05 | 6.169e-06 | 2.373e-06 | 5.592e-06 | 5.591e-06 |
| `normalization` | `field-MFS-model-pb.fits` | 2.652e-05 | 0.000e+00 | 1.424e-08 | 1.996e-06 | n/a |
| `normalization` | `field-MFS-image-pb.fits` | 1.717e-05 | 6.229e-06 | 2.386e-06 | 4.771e-06 | 5.465e-06 |
| `normalization` | `field-MFS-image.fits` | 1.717e-05 | 6.091e-06 | 2.336e-06 | 4.746e-06 | 5.464e-06 |
| `normalization` | `sector_1_I_freq_cube.fits` | 4.768e-06 | 7.153e-07 | 2.270e-07 | 3.775e-07 | 4.446e-07 |
| `normalization` | `field-MFS-dirty.fits` | 3.815e-06 | 4.768e-07 | 2.018e-07 | 1.209e-07 | 1.233e-07 |
| `peeling` | `field-MFS-image-pb.fits` | 2.176e-06 | 2.384e-07 | 9.129e-08 | 1.935e-07 | 1.833e-07 |
| `peeling` | `field-MFS-image.fits` | 1.997e-06 | 2.384e-07 | 8.846e-08 | 1.912e-07 | 1.813e-07 |
| `peeling` | `field-MFS-residual.fits` | 1.997e-06 | 2.384e-07 | 8.844e-08 | 1.911e-07 | 1.812e-07 |
| `peeling` | `field-MFS-dirty.fits` | 1.609e-06 | 2.384e-07 | 8.223e-08 | 1.777e-07 | 1.685e-07 |
| `full_stokes_clean_disabled` | `field-MFS-I-image-pb.fits` | 4.657e-09 | 4.366e-11 | 2.884e-11 | 6.196e-07 | 6.812e-07 |
| `full_stokes_clean_disabled` | `field-MFS-U-image-pb.fits` | 5.803e-10 | 7.276e-12 | 4.496e-12 | 6.078e-07 | 5.867e-07 |
| `image_cube` | `sector_1_I_freq_cube.fits` | 4.547e-10 | 5.821e-11 | 1.972e-11 | 4.170e-07 | 4.554e-07 |
| `full_stokes_clean_disabled` | `field-MFS-Q-image-pb.fits` | 3.547e-10 | 4.547e-12 | 2.675e-12 | 6.392e-07 | 5.667e-07 |
| `dd_only_calibration` | `field-MFS-image-pb.fits` | 3.165e-10 | 2.910e-11 | 9.710e-12 | 2.087e-07 | 2.294e-07 |
| `full_stokes_clean_disabled` | `field-MFS-V-image-pb.fits` | 3.124e-10 | 3.738e-12 | 2.380e-12 | 6.767e-07 | 6.288e-07 |
| `dd_only_calibration` | `field-MFS-image.fits` | 2.947e-10 | 2.910e-11 | 9.431e-12 | 2.066e-07 | 2.274e-07 |
| `dd_only_calibration` | `field-MFS-residual.fits` | 2.947e-10 | 2.910e-11 | 9.431e-12 | 2.066e-07 | 2.274e-07 |
| `dd_only_calibration` | `field-MFS-dirty.fits` | 2.692e-10 | 2.910e-11 | 9.447e-12 | 2.069e-07 | 2.277e-07 |
| `image_cube` | `field-MFS-dirty.fits` | 2.510e-10 | 2.910e-11 | 9.882e-12 | 2.165e-07 | 2.382e-07 |
| `restart` | `field-MFS-image-pb.fits` | 2.401e-10 | 2.910e-11 | 9.181e-12 | 1.973e-07 | 2.169e-07 |
| `di_only_calibration` | `field-MFS-dirty.fits` | 2.292e-10 | 2.910e-11 | 1.107e-11 | 2.330e-07 | 2.600e-07 |
| `restart` | `field-MFS-dirty.fits` | 2.265e-10 | 2.910e-11 | 9.067e-12 | 1.986e-07 | 2.186e-07 |
| `restart` | `field-MFS-image.fits` | 2.246e-10 | 2.547e-11 | 8.919e-12 | 1.954e-07 | 2.150e-07 |
| `restart` | `field-MFS-residual.fits` | 2.246e-10 | 2.547e-11 | 8.919e-12 | 1.954e-07 | 2.150e-07 |

## Warnings

- `dd_only_calibration`: output-record optional artifact basenames differ for image_1
- `di_only_calibration`: output-record optional artifact basenames differ for image_1
- `full_stokes_clean_disabled`: output-record optional artifact basenames differ for image_1
- `image_cube`: output-record optional artifact basenames differ for image_1
- `normalization`: output-record optional artifact basenames differ for image_1
- `normalization`: output-record optional artifact basenames differ for normalize_1
- `peeling`: output-record optional artifact basenames differ for image_1
- `restart`: output-record optional artifact basenames differ for image_1
