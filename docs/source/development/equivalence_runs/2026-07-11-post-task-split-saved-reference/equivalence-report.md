# Rapthor Saved-Reference Equivalence

Run root: `runs/science-gate-20260711-post-task-split-rerun`

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
| `normalization` | `field-MFS-dirty.fits` | 5.722e-06 | 5.960e-07 | 2.206e-07 | 1.322e-07 | 1.347e-07 |
| `normalization` | `field-MFS-image-pb.fits` | 5.722e-06 | 4.768e-07 | 1.857e-07 | 3.712e-07 | 4.252e-07 |
| `normalization` | `field-MFS-image.fits` | 5.722e-06 | 4.768e-07 | 1.826e-07 | 3.710e-07 | 4.271e-07 |
| `normalization` | `sector_1_I_freq_cube.fits` | 5.722e-06 | 6.855e-07 | 2.031e-07 | 3.377e-07 | 3.977e-07 |
| `peeling` | `field-MFS-image-pb.fits` | 2.027e-06 | 2.831e-07 | 9.749e-08 | 2.067e-07 | 1.957e-07 |
| `peeling` | `field-MFS-image.fits` | 1.907e-06 | 2.682e-07 | 9.466e-08 | 2.046e-07 | 1.940e-07 |
| `peeling` | `field-MFS-residual.fits` | 1.907e-06 | 2.682e-07 | 9.467e-08 | 2.046e-07 | 1.939e-07 |
| `peeling` | `field-MFS-dirty.fits` | 1.848e-06 | 2.980e-07 | 1.017e-07 | 2.197e-07 | 2.083e-07 |
| `normalization` | `field-MFS-residual.fits` | 9.537e-07 | 4.172e-07 | 1.632e-07 | 3.845e-07 | 3.844e-07 |
| `normalization` | `field-MFS-model-pb.fits` | 7.153e-07 | 0.000e+00 | 4.422e-10 | 6.197e-08 | n/a |
| `full_stokes_clean_disabled` | `field-MFS-I-image-pb.fits` | 4.686e-09 | 4.366e-11 | 2.863e-11 | 6.153e-07 | 6.764e-07 |
| `full_stokes_clean_disabled` | `field-MFS-U-image-pb.fits` | 5.657e-10 | 7.276e-12 | 4.491e-12 | 6.071e-07 | 5.860e-07 |
| `image_cube` | `sector_1_I_freq_cube.fits` | 5.311e-10 | 5.821e-11 | 1.953e-11 | 4.130e-07 | 4.510e-07 |
| `full_stokes_clean_disabled` | `field-MFS-Q-image-pb.fits` | 3.588e-10 | 4.547e-12 | 2.681e-12 | 6.406e-07 | 5.679e-07 |
| `full_stokes_clean_disabled` | `field-MFS-V-image-pb.fits` | 3.065e-10 | 3.865e-12 | 2.367e-12 | 6.731e-07 | 6.254e-07 |
| `image_cube` | `field-MFS-dirty.fits` | 2.765e-10 | 2.910e-11 | 9.583e-12 | 2.099e-07 | 2.310e-07 |
| `restart` | `field-MFS-image-pb.fits` | 2.474e-10 | 2.910e-11 | 1.078e-11 | 2.317e-07 | 2.548e-07 |
| `restart` | `field-MFS-image.fits` | 2.292e-10 | 2.910e-11 | 1.049e-11 | 2.299e-07 | 2.530e-07 |
| `restart` | `field-MFS-residual.fits` | 2.292e-10 | 2.910e-11 | 1.049e-11 | 2.299e-07 | 2.530e-07 |
| `dd_only_calibration` | `field-MFS-dirty.fits` | 2.237e-10 | 2.910e-11 | 1.037e-11 | 2.271e-07 | 2.500e-07 |
| `full_stokes_clean_disabled` | `field-MFS-I-image.fits` | 2.219e-10 | 2.612e-11 | 8.828e-12 | 1.933e-07 | 2.128e-07 |
| `di_only_calibration` | `field-MFS-dirty.fits` | 2.146e-10 | 2.284e-11 | 8.262e-12 | 1.738e-07 | 1.940e-07 |
| `image_cube` | `field-MFS-image-pb.fits` | 2.110e-10 | 2.910e-11 | 9.349e-12 | 2.009e-07 | 2.209e-07 |
| `restart` | `field-MFS-dirty.fits` | 2.037e-10 | 2.910e-11 | 1.034e-11 | 2.265e-07 | 2.493e-07 |
| `di_only_calibration` | `field-MFS-image-pb.fits` | 2.001e-10 | 2.728e-11 | 8.743e-12 | 1.804e-07 | 2.011e-07 |

## Warnings

- `dd_only_calibration`: output-record optional artifact basenames differ for image_1
- `di_only_calibration`: output-record optional artifact basenames differ for image_1
- `full_stokes_clean_disabled`: output-record optional artifact basenames differ for image_1
- `image_cube`: output-record optional artifact basenames differ for image_1
- `normalization`: output-record optional artifact basenames differ for image_1
- `normalization`: output-record optional artifact basenames differ for normalize_1
- `peeling`: output-record optional artifact basenames differ for image_1
- `restart`: output-record optional artifact basenames differ for image_1
