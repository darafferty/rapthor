# Rapthor Saved-Reference Equivalence

Run root: `runs/equivalence-strengthened-20260704-codex-green`

## Scenario Summary

| Scenario | Result | Ops | Records | FITS | Image HDUs | Table HDUs | H5 | Text |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `dd_only_calibration` | pass | 4 | 4 | 6 | 5 | 1 | 3 | 10 |
| `di_full_jones_calibration` | pass | 5 | 5 | 6 | 5 | 1 | 1 | 9 |
| `di_only_calibration` | pass | 5 | 5 | 6 | 5 | 1 | 2 | 9 |
| `full_stokes_clean_disabled` | pass | 4 | 4 | 9 | 8 | 1 | 3 | 8 |
| `image_cube` | pass | 4 | 4 | 7 | 6 | 1 | 3 | 12 |
| `normalization` | pass | 5 | 5 | 7 | 6 | 1 | 4 | 12 |
| `peeling` | pass | 4 | 4 | 6 | 5 | 1 | 3 | 11 |
| `restart` | pass | 4 | 4 | 6 | 5 | 1 | 3 | 10 |

## FITS Residual Metrics

| Scenario | Product | Max Abs Delta | P99 Abs Delta | Residual RMS | RMS / Ref RMS | RMS / Ref MAD |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `normalization` | `sector_1_I_freq_cube.fits` | 6.676e-06 | 6.706e-07 | 2.002e-07 | 3.329e-07 | 3.921e-07 |
| `normalization` | `field-MFS-image-pb.fits` | 5.722e-06 | 4.470e-07 | 1.689e-07 | 3.377e-07 | 3.868e-07 |
| `normalization` | `field-MFS-image.fits` | 4.768e-06 | 4.321e-07 | 1.660e-07 | 3.372e-07 | 3.882e-07 |
| `normalization` | `field-MFS-dirty.fits` | 3.815e-06 | 4.768e-07 | 2.000e-07 | 1.198e-07 | 1.222e-07 |
| `peeling` | `field-MFS-image-pb.fits` | 2.187e-06 | 2.384e-07 | 8.266e-08 | 1.752e-07 | 1.660e-07 |
| `di_full_jones_calibration` | `field-MFS-dirty.fits` | 2.086e-06 | 2.980e-07 | 1.036e-07 | 2.109e-07 | 1.962e-07 |
| `peeling` | `field-MFS-image.fits` | 2.049e-06 | 2.384e-07 | 8.026e-08 | 1.735e-07 | 1.645e-07 |
| `peeling` | `field-MFS-residual.fits` | 2.049e-06 | 2.384e-07 | 8.025e-08 | 1.734e-07 | 1.644e-07 |
| `di_full_jones_calibration` | `field-MFS-image-pb.fits` | 1.788e-06 | 3.576e-07 | 1.106e-07 | 2.207e-07 | 2.052e-07 |
| `di_full_jones_calibration` | `field-MFS-image.fits` | 1.758e-06 | 2.980e-07 | 1.076e-07 | 2.190e-07 | 2.037e-07 |
| `di_full_jones_calibration` | `field-MFS-residual.fits` | 1.758e-06 | 2.980e-07 | 1.076e-07 | 2.190e-07 | 2.037e-07 |
| `peeling` | `field-MFS-dirty.fits` | 1.550e-06 | 2.384e-07 | 8.371e-08 | 1.809e-07 | 1.715e-07 |
| `normalization` | `field-MFS-model-pb.fits` | 9.537e-07 | 0.000e+00 | 4.640e-10 | 6.502e-08 | n/a |
| `normalization` | `field-MFS-residual.fits` | 7.153e-07 | 3.874e-07 | 1.503e-07 | 3.542e-07 | 3.541e-07 |
| `image_cube` | `sector_1_I_freq_cube.fits` | 5.093e-10 | 4.547e-11 | 1.618e-11 | 3.421e-07 | 3.736e-07 |
| `full_stokes_clean_disabled` | `field-MFS-I-image-pb.fits` | 2.219e-10 | 2.410e-11 | 8.460e-12 | 1.818e-07 | 1.999e-07 |
| `restart` | `field-MFS-image-pb.fits` | 2.219e-10 | 2.910e-11 | 1.083e-11 | 2.327e-07 | 2.558e-07 |
| `dd_only_calibration` | `field-MFS-image-pb.fits` | 2.128e-10 | 2.365e-11 | 8.345e-12 | 1.793e-07 | 1.972e-07 |
| `full_stokes_clean_disabled` | `field-MFS-I-image.fits` | 2.074e-10 | 2.183e-11 | 8.221e-12 | 1.801e-07 | 1.982e-07 |
| `restart` | `field-MFS-image.fits` | 2.074e-10 | 2.910e-11 | 1.054e-11 | 2.309e-07 | 2.541e-07 |
| `restart` | `field-MFS-residual.fits` | 2.074e-10 | 2.910e-11 | 1.054e-11 | 2.309e-07 | 2.541e-07 |
| `image_cube` | `field-MFS-image-pb.fits` | 2.055e-10 | 2.728e-11 | 8.698e-12 | 1.869e-07 | 2.055e-07 |
| `dd_only_calibration` | `field-MFS-image.fits` | 2.001e-10 | 2.183e-11 | 8.105e-12 | 1.775e-07 | 1.954e-07 |
| `dd_only_calibration` | `field-MFS-residual.fits` | 2.001e-10 | 2.183e-11 | 8.105e-12 | 1.775e-07 | 1.954e-07 |
| `image_cube` | `field-MFS-image.fits` | 1.928e-10 | 2.547e-11 | 8.450e-12 | 1.851e-07 | 2.037e-07 |
