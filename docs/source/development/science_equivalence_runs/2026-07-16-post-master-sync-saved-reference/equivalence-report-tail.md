# Rapthor Saved-Reference Equivalence

Run root: `/app/runs/science-gate-20260716-post-master-sync-saved-reference-tail`

## Scenario Summary

| Scenario | Result | Ops | Records | FITS | Image HDUs | Table HDUs | H5 | Text |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `peeling` | pass | 4 | 4 | 6 | 5 | 1 | 3 | 11 |
| `restart` | pass | 4 | 4 | 6 | 5 | 1 | 3 | 10 |

## FITS Residual Metrics

| Scenario | Product | Max Abs Delta | P99 Abs Delta | Residual RMS | RMS / Ref RMS | RMS / Ref MAD |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `peeling` | `field-MFS-image-pb.fits` | 5.126e-06 | 3.278e-07 | 1.125e-07 | 2.385e-07 | 2.259e-07 |
| `peeling` | `field-MFS-image.fits` | 2.801e-06 | 2.980e-07 | 1.044e-07 | 2.257e-07 | 2.140e-07 |
| `peeling` | `field-MFS-residual.fits` | 2.801e-06 | 2.980e-07 | 1.044e-07 | 2.256e-07 | 2.139e-07 |
| `peeling` | `field-MFS-dirty.fits` | 2.652e-06 | 2.980e-07 | 1.023e-07 | 2.210e-07 | 2.095e-07 |
| `restart` | `field-MFS-image-pb.fits` | 6.112e-10 | 3.547e-11 | 1.194e-11 | 2.566e-07 | 2.821e-07 |
| `restart` | `field-MFS-image.fits` | 3.065e-10 | 3.092e-11 | 1.088e-11 | 2.383e-07 | 2.623e-07 |
| `restart` | `field-MFS-residual.fits` | 3.065e-10 | 3.092e-11 | 1.088e-11 | 2.383e-07 | 2.623e-07 |
| `restart` | `field-MFS-dirty.fits` | 2.783e-10 | 3.092e-11 | 1.070e-11 | 2.344e-07 | 2.579e-07 |
| `peeling` | `field-MFS-model-pb.fits` | 0.000e+00 | 0.000e+00 | 0.000e+00 | n/a | n/a |
| `restart` | `field-MFS-model-pb.fits` | 0.000e+00 | 0.000e+00 | 0.000e+00 | n/a | n/a |

## Warnings

- `peeling`: output-record optional artifact basenames differ for image_1
- `restart`: output-record optional artifact basenames differ for image_1
