# Rapthor Branch Equivalence

Scenario: `dd-phase-plus-di-fulljones-repeatability:current-rep-02_vs_current-rep-03`
Run root: `/tmp/rfjr/pairs/current-rep-02_vs_current-rep-03`

## Branch Runs

Repeatability pair: `current-rep-02_vs_current-rep-03` (current/rep-02 -> current/rep-03)

| Role | Repetition | Ref | Return Code | Parset | Work Dir | Log |
| --- | --- | --- | ---: | --- | --- | --- |
| reference | `current/rep-02` | `current` | 0 | `/tmp/rfjr/inputs/current/rep-02/current_dd_phase_plus_di_fulljones.parset` | `/tmp/rfjw/current/rep-02` | `/tmp/rfjr/current/rep-02/rapthor-command.log` |
| candidate | `current/rep-03` | `current` | 0 | `/tmp/rfjr/inputs/current/rep-03/current_dd_phase_plus_di_fulljones.parset` | `/tmp/rfjw/current/rep-03` | `/tmp/rfjr/current/rep-03/rapthor-command.log` |

## Comparison Summary

| Result | Ops | Records | FITS | Image HDUs | Table HDUs | H5 | Text | Diagnostics | Visuals |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pass | 5 | 5 | 7 | 6 | 1 | 4 | 10 | 1 | 5 |

## FITS Residual Metrics

| Product | Max Abs Delta | P99 Abs Delta | Residual RMS | RMS / Ref RMS | RMS / Ref MAD |
| --- | ---: | ---: | ---: | ---: | ---: |
| `field-MFS-dirty.fits` | 9.537e-07 | 8.941e-08 | 3.008e-08 | 1.735e-07 | 1.937e-07 |
| `field-MFS-image-pb-ast.fits` | 9.537e-07 | 7.637e-08 | 2.717e-08 | 3.626e-07 | 1.634e-06 |
| `field-MFS-image-pb.fits` | 9.537e-07 | 7.637e-08 | 2.717e-08 | 3.626e-07 | 1.634e-06 |
| `field-MFS-image.fits` | 9.537e-07 | 7.532e-08 | 2.655e-08 | 3.591e-07 | 1.630e-06 |
| `field-MFS-model-pb.fits` | 2.384e-07 | 0.000e+00 | 1.818e-10 | 1.046e-07 | n/a |
| `field-MFS-residual.fits` | 1.276e-07 | 5.402e-08 | 2.032e-08 | 1.171e-06 | 1.251e-06 |

## Image Diagnostics

| Operation | Sector | Field | Reference | Current | Delta | Relative Delta |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `image_1` | `sector_1` | `nsources` | 1.100e+01 | 1.100e+01 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `theoretical_rms` | 9.006e-03 | 9.006e-03 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `min_rms_flat_noise` | 8.330e-03 | 8.329e-03 | -3.092e-07 | -0.004% |
| `image_1` | `sector_1` | `median_rms_flat_noise` | 1.592e-02 | 1.592e-02 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `dynamic_range_global_flat_noise` | 5.506e+02 | 5.506e+02 | 2.044e-02 | 0.004% |
| `image_1` | `sector_1` | `min_rms_true_sky` | 8.457e-03 | 8.457e-03 | -7.972e-07 | -0.009% |
| `image_1` | `sector_1` | `median_rms_true_sky` | 1.625e-02 | 1.625e-02 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `dynamic_range_global_true_sky` | 5.423e+02 | 5.424e+02 | 5.112e-02 | 0.009% |

## Visual Comparisons

### Image: `image_1/field-MFS-image-pb-ast.fits`

![image_1/field-MFS-image-pb-ast.fits](visual-comparisons/images/image_1-field-mfs-image-pb-ast.fits.png)

### Image: `image_1/field-MFS-image-pb.fits`

![image_1/field-MFS-image-pb.fits](visual-comparisons/images/image_1-field-mfs-image-pb.fits.png)

### Image: `image_1/field-MFS-residual.fits`

![image_1/field-MFS-residual.fits](visual-comparisons/images/image_1-field-mfs-residual.fits.png)

### Solution: `calibrate_1/fast_phase_dir[Patch_rich_centre].png`

![calibrate_1/fast_phase_dir[Patch_rich_centre].png](visual-comparisons/solutions/calibrate_1-fast_phase_dir-patch_rich_centre-.png.png)

### Solution: `calibrate_1/medium1_phase_dir[Patch_rich_centre].png`

![calibrate_1/medium1_phase_dir[Patch_rich_centre].png](visual-comparisons/solutions/calibrate_1-medium1_phase_dir-patch_rich_centre-.png.png)

