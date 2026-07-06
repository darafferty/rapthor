# Rapthor Branch Equivalence

Scenario: `dd-phase-plus-di-fulljones-normalized-repeatability:current-rep-01_vs_current-rep-02`
Run root: `/tmp/rfjnr/pairs/current-rep-01_vs_current-rep-02`

## Branch Runs

Repeatability pair: `current-rep-01_vs_current-rep-02` (current/rep-01 -> current/rep-02)

| Role | Repetition | Ref | Return Code | Parset | Work Dir | Log |
| --- | --- | --- | ---: | --- | --- | --- |
| reference | `current/rep-01` | `current` | 0 | `/tmp/rfjnr/inputs/current/rep-01/current_dd_phase_plus_di_fulljones.parset` | `/tmp/rfjnw/current/rep-01` | `/tmp/rfjnr/current/rep-01/rapthor-command.log` |
| candidate | `current/rep-02` | `current` | 0 | `/tmp/rfjnr/inputs/current/rep-02/current_dd_phase_plus_di_fulljones.parset` | `/tmp/rfjnw/current/rep-02` | `/tmp/rfjnr/current/rep-02/rapthor-command.log` |

## Comparison Summary

| Result | Ops | Records | FITS | Image HDUs | Table HDUs | H5 | Text | Diagnostics | Visuals |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pass | 5 | 5 | 7 | 6 | 1 | 4 | 10 | 1 | 5 |

## FITS Residual Metrics

| Product | Max Abs Delta | P99 Abs Delta | Residual RMS | RMS / Ref RMS | RMS / Ref MAD |
| --- | ---: | ---: | ---: | ---: | ---: |
| `field-MFS-dirty.fits` | 9.537e-07 | 8.941e-08 | 2.973e-08 | 1.719e-07 | 1.919e-07 |
| `field-MFS-image-pb-ast.fits` | 9.537e-07 | 8.009e-08 | 2.906e-08 | 3.888e-07 | 1.752e-06 |
| `field-MFS-image-pb.fits` | 9.537e-07 | 8.009e-08 | 2.906e-08 | 3.888e-07 | 1.752e-06 |
| `field-MFS-image.fits` | 9.537e-07 | 7.823e-08 | 2.863e-08 | 3.881e-07 | 1.762e-06 |
| `field-MFS-residual.fits` | 4.172e-07 | 5.774e-08 | 2.251e-08 | 1.300e-06 | 1.389e-06 |
| `field-MFS-model-pb.fits` | 2.384e-07 | 0.000e+00 | 1.751e-10 | 1.010e-07 | n/a |

## Image Diagnostics

| Operation | Sector | Field | Reference | Current | Delta | Relative Delta |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `image_1` | `sector_1` | `nsources` | 1.100e+01 | 1.100e+01 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `theoretical_rms` | 9.006e-03 | 9.006e-03 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `min_rms_flat_noise` | 8.312e-03 | 8.312e-03 | -3.250e-07 | -0.004% |
| `image_1` | `sector_1` | `median_rms_flat_noise` | 1.589e-02 | 1.589e-02 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `dynamic_range_global_flat_noise` | 5.505e+02 | 5.506e+02 | 2.147e-02 | 0.004% |
| `image_1` | `sector_1` | `min_rms_true_sky` | 8.439e-03 | 8.438e-03 | -6.640e-07 | -0.008% |
| `image_1` | `sector_1` | `median_rms_true_sky` | 1.622e-02 | 1.622e-02 | -1.118e-08 | -0.000% |
| `image_1` | `sector_1` | `dynamic_range_global_true_sky` | 5.423e+02 | 5.423e+02 | 4.262e-02 | 0.008% |

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

