# Rapthor Branch Equivalence

Scenario: `dd-phase-plus-di-fulljones-normalized-repeatability:current-rep-01_vs_current-rep-03`
Run root: `/tmp/rfjnr/pairs/current-rep-01_vs_current-rep-03`

## Branch Runs

Repeatability pair: `current-rep-01_vs_current-rep-03` (current/rep-01 -> current/rep-03)

| Role | Repetition | Ref | Return Code | Parset | Work Dir | Log |
| --- | --- | --- | ---: | --- | --- | --- |
| reference | `current/rep-01` | `current` | 0 | `/tmp/rfjnr/inputs/current/rep-01/current_dd_phase_plus_di_fulljones.parset` | `/tmp/rfjnw/current/rep-01` | `/tmp/rfjnr/current/rep-01/rapthor-command.log` |
| candidate | `current/rep-03` | `current` | 0 | `/tmp/rfjnr/inputs/current/rep-03/current_dd_phase_plus_di_fulljones.parset` | `/tmp/rfjnw/current/rep-03` | `/tmp/rfjnr/current/rep-03/rapthor-command.log` |

## Comparison Summary

| Result | Ops | Records | FITS | Image HDUs | Table HDUs | H5 | Text | Diagnostics | Visuals |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pass | 5 | 5 | 7 | 6 | 1 | 4 | 10 | 1 | 5 |

## FITS Residual Metrics

| Product | Max Abs Delta | P99 Abs Delta | Residual RMS | RMS / Ref RMS | RMS / Ref MAD |
| --- | ---: | ---: | ---: | ---: | ---: |
| `field-MFS-dirty.fits` | 9.537e-07 | 7.451e-08 | 2.810e-08 | 1.625e-07 | 1.814e-07 |
| `field-MFS-image-pb-ast.fits` | 9.537e-07 | 7.264e-08 | 2.620e-08 | 3.505e-07 | 1.580e-06 |
| `field-MFS-image-pb.fits` | 9.537e-07 | 7.264e-08 | 2.620e-08 | 3.505e-07 | 1.580e-06 |
| `field-MFS-image.fits` | 9.537e-07 | 7.078e-08 | 2.557e-08 | 3.467e-07 | 1.574e-06 |
| `field-MFS-residual.fits` | 2.682e-07 | 5.402e-08 | 2.041e-08 | 1.179e-06 | 1.259e-06 |
| `field-MFS-model-pb.fits` | 2.384e-07 | 0.000e+00 | 1.806e-10 | 1.041e-07 | n/a |

## Image Diagnostics

| Operation | Sector | Field | Reference | Current | Delta | Relative Delta |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `image_1` | `sector_1` | `nsources` | 1.100e+01 | 1.100e+01 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `theoretical_rms` | 9.006e-03 | 9.006e-03 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `min_rms_flat_noise` | 8.312e-03 | 8.313e-03 | 5.076e-07 | 0.006% |
| `image_1` | `sector_1` | `median_rms_flat_noise` | 1.589e-02 | 1.589e-02 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `dynamic_range_global_flat_noise` | 5.505e+02 | 5.505e+02 | -3.356e-02 | -0.006% |
| `image_1` | `sector_1` | `min_rms_true_sky` | 8.439e-03 | 8.438e-03 | -8.382e-07 | -0.010% |
| `image_1` | `sector_1` | `median_rms_true_sky` | 1.622e-02 | 1.622e-02 | -3.725e-09 | -0.000% |
| `image_1` | `sector_1` | `dynamic_range_global_true_sky` | 5.423e+02 | 5.423e+02 | 5.381e-02 | 0.010% |

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

