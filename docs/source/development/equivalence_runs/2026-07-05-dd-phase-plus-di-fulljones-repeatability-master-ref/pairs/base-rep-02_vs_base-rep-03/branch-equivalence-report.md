# Rapthor Branch Equivalence

Scenario: `dd-phase-plus-di-fulljones-repeatability:base-rep-02_vs_base-rep-03`
Run root: `/tmp/rfjr/pairs/base-rep-02_vs_base-rep-03`

## Branch Runs

Repeatability pair: `base-rep-02_vs_base-rep-03` (base/rep-02 -> base/rep-03)

| Role | Repetition | Ref | Return Code | Parset | Work Dir | Log |
| --- | --- | --- | ---: | --- | --- | --- |
| reference | `base/rep-02` | `master` | 0 | `/tmp/rfjr/inputs/base/rep-02/master_dd_phase_plus_di_fulljones.parset` | `/tmp/rfjw/base/rep-02` | `/tmp/rfjr/base/rep-02/rapthor-command.log` |
| candidate | `base/rep-03` | `master` | 0 | `/tmp/rfjr/inputs/base/rep-03/master_dd_phase_plus_di_fulljones.parset` | `/tmp/rfjw/base/rep-03` | `/tmp/rfjr/base/rep-03/rapthor-command.log` |

## Comparison Summary

| Result | Ops | Records | FITS | Image HDUs | Table HDUs | H5 | Text | Diagnostics | Visuals |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pass | 5 | 5 | 7 | 6 | 1 | 3 | 10 | 1 | 5 |

## FITS Residual Metrics

| Product | Max Abs Delta | P99 Abs Delta | Residual RMS | RMS / Ref RMS | RMS / Ref MAD |
| --- | ---: | ---: | ---: | ---: | ---: |
| `field-MFS-image-pb-ast.fits` | 1.192e-06 | 8.196e-08 | 3.042e-08 | 4.070e-07 | 1.834e-06 |
| `field-MFS-image-pb.fits` | 1.192e-06 | 8.196e-08 | 3.042e-08 | 4.070e-07 | 1.834e-06 |
| `field-MFS-dirty.fits` | 9.537e-07 | 7.451e-08 | 2.817e-08 | 1.629e-07 | 1.819e-07 |
| `field-MFS-image.fits` | 9.537e-07 | 8.009e-08 | 2.938e-08 | 3.983e-07 | 1.808e-06 |
| `field-MFS-model-pb.fits` | 4.768e-07 | 0.000e+00 | 3.385e-10 | 1.952e-07 | n/a |
| `field-MFS-residual.fits` | 4.619e-07 | 5.960e-08 | 2.345e-08 | 1.355e-06 | 1.447e-06 |

## Image Diagnostics

| Operation | Sector | Field | Reference | Current | Delta | Relative Delta |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `image_1` | `sector_1` | `nsources` | 1.100e+01 | 1.100e+01 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `theoretical_rms` | 9.006e-03 | 9.006e-03 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `min_rms_flat_noise` | 8.312e-03 | 8.312e-03 | 1.714e-07 | 0.002% |
| `image_1` | `sector_1` | `median_rms_flat_noise` | 1.589e-02 | 1.589e-02 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `dynamic_range_global_flat_noise` | 5.506e+02 | 5.505e+02 | -1.124e-02 | -0.002% |
| `image_1` | `sector_1` | `min_rms_true_sky` | 8.438e-03 | 8.437e-03 | -3.176e-07 | -0.004% |
| `image_1` | `sector_1` | `median_rms_true_sky` | 1.622e-02 | 1.622e-02 | 3.725e-09 | 0.000% |
| `image_1` | `sector_1` | `dynamic_range_global_true_sky` | 5.424e+02 | 5.424e+02 | 2.036e-02 | 0.004% |

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

