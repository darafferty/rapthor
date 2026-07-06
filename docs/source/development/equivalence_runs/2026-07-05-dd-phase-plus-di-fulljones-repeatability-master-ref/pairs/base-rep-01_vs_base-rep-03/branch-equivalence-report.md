# Rapthor Branch Equivalence

Scenario: `dd-phase-plus-di-fulljones-repeatability:base-rep-01_vs_base-rep-03`
Run root: `/tmp/rfjr/pairs/base-rep-01_vs_base-rep-03`

## Branch Runs

Repeatability pair: `base-rep-01_vs_base-rep-03` (base/rep-01 -> base/rep-03)

| Role | Repetition | Ref | Return Code | Parset | Work Dir | Log |
| --- | --- | --- | ---: | --- | --- | --- |
| reference | `base/rep-01` | `master` | 0 | `/tmp/rfjr/inputs/base/rep-01/master_dd_phase_plus_di_fulljones.parset` | `/tmp/rfjw/base/rep-01` | `/tmp/rfjr/base/rep-01/rapthor-command.log` |
| candidate | `base/rep-03` | `master` | 0 | `/tmp/rfjr/inputs/base/rep-03/master_dd_phase_plus_di_fulljones.parset` | `/tmp/rfjw/base/rep-03` | `/tmp/rfjr/base/rep-03/rapthor-command.log` |

## Comparison Summary

| Result | Ops | Records | FITS | Image HDUs | Table HDUs | H5 | Text | Diagnostics | Visuals |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pass | 5 | 5 | 7 | 6 | 1 | 3 | 10 | 1 | 5 |

## FITS Residual Metrics

| Product | Max Abs Delta | P99 Abs Delta | Residual RMS | RMS / Ref RMS | RMS / Ref MAD |
| --- | ---: | ---: | ---: | ---: | ---: |
| `field-MFS-dirty.fits` | 9.537e-07 | 8.941e-08 | 3.254e-08 | 1.882e-07 | 2.101e-07 |
| `field-MFS-image-pb-ast.fits` | 9.537e-07 | 7.823e-08 | 2.838e-08 | 3.796e-07 | 1.711e-06 |
| `field-MFS-image-pb.fits` | 9.537e-07 | 7.823e-08 | 2.838e-08 | 3.796e-07 | 1.711e-06 |
| `field-MFS-image.fits` | 9.537e-07 | 7.637e-08 | 2.813e-08 | 3.813e-07 | 1.731e-06 |
| `field-MFS-model-pb.fits` | 4.768e-07 | 0.000e+00 | 3.579e-10 | 2.063e-07 | n/a |
| `field-MFS-residual.fits` | 3.427e-07 | 5.867e-08 | 2.255e-08 | 1.303e-06 | 1.392e-06 |

## Image Diagnostics

| Operation | Sector | Field | Reference | Current | Delta | Relative Delta |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `image_1` | `sector_1` | `nsources` | 1.100e+01 | 1.100e+01 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `theoretical_rms` | 9.006e-03 | 9.006e-03 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `min_rms_flat_noise` | 8.312e-03 | 8.312e-03 | -2.794e-09 | -0.000% |
| `image_1` | `sector_1` | `median_rms_flat_noise` | 1.589e-02 | 1.589e-02 | 1.863e-09 | 0.000% |
| `image_1` | `sector_1` | `dynamic_range_global_flat_noise` | 5.505e+02 | 5.505e+02 | 2.998e-04 | 0.000% |
| `image_1` | `sector_1` | `min_rms_true_sky` | 8.439e-03 | 8.437e-03 | -1.611e-06 | -0.019% |
| `image_1` | `sector_1` | `median_rms_true_sky` | 1.622e-02 | 1.622e-02 | 5.588e-09 | 0.000% |
| `image_1` | `sector_1` | `dynamic_range_global_true_sky` | 5.423e+02 | 5.424e+02 | 1.035e-01 | 0.019% |

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

