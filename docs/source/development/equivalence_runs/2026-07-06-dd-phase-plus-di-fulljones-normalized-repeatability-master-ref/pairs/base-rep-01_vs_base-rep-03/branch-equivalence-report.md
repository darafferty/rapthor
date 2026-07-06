# Rapthor Branch Equivalence

Scenario: `dd-phase-plus-di-fulljones-normalized-repeatability:base-rep-01_vs_base-rep-03`
Run root: `/tmp/rfjnr/pairs/base-rep-01_vs_base-rep-03`

## Branch Runs

Repeatability pair: `base-rep-01_vs_base-rep-03` (base/rep-01 -> base/rep-03)

| Role | Repetition | Ref | Return Code | Parset | Work Dir | Log |
| --- | --- | --- | ---: | --- | --- | --- |
| reference | `base/rep-01` | `master` | 0 | `/tmp/rfjnr/inputs/base/rep-01/master_dd_phase_plus_di_fulljones.parset` | `/tmp/rfjnw/base/rep-01` | `/tmp/rfjnr/base/rep-01/rapthor-command.log` |
| candidate | `base/rep-03` | `master` | 0 | `/tmp/rfjnr/inputs/base/rep-03/master_dd_phase_plus_di_fulljones.parset` | `/tmp/rfjnw/base/rep-03` | `/tmp/rfjnr/base/rep-03/rapthor-command.log` |

## Comparison Summary

| Result | Ops | Records | FITS | Image HDUs | Table HDUs | H5 | Text | Diagnostics | Visuals |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pass | 5 | 5 | 7 | 6 | 1 | 3 | 10 | 1 | 5 |

## FITS Residual Metrics

| Product | Max Abs Delta | P99 Abs Delta | Residual RMS | RMS / Ref RMS | RMS / Ref MAD |
| --- | ---: | ---: | ---: | ---: | ---: |
| `field-MFS-dirty.fits` | 9.537e-07 | 8.941e-08 | 3.157e-08 | 1.826e-07 | 2.038e-07 |
| `field-MFS-image-pb-ast.fits` | 9.537e-07 | 7.637e-08 | 2.799e-08 | 3.744e-07 | 1.688e-06 |
| `field-MFS-image-pb.fits` | 9.537e-07 | 7.637e-08 | 2.799e-08 | 3.744e-07 | 1.688e-06 |
| `field-MFS-image.fits` | 9.537e-07 | 7.451e-08 | 2.730e-08 | 3.701e-07 | 1.680e-06 |
| `field-MFS-model-pb.fits` | 4.768e-07 | 0.000e+00 | 2.711e-10 | 1.563e-07 | n/a |
| `field-MFS-residual.fits` | 3.427e-07 | 5.797e-08 | 2.256e-08 | 1.304e-06 | 1.392e-06 |

## Image Diagnostics

| Operation | Sector | Field | Reference | Current | Delta | Relative Delta |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `image_1` | `sector_1` | `nsources` | 1.100e+01 | 1.100e+01 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `theoretical_rms` | 9.006e-03 | 9.006e-03 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `min_rms_flat_noise` | 8.311e-03 | 8.312e-03 | 6.696e-07 | 0.008% |
| `image_1` | `sector_1` | `median_rms_flat_noise` | 1.589e-02 | 1.589e-02 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `dynamic_range_global_flat_noise` | 5.506e+02 | 5.506e+02 | -4.424e-02 | -0.008% |
| `image_1` | `sector_1` | `min_rms_true_sky` | 8.438e-03 | 8.438e-03 | -3.427e-07 | -0.004% |
| `image_1` | `sector_1` | `median_rms_true_sky` | 1.622e-02 | 1.622e-02 | -3.725e-09 | -0.000% |
| `image_1` | `sector_1` | `dynamic_range_global_true_sky` | 5.423e+02 | 5.423e+02 | 2.214e-02 | 0.004% |

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

