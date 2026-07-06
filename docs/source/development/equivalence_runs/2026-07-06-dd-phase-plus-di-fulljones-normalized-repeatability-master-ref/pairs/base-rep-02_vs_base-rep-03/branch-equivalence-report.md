# Rapthor Branch Equivalence

Scenario: `dd-phase-plus-di-fulljones-normalized-repeatability:base-rep-02_vs_base-rep-03`
Run root: `/tmp/rfjnr/pairs/base-rep-02_vs_base-rep-03`

## Branch Runs

Repeatability pair: `base-rep-02_vs_base-rep-03` (base/rep-02 -> base/rep-03)

| Role | Repetition | Ref | Return Code | Parset | Work Dir | Log |
| --- | --- | --- | ---: | --- | --- | --- |
| reference | `base/rep-02` | `master` | 0 | `/tmp/rfjnr/inputs/base/rep-02/master_dd_phase_plus_di_fulljones.parset` | `/tmp/rfjnw/base/rep-02` | `/tmp/rfjnr/base/rep-02/rapthor-command.log` |
| candidate | `base/rep-03` | `master` | 0 | `/tmp/rfjnr/inputs/base/rep-03/master_dd_phase_plus_di_fulljones.parset` | `/tmp/rfjnw/base/rep-03` | `/tmp/rfjnr/base/rep-03/rapthor-command.log` |

## Comparison Summary

| Result | Ops | Records | FITS | Image HDUs | Table HDUs | H5 | Text | Diagnostics | Visuals |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pass | 5 | 5 | 7 | 6 | 1 | 3 | 10 | 1 | 5 |

## FITS Residual Metrics

| Product | Max Abs Delta | P99 Abs Delta | Residual RMS | RMS / Ref RMS | RMS / Ref MAD |
| --- | ---: | ---: | ---: | ---: | ---: |
| `field-MFS-dirty.fits` | 9.537e-07 | 8.941e-08 | 3.097e-08 | 1.791e-07 | 1.999e-07 |
| `field-MFS-image-pb-ast.fits` | 9.537e-07 | 8.114e-08 | 2.973e-08 | 3.978e-07 | 1.793e-06 |
| `field-MFS-image-pb.fits` | 9.537e-07 | 8.114e-08 | 2.973e-08 | 3.978e-07 | 1.793e-06 |
| `field-MFS-image.fits` | 9.537e-07 | 8.009e-08 | 2.928e-08 | 3.970e-07 | 1.802e-06 |
| `field-MFS-model-pb.fits` | 2.384e-07 | 0.000e+00 | 1.705e-10 | 9.830e-08 | n/a |
| `field-MFS-residual.fits` | 2.235e-07 | 5.821e-08 | 2.213e-08 | 1.279e-06 | 1.366e-06 |

## Image Diagnostics

| Operation | Sector | Field | Reference | Current | Delta | Relative Delta |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `image_1` | `sector_1` | `nsources` | 1.100e+01 | 1.100e+01 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `theoretical_rms` | 9.006e-03 | 9.006e-03 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `min_rms_flat_noise` | 8.312e-03 | 8.312e-03 | -3.083e-07 | -0.004% |
| `image_1` | `sector_1` | `median_rms_flat_noise` | 1.589e-02 | 1.589e-02 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `dynamic_range_global_flat_noise` | 5.505e+02 | 5.506e+02 | 2.053e-02 | 0.004% |
| `image_1` | `sector_1` | `min_rms_true_sky` | 8.440e-03 | 8.438e-03 | -1.807e-06 | -0.021% |
| `image_1` | `sector_1` | `median_rms_true_sky` | 1.622e-02 | 1.622e-02 | 1.490e-08 | 0.000% |
| `image_1` | `sector_1` | `dynamic_range_global_true_sky` | 5.422e+02 | 5.423e+02 | 1.162e-01 | 0.021% |

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

