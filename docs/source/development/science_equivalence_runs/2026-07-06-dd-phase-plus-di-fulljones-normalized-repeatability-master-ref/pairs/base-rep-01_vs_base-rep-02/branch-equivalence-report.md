# Rapthor Branch Equivalence

Scenario: `dd-phase-plus-di-fulljones-normalized-repeatability:base-rep-01_vs_base-rep-02`
Run root: `/tmp/rfjnr/pairs/base-rep-01_vs_base-rep-02`

## Branch Runs

Repeatability pair: `base-rep-01_vs_base-rep-02` (base/rep-01 -> base/rep-02)

| Role | Repetition | Ref | Return Code | Parset | Work Dir | Log |
| --- | --- | --- | ---: | --- | --- | --- |
| reference | `base/rep-01` | `master` | 0 | `/tmp/rfjnr/inputs/base/rep-01/master_dd_phase_plus_di_fulljones.parset` | `/tmp/rfjnw/base/rep-01` | `/tmp/rfjnr/base/rep-01/rapthor-command.log` |
| candidate | `base/rep-02` | `master` | 0 | `/tmp/rfjnr/inputs/base/rep-02/master_dd_phase_plus_di_fulljones.parset` | `/tmp/rfjnw/base/rep-02` | `/tmp/rfjnr/base/rep-02/rapthor-command.log` |

## Comparison Summary

| Result | Ops | Records | FITS | Image HDUs | Table HDUs | H5 | Text | Diagnostics | Visuals |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pass | 5 | 5 | 7 | 6 | 1 | 3 | 10 | 1 | 5 |

## FITS Residual Metrics

| Product | Max Abs Delta | P99 Abs Delta | Residual RMS | RMS / Ref RMS | RMS / Ref MAD |
| --- | ---: | ---: | ---: | ---: | ---: |
| `field-MFS-image-pb-ast.fits` | 9.537e-07 | 7.823e-08 | 2.819e-08 | 3.771e-07 | 1.700e-06 |
| `field-MFS-image-pb.fits` | 9.537e-07 | 7.823e-08 | 2.819e-08 | 3.771e-07 | 1.700e-06 |
| `field-MFS-image.fits` | 9.537e-07 | 7.637e-08 | 2.791e-08 | 3.783e-07 | 1.717e-06 |
| `field-MFS-dirty.fits` | 7.153e-07 | 7.451e-08 | 2.834e-08 | 1.639e-07 | 1.830e-07 |
| `field-MFS-model-pb.fits` | 2.384e-07 | 0.000e+00 | 1.900e-10 | 1.096e-07 | n/a |
| `field-MFS-residual.fits` | 1.937e-07 | 5.588e-08 | 2.140e-08 | 1.237e-06 | 1.321e-06 |

## Image Diagnostics

| Operation | Sector | Field | Reference | Current | Delta | Relative Delta |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `image_1` | `sector_1` | `nsources` | 1.100e+01 | 1.100e+01 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `theoretical_rms` | 9.006e-03 | 9.006e-03 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `min_rms_flat_noise` | 8.311e-03 | 8.312e-03 | 9.779e-07 | 0.012% |
| `image_1` | `sector_1` | `median_rms_flat_noise` | 1.589e-02 | 1.589e-02 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `dynamic_range_global_flat_noise` | 5.506e+02 | 5.505e+02 | -6.478e-02 | -0.012% |
| `image_1` | `sector_1` | `min_rms_true_sky` | 8.438e-03 | 8.440e-03 | 1.464e-06 | 0.017% |
| `image_1` | `sector_1` | `median_rms_true_sky` | 1.622e-02 | 1.622e-02 | -1.863e-08 | -0.000% |
| `image_1` | `sector_1` | `dynamic_range_global_true_sky` | 5.423e+02 | 5.422e+02 | -9.402e-02 | -0.017% |

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

