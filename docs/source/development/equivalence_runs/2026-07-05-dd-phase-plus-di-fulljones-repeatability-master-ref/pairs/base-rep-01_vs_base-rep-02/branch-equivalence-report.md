# Rapthor Branch Equivalence

Scenario: `dd-phase-plus-di-fulljones-repeatability:base-rep-01_vs_base-rep-02`
Run root: `/tmp/rfjr/pairs/base-rep-01_vs_base-rep-02`

## Branch Runs

Repeatability pair: `base-rep-01_vs_base-rep-02` (base/rep-01 -> base/rep-02)

| Role | Repetition | Ref | Return Code | Parset | Work Dir | Log |
| --- | --- | --- | ---: | --- | --- | --- |
| reference | `base/rep-01` | `master` | 0 | `/tmp/rfjr/inputs/base/rep-01/master_dd_phase_plus_di_fulljones.parset` | `/tmp/rfjw/base/rep-01` | `/tmp/rfjr/base/rep-01/rapthor-command.log` |
| candidate | `base/rep-02` | `master` | 0 | `/tmp/rfjr/inputs/base/rep-02/master_dd_phase_plus_di_fulljones.parset` | `/tmp/rfjw/base/rep-02` | `/tmp/rfjr/base/rep-02/rapthor-command.log` |

## Comparison Summary

| Result | Ops | Records | FITS | Image HDUs | Table HDUs | H5 | Text | Diagnostics | Visuals |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pass | 5 | 5 | 7 | 6 | 1 | 3 | 10 | 1 | 5 |

## FITS Residual Metrics

| Product | Max Abs Delta | P99 Abs Delta | Residual RMS | RMS / Ref RMS | RMS / Ref MAD |
| --- | ---: | ---: | ---: | ---: | ---: |
| `field-MFS-image-pb-ast.fits` | 1.431e-06 | 8.009e-08 | 2.993e-08 | 4.004e-07 | 1.805e-06 |
| `field-MFS-image-pb.fits` | 1.431e-06 | 8.009e-08 | 2.993e-08 | 4.004e-07 | 1.805e-06 |
| `field-MFS-dirty.fits` | 9.537e-07 | 8.941e-08 | 3.340e-08 | 1.931e-07 | 2.156e-07 |
| `field-MFS-image.fits` | 9.537e-07 | 7.986e-08 | 2.891e-08 | 3.919e-07 | 1.779e-06 |
| `field-MFS-residual.fits` | 2.533e-07 | 5.821e-08 | 2.205e-08 | 1.274e-06 | 1.361e-06 |
| `field-MFS-model-pb.fits` | 2.384e-07 | 0.000e+00 | 1.729e-10 | 9.966e-08 | n/a |

## Image Diagnostics

| Operation | Sector | Field | Reference | Current | Delta | Relative Delta |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `image_1` | `sector_1` | `nsources` | 1.100e+01 | 1.100e+01 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `theoretical_rms` | 9.006e-03 | 9.006e-03 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `min_rms_flat_noise` | 8.312e-03 | 8.312e-03 | -1.742e-07 | -0.002% |
| `image_1` | `sector_1` | `median_rms_flat_noise` | 1.589e-02 | 1.589e-02 | 1.863e-09 | 0.000% |
| `image_1` | `sector_1` | `dynamic_range_global_flat_noise` | 5.505e+02 | 5.506e+02 | 1.154e-02 | 0.002% |
| `image_1` | `sector_1` | `min_rms_true_sky` | 8.439e-03 | 8.438e-03 | -1.294e-06 | -0.015% |
| `image_1` | `sector_1` | `median_rms_true_sky` | 1.622e-02 | 1.622e-02 | 1.863e-09 | 0.000% |
| `image_1` | `sector_1` | `dynamic_range_global_true_sky` | 5.423e+02 | 5.424e+02 | 8.314e-02 | 0.015% |

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

