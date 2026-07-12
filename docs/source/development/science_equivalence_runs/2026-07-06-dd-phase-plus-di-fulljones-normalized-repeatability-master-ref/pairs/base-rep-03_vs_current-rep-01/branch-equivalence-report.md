# Rapthor Branch Equivalence

Scenario: `dd-phase-plus-di-fulljones-normalized-repeatability:base-rep-03_vs_current-rep-01`
Run root: `/tmp/rfjnr/pairs/base-rep-03_vs_current-rep-01`

## Branch Runs

Repeatability pair: `base-rep-03_vs_current-rep-01` (base/rep-03 -> current/rep-01)

| Role | Repetition | Ref | Return Code | Parset | Work Dir | Log |
| --- | --- | --- | ---: | --- | --- | --- |
| reference | `base/rep-03` | `master` | 0 | `/tmp/rfjnr/inputs/base/rep-03/master_dd_phase_plus_di_fulljones.parset` | `/tmp/rfjnw/base/rep-03` | `/tmp/rfjnr/base/rep-03/rapthor-command.log` |
| candidate | `current/rep-01` | `current` | 0 | `/tmp/rfjnr/inputs/current/rep-01/current_dd_phase_plus_di_fulljones.parset` | `/tmp/rfjnw/current/rep-01` | `/tmp/rfjnr/current/rep-01/rapthor-command.log` |

## Comparison Summary

| Result | Ops | Records | FITS | Image HDUs | Table HDUs | H5 | Text | Diagnostics | Visuals |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pass | 5 | 5 | 7 | 6 | 1 | 3 | 10 | 1 | 5 |

## FITS Residual Metrics

| Product | Max Abs Delta | P99 Abs Delta | Residual RMS | RMS / Ref RMS | RMS / Ref MAD |
| --- | ---: | ---: | ---: | ---: | ---: |
| `field-MFS-image-pb-ast.fits` | 1.192e-06 | 7.963e-08 | 2.921e-08 | 3.908e-07 | 1.761e-06 |
| `field-MFS-image-pb.fits` | 1.192e-06 | 7.963e-08 | 2.921e-08 | 3.908e-07 | 1.761e-06 |
| `field-MFS-dirty.fits` | 9.537e-07 | 7.451e-08 | 2.820e-08 | 1.631e-07 | 1.820e-07 |
| `field-MFS-image.fits` | 9.537e-07 | 7.823e-08 | 2.860e-08 | 3.877e-07 | 1.760e-06 |
| `field-MFS-residual.fits` | 2.682e-07 | 5.635e-08 | 2.185e-08 | 1.262e-06 | 1.348e-06 |
| `field-MFS-model-pb.fits` | 2.384e-07 | 0.000e+00 | 1.837e-10 | 1.059e-07 | n/a |

## Image Diagnostics

| Operation | Sector | Field | Reference | Current | Delta | Relative Delta |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `image_1` | `sector_1` | `nsources` | 1.100e+01 | 1.100e+01 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `theoretical_rms` | 9.006e-03 | 9.006e-03 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `min_rms_flat_noise` | 8.312e-03 | 8.312e-03 | 3.111e-07 | 0.004% |
| `image_1` | `sector_1` | `median_rms_flat_noise` | 1.589e-02 | 1.589e-02 | 1.863e-09 | 0.000% |
| `image_1` | `sector_1` | `dynamic_range_global_flat_noise` | 5.506e+02 | 5.505e+02 | -2.066e-02 | -0.004% |
| `image_1` | `sector_1` | `min_rms_true_sky` | 8.438e-03 | 8.439e-03 | 1.004e-06 | 0.012% |
| `image_1` | `sector_1` | `median_rms_true_sky` | 1.622e-02 | 1.622e-02 | -3.725e-09 | -0.000% |
| `image_1` | `sector_1` | `dynamic_range_global_true_sky` | 5.423e+02 | 5.423e+02 | -6.458e-02 | -0.012% |

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


## Difference Classification

| Category | Disposition | Count | Examples | Recommendation |
| --- | --- | ---: | --- | --- |
| `output_record_auxiliary_artifacts` | warning | 2 | `calibrate_1`, `calibrate_di_1` | Keep as non-blocking when only plot filenames or known local intermediate aliases differ and final scientific products pass. |

## Warnings

- output-record auxiliary artifact basenames differ for calibrate_1
- output-record auxiliary artifact basenames differ for calibrate_di_1
