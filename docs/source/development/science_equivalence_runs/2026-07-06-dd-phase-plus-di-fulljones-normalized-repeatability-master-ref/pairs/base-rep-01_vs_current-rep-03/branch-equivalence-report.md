# Rapthor Branch Equivalence

Scenario: `dd-phase-plus-di-fulljones-normalized-repeatability:base-rep-01_vs_current-rep-03`
Run root: `/tmp/rfjnr/pairs/base-rep-01_vs_current-rep-03`

## Branch Runs

Repeatability pair: `base-rep-01_vs_current-rep-03` (base/rep-01 -> current/rep-03)

| Role | Repetition | Ref | Return Code | Parset | Work Dir | Log |
| --- | --- | --- | ---: | --- | --- | --- |
| reference | `base/rep-01` | `master` | 0 | `/tmp/rfjnr/inputs/base/rep-01/master_dd_phase_plus_di_fulljones.parset` | `/tmp/rfjnw/base/rep-01` | `/tmp/rfjnr/base/rep-01/rapthor-command.log` |
| candidate | `current/rep-03` | `current` | 0 | `/tmp/rfjnr/inputs/current/rep-03/current_dd_phase_plus_di_fulljones.parset` | `/tmp/rfjnw/current/rep-03` | `/tmp/rfjnr/current/rep-03/rapthor-command.log` |

## Comparison Summary

| Result | Ops | Records | FITS | Image HDUs | Table HDUs | H5 | Text | Diagnostics | Visuals |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pass | 5 | 5 | 7 | 6 | 1 | 3 | 10 | 1 | 5 |

## FITS Residual Metrics

| Product | Max Abs Delta | P99 Abs Delta | Residual RMS | RMS / Ref RMS | RMS / Ref MAD |
| --- | ---: | ---: | ---: | ---: | ---: |
| `field-MFS-image-pb-ast.fits` | 1.431e-06 | 7.544e-08 | 2.752e-08 | 3.681e-07 | 1.659e-06 |
| `field-MFS-image-pb.fits` | 1.431e-06 | 7.544e-08 | 2.752e-08 | 3.681e-07 | 1.659e-06 |
| `field-MFS-image.fits` | 1.431e-06 | 7.451e-08 | 2.718e-08 | 3.684e-07 | 1.672e-06 |
| `field-MFS-dirty.fits` | 9.537e-07 | 8.941e-08 | 2.961e-08 | 1.712e-07 | 1.911e-07 |
| `field-MFS-residual.fits` | 2.533e-07 | 5.402e-08 | 2.091e-08 | 1.208e-06 | 1.290e-06 |
| `field-MFS-model-pb.fits` | 2.384e-07 | 0.000e+00 | 1.425e-10 | 8.213e-08 | n/a |

## Image Diagnostics

| Operation | Sector | Field | Reference | Current | Delta | Relative Delta |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `image_1` | `sector_1` | `nsources` | 1.100e+01 | 1.100e+01 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `theoretical_rms` | 9.006e-03 | 9.006e-03 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `min_rms_flat_noise` | 8.311e-03 | 8.313e-03 | 1.488e-06 | 0.018% |
| `image_1` | `sector_1` | `median_rms_flat_noise` | 1.589e-02 | 1.589e-02 | 1.863e-09 | 0.000% |
| `image_1` | `sector_1` | `dynamic_range_global_flat_noise` | 5.506e+02 | 5.505e+02 | -9.846e-02 | -0.018% |
| `image_1` | `sector_1` | `min_rms_true_sky` | 8.438e-03 | 8.438e-03 | -1.770e-07 | -0.002% |
| `image_1` | `sector_1` | `median_rms_true_sky` | 1.622e-02 | 1.622e-02 | -1.118e-08 | -0.000% |
| `image_1` | `sector_1` | `dynamic_range_global_true_sky` | 5.423e+02 | 5.423e+02 | 1.137e-02 | 0.002% |

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
