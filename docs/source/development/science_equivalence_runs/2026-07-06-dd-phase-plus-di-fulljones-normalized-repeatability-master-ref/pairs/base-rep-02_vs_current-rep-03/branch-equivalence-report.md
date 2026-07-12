# Rapthor Branch Equivalence

Scenario: `dd-phase-plus-di-fulljones-normalized-repeatability:base-rep-02_vs_current-rep-03`
Run root: `/tmp/rfjnr/pairs/base-rep-02_vs_current-rep-03`

## Branch Runs

Repeatability pair: `base-rep-02_vs_current-rep-03` (base/rep-02 -> current/rep-03)

| Role | Repetition | Ref | Return Code | Parset | Work Dir | Log |
| --- | --- | --- | ---: | --- | --- | --- |
| reference | `base/rep-02` | `master` | 0 | `/tmp/rfjnr/inputs/base/rep-02/master_dd_phase_plus_di_fulljones.parset` | `/tmp/rfjnw/base/rep-02` | `/tmp/rfjnr/base/rep-02/rapthor-command.log` |
| candidate | `current/rep-03` | `current` | 0 | `/tmp/rfjnr/inputs/current/rep-03/current_dd_phase_plus_di_fulljones.parset` | `/tmp/rfjnw/current/rep-03` | `/tmp/rfjnr/current/rep-03/rapthor-command.log` |

## Comparison Summary

| Result | Ops | Records | FITS | Image HDUs | Table HDUs | H5 | Text | Diagnostics | Visuals |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pass | 5 | 5 | 7 | 6 | 1 | 3 | 10 | 1 | 5 |

## FITS Residual Metrics

| Product | Max Abs Delta | P99 Abs Delta | Residual RMS | RMS / Ref RMS | RMS / Ref MAD |
| --- | ---: | ---: | ---: | ---: | ---: |
| `field-MFS-dirty.fits` | 9.537e-07 | 8.196e-08 | 2.903e-08 | 1.679e-07 | 1.874e-07 |
| `field-MFS-image-pb-ast.fits` | 9.537e-07 | 7.078e-08 | 2.576e-08 | 3.446e-07 | 1.553e-06 |
| `field-MFS-image-pb.fits` | 9.537e-07 | 7.078e-08 | 2.576e-08 | 3.446e-07 | 1.553e-06 |
| `field-MFS-image.fits` | 9.537e-07 | 7.031e-08 | 2.573e-08 | 3.488e-07 | 1.583e-06 |
| `field-MFS-model-pb.fits` | 1.788e-07 | 0.000e+00 | 1.463e-10 | 8.436e-08 | n/a |
| `field-MFS-residual.fits` | 1.490e-07 | 5.309e-08 | 2.044e-08 | 1.181e-06 | 1.261e-06 |

## Image Diagnostics

| Operation | Sector | Field | Reference | Current | Delta | Relative Delta |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `image_1` | `sector_1` | `nsources` | 1.100e+01 | 1.100e+01 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `theoretical_rms` | 9.006e-03 | 9.006e-03 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `min_rms_flat_noise` | 8.312e-03 | 8.313e-03 | 5.104e-07 | 0.006% |
| `image_1` | `sector_1` | `median_rms_flat_noise` | 1.589e-02 | 1.589e-02 | 1.863e-09 | 0.000% |
| `image_1` | `sector_1` | `dynamic_range_global_flat_noise` | 5.505e+02 | 5.505e+02 | -3.369e-02 | -0.006% |
| `image_1` | `sector_1` | `min_rms_true_sky` | 8.440e-03 | 8.438e-03 | -1.641e-06 | -0.019% |
| `image_1` | `sector_1` | `median_rms_true_sky` | 1.622e-02 | 1.622e-02 | 7.451e-09 | 0.000% |
| `image_1` | `sector_1` | `dynamic_range_global_true_sky` | 5.422e+02 | 5.423e+02 | 1.054e-01 | 0.019% |

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
