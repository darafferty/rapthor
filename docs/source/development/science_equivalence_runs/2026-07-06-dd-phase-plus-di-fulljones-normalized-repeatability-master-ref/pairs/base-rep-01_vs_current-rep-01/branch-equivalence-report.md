# Rapthor Branch Equivalence

Scenario: `dd-phase-plus-di-fulljones-normalized-repeatability:base-rep-01_vs_current-rep-01`
Run root: `/tmp/rfjnr/pairs/base-rep-01_vs_current-rep-01`

## Branch Runs

Repeatability pair: `base-rep-01_vs_current-rep-01` (base/rep-01 -> current/rep-01)

| Role | Repetition | Ref | Return Code | Parset | Work Dir | Log |
| --- | --- | --- | ---: | --- | --- | --- |
| reference | `base/rep-01` | `master` | 0 | `/tmp/rfjnr/inputs/base/rep-01/master_dd_phase_plus_di_fulljones.parset` | `/tmp/rfjnw/base/rep-01` | `/tmp/rfjnr/base/rep-01/rapthor-command.log` |
| candidate | `current/rep-01` | `current` | 0 | `/tmp/rfjnr/inputs/current/rep-01/current_dd_phase_plus_di_fulljones.parset` | `/tmp/rfjnw/current/rep-01` | `/tmp/rfjnr/current/rep-01/rapthor-command.log` |

## Comparison Summary

| Result | Ops | Records | FITS | Image HDUs | Table HDUs | H5 | Text | Diagnostics | Visuals |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pass | 5 | 5 | 7 | 6 | 1 | 3 | 10 | 1 | 5 |

## FITS Residual Metrics

| Product | Max Abs Delta | P99 Abs Delta | Residual RMS | RMS / Ref RMS | RMS / Ref MAD |
| --- | ---: | ---: | ---: | ---: | ---: |
| `field-MFS-image.fits` | 1.431e-06 | 7.637e-08 | 2.789e-08 | 3.781e-07 | 1.716e-06 |
| `field-MFS-dirty.fits` | 9.537e-07 | 8.941e-08 | 3.068e-08 | 1.774e-07 | 1.981e-07 |
| `field-MFS-image-pb-ast.fits` | 9.537e-07 | 7.823e-08 | 2.825e-08 | 3.779e-07 | 1.703e-06 |
| `field-MFS-image-pb.fits` | 9.537e-07 | 7.823e-08 | 2.825e-08 | 3.779e-07 | 1.703e-06 |
| `field-MFS-residual.fits` | 3.278e-07 | 5.588e-08 | 2.146e-08 | 1.240e-06 | 1.324e-06 |
| `field-MFS-model-pb.fits` | 2.384e-07 | 0.000e+00 | 2.140e-10 | 1.234e-07 | n/a |

## Image Diagnostics

| Operation | Sector | Field | Reference | Current | Delta | Relative Delta |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `image_1` | `sector_1` | `nsources` | 1.100e+01 | 1.100e+01 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `theoretical_rms` | 9.006e-03 | 9.006e-03 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `min_rms_flat_noise` | 8.311e-03 | 8.312e-03 | 9.807e-07 | 0.012% |
| `image_1` | `sector_1` | `median_rms_flat_noise` | 1.589e-02 | 1.589e-02 | 1.863e-09 | 0.000% |
| `image_1` | `sector_1` | `dynamic_range_global_flat_noise` | 5.506e+02 | 5.505e+02 | -6.490e-02 | -0.012% |
| `image_1` | `sector_1` | `min_rms_true_sky` | 8.438e-03 | 8.439e-03 | 6.612e-07 | 0.008% |
| `image_1` | `sector_1` | `median_rms_true_sky` | 1.622e-02 | 1.622e-02 | -7.451e-09 | -0.000% |
| `image_1` | `sector_1` | `dynamic_range_global_true_sky` | 5.423e+02 | 5.423e+02 | -4.244e-02 | -0.008% |

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
