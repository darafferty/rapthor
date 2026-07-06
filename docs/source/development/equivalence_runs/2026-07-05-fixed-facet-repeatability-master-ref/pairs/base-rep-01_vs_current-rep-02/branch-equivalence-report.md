# Rapthor Branch Equivalence

Scenario: `fixed-facet-carryover-repeatability:base-rep-01_vs_current-rep-02`
Run root: `/tmp/rffr/pairs/base-rep-01_vs_current-rep-02`

## Branch Runs

Repeatability pair: `base-rep-01_vs_current-rep-02` (base/rep-01 -> current/rep-02)

| Role | Repetition | Ref | Return Code | Parset | Work Dir | Log |
| --- | --- | --- | ---: | --- | --- | --- |
| reference | `base/rep-01` | `master` | 0 | `/tmp/rffr/inputs/base/rep-01/master_fixed_facet_carryover.parset` | `/tmp/rffw/base/rep-01` | `/tmp/rffr/base/rep-01/rapthor-command.log` |
| candidate | `current/rep-02` | `current` | 0 | `/tmp/rffr/inputs/current/rep-02/current_fixed_facet_carryover.parset` | `/tmp/rffw/current/rep-02` | `/tmp/rffr/current/rep-02/rapthor-command.log` |

## Comparison Summary

| Result | Ops | Records | FITS | Image HDUs | Table HDUs | H5 | Text | Diagnostics | Visuals |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fail | 4 | 4 | 7 | 6 | 1 | 4 | 16 | 1 | 7 |

## FITS Residual Metrics

| Product | Max Abs Delta | P99 Abs Delta | Residual RMS | RMS / Ref RMS | RMS / Ref MAD |
| --- | ---: | ---: | ---: | ---: | ---: |
| `field-MFS-model-pb.fits.fz` | 2.348e-01 | 0.000e+00 | 3.437e-03 | 1.135e+00 | n/a |
| `field-MFS-image-pb-ast.fits.fz` | 1.809e-02 | 2.854e-06 | 7.633e-06 | 9.061e-05 | 1.830e-04 |
| `field-MFS-image-pb.fits.fz` | 1.809e-02 | 2.850e-06 | 7.632e-06 | 9.060e-05 | 1.830e-04 |
| `field-MFS-residual.fits.fz` | 1.773e-02 | 2.775e-06 | 7.477e-06 | 1.793e-04 | 1.835e-04 |
| `field-MFS-image.fits.fz` | 1.773e-02 | 2.794e-06 | 7.478e-06 | 9.011e-05 | 1.830e-04 |
| `field-MFS-dirty.fits.fz` | 1.463e-05 | 9.939e-06 | 4.155e-06 | 2.405e-05 | 2.683e-05 |

## Image Diagnostics

| Operation | Sector | Field | Reference | Current | Delta | Relative Delta |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `image_1` | `sector_1` | `nsources` | 1.000e+01 | 1.000e+01 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `theoretical_rms` | 9.006e-03 | 9.006e-03 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `min_rms_flat_noise` | 1.987e-02 | 1.987e-02 | 3.166e-08 | 0.000% |
| `image_1` | `sector_1` | `median_rms_flat_noise` | 3.982e-02 | 3.982e-02 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `dynamic_range_global_flat_noise` | 2.296e+02 | 2.296e+02 | -3.659e-04 | -0.000% |
| `image_1` | `sector_1` | `min_rms_true_sky` | 2.038e-02 | 2.038e-02 | -2.235e-08 | -0.000% |
| `image_1` | `sector_1` | `median_rms_true_sky` | 4.061e-02 | 4.061e-02 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `dynamic_range_global_true_sky` | 2.239e+02 | 2.239e+02 | 2.689e-04 | 0.000% |

## Visual Comparisons

### Image: `image_1/field-MFS-image-pb-ast.fits.fz`

![image_1/field-MFS-image-pb-ast.fits.fz](visual-comparisons/images/image_1-field-mfs-image-pb-ast.fits.fz.png)

### Image: `image_1/field-MFS-image-pb.fits.fz`

![image_1/field-MFS-image-pb.fits.fz](visual-comparisons/images/image_1-field-mfs-image-pb.fits.fz.png)

### Image: `image_1/field-MFS-residual.fits.fz`

![image_1/field-MFS-residual.fits.fz](visual-comparisons/images/image_1-field-mfs-residual.fits.fz.png)

### Solution: `calibrate_1/fast_phase_dir[Patch_rich_centre].png`

![calibrate_1/fast_phase_dir[Patch_rich_centre].png](visual-comparisons/solutions/calibrate_1-fast_phase_dir-patch_rich_centre-.png.png)

### Solution: `calibrate_1/medium1_phase_dir[Patch_rich_centre].png`

![calibrate_1/medium1_phase_dir[Patch_rich_centre].png](visual-comparisons/solutions/calibrate_1-medium1_phase_dir-patch_rich_centre-.png.png)

### Solution: `calibrate_2/fast_phase_dir[Patch_rich_centre].png`

![calibrate_2/fast_phase_dir[Patch_rich_centre].png](visual-comparisons/solutions/calibrate_2-fast_phase_dir-patch_rich_centre-.png.png)

### Solution: `calibrate_2/medium1_phase_dir[Patch_rich_centre].png`

![calibrate_2/medium1_phase_dir[Patch_rich_centre].png](visual-comparisons/solutions/calibrate_2-medium1_phase_dir-patch_rich_centre-.png.png)


## Warnings

- output-record summary differs for calibrate_1
- output-record summary differs for calibrate_2

## Failures

- FITS image pixels differ for field-MFS-dirty.fits.fz: max_abs_delta=1.4632940292358398e-05, p99_abs_delta=9.939074516296387e-06, residual_rms=4.154633590392317e-06
- FITS image pixels differ for field-MFS-image-pb-ast.fits.fz: max_abs_delta=0.018093805061653256, p99_abs_delta=2.853572368621826e-06, residual_rms=7.633018205444095e-06
- FITS image pixels differ for field-MFS-image-pb.fits.fz: max_abs_delta=0.018092046258971095, p99_abs_delta=2.8498470783233643e-06, residual_rms=7.632218418855963e-06
- FITS image pixels differ for field-MFS-image.fits.fz: max_abs_delta=0.017727128113619983, p99_abs_delta=2.7939677238464355e-06, residual_rms=7.478311753036764e-06
- FITS std differs for field-MFS-model-pb.fits.fz: 0.0030284670880311054 != 0.0030178976105584123
- FITS rms differs for field-MFS-model-pb.fits.fz: 0.00302847139244273 != 0.003017901545826119
- FITS max differs for field-MFS-model-pb.fits.fz: 1.6887661218643188 != 1.6711139678955078
- FITS image pixels differ for field-MFS-model-pb.fits.fz: max_abs_delta=0.23480141162872314, p99_abs_delta=0.0, residual_rms=0.003436614888552825
- FITS image pixels differ for field-MFS-residual.fits.fz: max_abs_delta=0.017727709258906543, p99_abs_delta=2.775341272354126e-06, residual_rms=7.477378188858505e-06
- text product differs for sector_1_facets_ds9.reg
