# Rapthor Branch Equivalence

Scenario: `fixed-facet-carryover-repeatability:base-rep-02_vs_current-rep-02`
Run root: `/tmp/rffr/pairs/base-rep-02_vs_current-rep-02`

## Branch Runs

Repeatability pair: `base-rep-02_vs_current-rep-02` (base/rep-02 -> current/rep-02)

| Role | Repetition | Ref | Return Code | Parset | Work Dir | Log |
| --- | --- | --- | ---: | --- | --- | --- |
| reference | `base/rep-02` | `master` | 0 | `/tmp/rffr/inputs/base/rep-02/master_fixed_facet_carryover.parset` | `/tmp/rffw/base/rep-02` | `/tmp/rffr/base/rep-02/rapthor-command.log` |
| candidate | `current/rep-02` | `current` | 0 | `/tmp/rffr/inputs/current/rep-02/current_fixed_facet_carryover.parset` | `/tmp/rffw/current/rep-02` | `/tmp/rffr/current/rep-02/rapthor-command.log` |

## Comparison Summary

| Result | Ops | Records | FITS | Image HDUs | Table HDUs | H5 | Text | Diagnostics | Visuals |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fail | 4 | 4 | 7 | 6 | 1 | 4 | 16 | 1 | 7 |

## FITS Residual Metrics

| Product | Max Abs Delta | P99 Abs Delta | Residual RMS | RMS / Ref RMS | RMS / Ref MAD |
| --- | ---: | ---: | ---: | ---: | ---: |
| `field-MFS-model-pb.fits.fz` | 2.315e-01 | 0.000e+00 | 3.461e-03 | 1.145e+00 | n/a |
| `field-MFS-image-pb-ast.fits.fz` | 1.809e-02 | 2.850e-06 | 7.633e-06 | 9.061e-05 | 1.830e-04 |
| `field-MFS-image-pb.fits.fz` | 1.809e-02 | 2.848e-06 | 7.632e-06 | 9.060e-05 | 1.830e-04 |
| `field-MFS-residual.fits.fz` | 1.773e-02 | 2.772e-06 | 7.477e-06 | 1.793e-04 | 1.835e-04 |
| `field-MFS-image.fits.fz` | 1.773e-02 | 2.794e-06 | 7.478e-06 | 9.011e-05 | 1.830e-04 |
| `field-MFS-dirty.fits.fz` | 1.463e-05 | 9.928e-06 | 4.153e-06 | 2.404e-05 | 2.682e-05 |

## Image Diagnostics

| Operation | Sector | Field | Reference | Current | Delta | Relative Delta |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `image_1` | `sector_1` | `nsources` | 1.000e+01 | 1.000e+01 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `theoretical_rms` | 9.006e-03 | 9.006e-03 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `min_rms_flat_noise` | 1.987e-02 | 1.987e-02 | -1.676e-08 | -0.000% |
| `image_1` | `sector_1` | `median_rms_flat_noise` | 3.982e-02 | 3.982e-02 | 7.451e-09 | 0.000% |
| `image_1` | `sector_1` | `dynamic_range_global_flat_noise` | 2.296e+02 | 2.296e+02 | 1.697e-04 | 0.000% |
| `image_1` | `sector_1` | `min_rms_true_sky` | 2.038e-02 | 2.038e-02 | -5.588e-08 | -0.000% |
| `image_1` | `sector_1` | `median_rms_true_sky` | 4.061e-02 | 4.061e-02 | 7.451e-09 | 0.000% |
| `image_1` | `sector_1` | `dynamic_range_global_true_sky` | 2.239e+02 | 2.239e+02 | 6.607e-04 | 0.000% |

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

- FITS image pixels differ for field-MFS-dirty.fits.fz: max_abs_delta=1.4625489711761475e-05, p99_abs_delta=9.927898645401001e-06, residual_rms=4.153482534912896e-06
- FITS image pixels differ for field-MFS-image-pb-ast.fits.fz: max_abs_delta=0.01809370261617005, p99_abs_delta=2.8498470783233643e-06, residual_rms=7.632830166118527e-06
- FITS image pixels differ for field-MFS-image-pb.fits.fz: max_abs_delta=0.018091943813487887, p99_abs_delta=2.8479844331741333e-06, residual_rms=7.632156967277835e-06
- FITS image pixels differ for field-MFS-image.fits.fz: max_abs_delta=0.01772711507510394, p99_abs_delta=2.7939677238464355e-06, residual_rms=7.478325336392258e-06
- FITS std differs for field-MFS-model-pb.fits.fz: 0.0030227072249057975 != 0.0030178976105584123
- FITS rms differs for field-MFS-model-pb.fits.fz: 0.0030227113173986993 != 0.003017901545826119
- FITS max differs for field-MFS-model-pb.fits.fz: 1.6860357522964478 != 1.6711139678955078
- FITS image pixels differ for field-MFS-model-pb.fits.fz: max_abs_delta=0.2314896062016487, p99_abs_delta=0.0, residual_rms=0.0034606536230676783
- FITS image pixels differ for field-MFS-residual.fits.fz: max_abs_delta=0.017727988655678928, p99_abs_delta=2.771615982055664e-06, residual_rms=7.4774597523997586e-06
- text product differs for sector_1_facets_ds9.reg
