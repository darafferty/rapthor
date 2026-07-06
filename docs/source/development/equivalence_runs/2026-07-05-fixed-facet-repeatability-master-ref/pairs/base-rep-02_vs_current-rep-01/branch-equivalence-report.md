# Rapthor Branch Equivalence

Scenario: `fixed-facet-carryover-repeatability:base-rep-02_vs_current-rep-01`
Run root: `/tmp/rffr/pairs/base-rep-02_vs_current-rep-01`

## Branch Runs

Repeatability pair: `base-rep-02_vs_current-rep-01` (base/rep-02 -> current/rep-01)

| Role | Repetition | Ref | Return Code | Parset | Work Dir | Log |
| --- | --- | --- | ---: | --- | --- | --- |
| reference | `base/rep-02` | `master` | 0 | `/tmp/rffr/inputs/base/rep-02/master_fixed_facet_carryover.parset` | `/tmp/rffw/base/rep-02` | `/tmp/rffr/base/rep-02/rapthor-command.log` |
| candidate | `current/rep-01` | `current` | 0 | `/tmp/rffr/inputs/current/rep-01/current_fixed_facet_carryover.parset` | `/tmp/rffw/current/rep-01` | `/tmp/rffr/current/rep-01/rapthor-command.log` |

## Comparison Summary

| Result | Ops | Records | FITS | Image HDUs | Table HDUs | H5 | Text | Diagnostics | Visuals |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fail | 4 | 4 | 7 | 6 | 1 | 4 | 16 | 1 | 7 |

## FITS Residual Metrics

| Product | Max Abs Delta | P99 Abs Delta | Residual RMS | RMS / Ref RMS | RMS / Ref MAD |
| --- | ---: | ---: | ---: | ---: | ---: |
| `field-MFS-model-pb.fits.fz` | 2.321e-01 | 0.000e+00 | 3.450e-03 | 1.141e+00 | n/a |
| `field-MFS-dirty.fits.fz` | 1.478e-05 | 9.924e-06 | 4.154e-06 | 2.404e-05 | 2.682e-05 |
| `field-MFS-image-pb.fits.fz` | 4.331e-06 | 2.850e-06 | 1.193e-06 | 1.417e-05 | 2.861e-05 |
| `field-MFS-image-pb-ast.fits.fz` | 4.303e-06 | 2.848e-06 | 1.193e-06 | 1.416e-05 | 2.859e-05 |
| `field-MFS-residual.fits.fz` | 4.239e-06 | 2.773e-06 | 1.161e-06 | 2.785e-05 | 2.850e-05 |
| `field-MFS-image.fits.fz` | 4.217e-06 | 2.794e-06 | 1.169e-06 | 1.409e-05 | 2.861e-05 |

## Image Diagnostics

| Operation | Sector | Field | Reference | Current | Delta | Relative Delta |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `image_1` | `sector_1` | `nsources` | 1.000e+01 | 1.000e+01 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `theoretical_rms` | 9.006e-03 | 9.006e-03 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `min_rms_flat_noise` | 1.987e-02 | 1.987e-02 | 5.588e-09 | 0.000% |
| `image_1` | `sector_1` | `median_rms_flat_noise` | 3.982e-02 | 3.982e-02 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `dynamic_range_global_flat_noise` | 2.296e+02 | 2.296e+02 | -8.857e-05 | -0.000% |
| `image_1` | `sector_1` | `min_rms_true_sky` | 2.038e-02 | 2.038e-02 | 2.794e-08 | 0.000% |
| `image_1` | `sector_1` | `median_rms_true_sky` | 4.061e-02 | 4.061e-02 | 7.451e-09 | 0.000% |
| `image_1` | `sector_1` | `dynamic_range_global_true_sky` | 2.239e+02 | 2.239e+02 | -2.835e-04 | -0.000% |

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

- FITS image pixels differ for field-MFS-dirty.fits.fz: max_abs_delta=1.4781951904296875e-05, p99_abs_delta=9.924173355102539e-06, residual_rms=4.154365297826479e-06
- FITS image pixels differ for field-MFS-image-pb-ast.fits.fz: max_abs_delta=4.302710294723511e-06, p99_abs_delta=2.8479844331741333e-06, residual_rms=1.1925494903608718e-06
- FITS image pixels differ for field-MFS-image-pb.fits.fz: max_abs_delta=4.330649971961975e-06, p99_abs_delta=2.8498470783233643e-06, residual_rms=1.1933072159510533e-06
- FITS image pixels differ for field-MFS-image.fits.fz: max_abs_delta=4.217028617858887e-06, p99_abs_delta=2.7939677238464355e-06, residual_rms=1.1694192628878476e-06
- FITS std differs for field-MFS-model-pb.fits.fz: 0.0030227072249057975 != 0.003051534233221859
- FITS rms differs for field-MFS-model-pb.fits.fz: 0.0030227113173986993 != 0.0030515385861614027
- FITS max differs for field-MFS-model-pb.fits.fz: 1.6860357522964478 != 1.6721535921096802
- FITS image pixels differ for field-MFS-model-pb.fits.fz: max_abs_delta=0.2320513129234314, p99_abs_delta=0.0, residual_rms=0.003449619837314193
- FITS image pixels differ for field-MFS-residual.fits.fz: max_abs_delta=4.239380359649658e-06, p99_abs_delta=2.773478627204895e-06, residual_rms=1.1613526088883055e-06
- text product differs for sector_1_facets_ds9.reg
