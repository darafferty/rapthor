# Rapthor Branch Equivalence

Scenario: `fixed-facet-carryover-repeatability:base-rep-03_vs_current-rep-03`
Run root: `/tmp/rffr/pairs/base-rep-03_vs_current-rep-03`

## Branch Runs

Repeatability pair: `base-rep-03_vs_current-rep-03` (base/rep-03 -> current/rep-03)

| Role | Repetition | Ref | Return Code | Parset | Work Dir | Log |
| --- | --- | --- | ---: | --- | --- | --- |
| reference | `base/rep-03` | `master` | 0 | `/tmp/rffr/inputs/base/rep-03/master_fixed_facet_carryover.parset` | `/tmp/rffw/base/rep-03` | `/tmp/rffr/base/rep-03/rapthor-command.log` |
| candidate | `current/rep-03` | `current` | 0 | `/tmp/rffr/inputs/current/rep-03/current_fixed_facet_carryover.parset` | `/tmp/rffw/current/rep-03` | `/tmp/rffr/current/rep-03/rapthor-command.log` |

## Comparison Summary

| Result | Ops | Records | FITS | Image HDUs | Table HDUs | H5 | Text | Diagnostics | Visuals |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fail | 4 | 4 | 7 | 6 | 1 | 4 | 16 | 1 | 7 |

## FITS Residual Metrics

| Product | Max Abs Delta | P99 Abs Delta | Residual RMS | RMS / Ref RMS | RMS / Ref MAD |
| --- | ---: | ---: | ---: | ---: | ---: |
| `field-MFS-model-pb.fits.fz` | 2.269e-01 | 0.000e+00 | 3.449e-03 | 1.137e+00 | n/a |
| `field-MFS-dirty.fits.fz` | 1.458e-05 | 9.939e-06 | 4.155e-06 | 2.405e-05 | 2.683e-05 |
| `field-MFS-image-pb.fits.fz` | 4.333e-06 | 2.846e-06 | 1.193e-06 | 1.416e-05 | 2.861e-05 |
| `field-MFS-image-pb-ast.fits.fz` | 4.265e-06 | 2.848e-06 | 1.193e-06 | 1.416e-05 | 2.861e-05 |
| `field-MFS-image.fits.fz` | 4.251e-06 | 2.794e-06 | 1.169e-06 | 1.408e-05 | 2.860e-05 |
| `field-MFS-residual.fits.fz` | 4.221e-06 | 2.774e-06 | 1.162e-06 | 2.786e-05 | 2.852e-05 |

## Image Diagnostics

| Operation | Sector | Field | Reference | Current | Delta | Relative Delta |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `image_1` | `sector_1` | `nsources` | 1.000e+01 | 1.000e+01 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `theoretical_rms` | 9.006e-03 | 9.006e-03 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `min_rms_flat_noise` | 1.987e-02 | 1.987e-02 | 3.166e-08 | 0.000% |
| `image_1` | `sector_1` | `median_rms_flat_noise` | 3.982e-02 | 3.982e-02 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `dynamic_range_global_flat_noise` | 2.296e+02 | 2.296e+02 | -3.899e-04 | -0.000% |
| `image_1` | `sector_1` | `min_rms_true_sky` | 2.038e-02 | 2.038e-02 | 1.304e-08 | 0.000% |
| `image_1` | `sector_1` | `median_rms_true_sky` | 4.061e-02 | 4.061e-02 | 3.725e-09 | 0.000% |
| `image_1` | `sector_1` | `dynamic_range_global_true_sky` | 2.239e+02 | 2.239e+02 | -1.432e-04 | -0.000% |

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

- FITS image pixels differ for field-MFS-dirty.fits.fz: max_abs_delta=1.4580786228179932e-05, p99_abs_delta=9.939074516296387e-06, residual_rms=4.1551793745963065e-06
- FITS image pixels differ for field-MFS-image-pb-ast.fits.fz: max_abs_delta=4.265457391738892e-06, p99_abs_delta=2.8479844331741333e-06, residual_rms=1.1931866427014239e-06
- FITS image pixels differ for field-MFS-image-pb.fits.fz: max_abs_delta=4.332512617111206e-06, p99_abs_delta=2.8461217880249023e-06, residual_rms=1.1932150780304785e-06
- FITS image pixels differ for field-MFS-image.fits.fz: max_abs_delta=4.250556230545044e-06, p99_abs_delta=2.7939677238464355e-06, residual_rms=1.1689341724137631e-06
- FITS std differs for field-MFS-model-pb.fits.fz: 0.0030333440612578507 != 0.0030277587799282164
- FITS rms differs for field-MFS-model-pb.fits.fz: 0.0030333481047317788 != 0.0030277629097303737
- FITS max differs for field-MFS-model-pb.fits.fz: 1.675413966178894 != 1.691113829612732
- FITS image pixels differ for field-MFS-model-pb.fits.fz: max_abs_delta=0.22692637145519257, p99_abs_delta=0.0, residual_rms=0.003449130329555753
- FITS image pixels differ for field-MFS-residual.fits.fz: max_abs_delta=4.220753908157349e-06, p99_abs_delta=2.7739442884922028e-06, residual_rms=1.161962642387442e-06
- text product differs for sector_1_facets_ds9.reg
