# Rapthor Branch Equivalence

Scenario: `fixed-facet-carryover-repeatability:base-rep-02_vs_base-rep-03`
Run root: `/tmp/rffr/pairs/base-rep-02_vs_base-rep-03`

## Branch Runs

Repeatability pair: `base-rep-02_vs_base-rep-03` (base/rep-02 -> base/rep-03)

| Role | Repetition | Ref | Return Code | Parset | Work Dir | Log |
| --- | --- | --- | ---: | --- | --- | --- |
| reference | `base/rep-02` | `master` | 0 | `/tmp/rffr/inputs/base/rep-02/master_fixed_facet_carryover.parset` | `/tmp/rffw/base/rep-02` | `/tmp/rffr/base/rep-02/rapthor-command.log` |
| candidate | `base/rep-03` | `master` | 0 | `/tmp/rffr/inputs/base/rep-03/master_fixed_facet_carryover.parset` | `/tmp/rffw/base/rep-03` | `/tmp/rffr/base/rep-03/rapthor-command.log` |

## Comparison Summary

| Result | Ops | Records | FITS | Image HDUs | Table HDUs | H5 | Text | Diagnostics | Visuals |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fail | 4 | 4 | 7 | 6 | 1 | 4 | 16 | 1 | 7 |

## FITS Residual Metrics

| Product | Max Abs Delta | P99 Abs Delta | Residual RMS | RMS / Ref RMS | RMS / Ref MAD |
| --- | ---: | ---: | ---: | ---: | ---: |
| `field-MFS-model-pb.fits.fz` | 2.260e-01 | 0.000e+00 | 3.455e-03 | 1.143e+00 | n/a |
| `field-MFS-dirty.fits.fz` | 1.464e-05 | 9.924e-06 | 4.153e-06 | 2.404e-05 | 2.682e-05 |
| `field-MFS-image-pb-ast.fits.fz` | 4.236e-06 | 2.847e-06 | 1.193e-06 | 1.416e-05 | 2.859e-05 |
| `field-MFS-image-pb.fits.fz` | 4.236e-06 | 2.847e-06 | 1.193e-06 | 1.416e-05 | 2.859e-05 |
| `field-MFS-residual.fits.fz` | 4.189e-06 | 2.772e-06 | 1.161e-06 | 2.785e-05 | 2.850e-05 |
| `field-MFS-image.fits.fz` | 4.180e-06 | 2.794e-06 | 1.169e-06 | 1.409e-05 | 2.861e-05 |

## Image Diagnostics

| Operation | Sector | Field | Reference | Current | Delta | Relative Delta |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `image_1` | `sector_1` | `nsources` | 1.000e+01 | 1.000e+01 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `theoretical_rms` | 9.006e-03 | 9.006e-03 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `min_rms_flat_noise` | 1.987e-02 | 1.987e-02 | -5.774e-08 | -0.000% |
| `image_1` | `sector_1` | `median_rms_flat_noise` | 3.982e-02 | 3.982e-02 | 3.725e-09 | 0.000% |
| `image_1` | `sector_1` | `dynamic_range_global_flat_noise` | 2.296e+02 | 2.296e+02 | 6.672e-04 | 0.000% |
| `image_1` | `sector_1` | `min_rms_true_sky` | 2.038e-02 | 2.038e-02 | -4.470e-08 | -0.000% |
| `image_1` | `sector_1` | `median_rms_true_sky` | 4.061e-02 | 4.061e-02 | 3.725e-09 | 0.000% |
| `image_1` | `sector_1` | `dynamic_range_global_true_sky` | 2.239e+02 | 2.239e+02 | 5.145e-04 | 0.000% |

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


## Failures

- FITS image pixels differ for field-MFS-dirty.fits.fz: max_abs_delta=1.463666558265686e-05, p99_abs_delta=9.924173355102539e-06, residual_rms=4.1532694814641955e-06
- FITS image pixels differ for field-MFS-image-pb-ast.fits.fz: max_abs_delta=4.235655069351196e-06, p99_abs_delta=2.847053110599518e-06, residual_rms=1.1927334421219336e-06
- FITS image pixels differ for field-MFS-image-pb.fits.fz: max_abs_delta=4.235655069351196e-06, p99_abs_delta=2.847053110599518e-06, residual_rms=1.1927334421219336e-06
- FITS image pixels differ for field-MFS-image.fits.fz: max_abs_delta=4.179775714874268e-06, p99_abs_delta=2.7939677238464355e-06, residual_rms=1.1692352033004302e-06
- FITS std differs for field-MFS-model-pb.fits.fz: 0.0030227072249057975 != 0.0030333440612578507
- FITS rms differs for field-MFS-model-pb.fits.fz: 0.0030227113173986993 != 0.0030333481047317788
- FITS max differs for field-MFS-model-pb.fits.fz: 1.6860357522964478 != 1.675413966178894
- FITS image pixels differ for field-MFS-model-pb.fits.fz: max_abs_delta=0.2260182872414589, p99_abs_delta=0.0, residual_rms=0.00345521764748434
- FITS image pixels differ for field-MFS-residual.fits.fz: max_abs_delta=4.189088940620422e-06, p99_abs_delta=2.772489679045962e-06, residual_rms=1.1611743576310074e-06
