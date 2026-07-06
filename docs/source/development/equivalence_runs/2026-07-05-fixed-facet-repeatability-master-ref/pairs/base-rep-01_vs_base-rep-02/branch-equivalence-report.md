# Rapthor Branch Equivalence

Scenario: `fixed-facet-carryover-repeatability:base-rep-01_vs_base-rep-02`
Run root: `/tmp/rffr/pairs/base-rep-01_vs_base-rep-02`

## Branch Runs

Repeatability pair: `base-rep-01_vs_base-rep-02` (base/rep-01 -> base/rep-02)

| Role | Repetition | Ref | Return Code | Parset | Work Dir | Log |
| --- | --- | --- | ---: | --- | --- | --- |
| reference | `base/rep-01` | `master` | 0 | `/tmp/rffr/inputs/base/rep-01/master_fixed_facet_carryover.parset` | `/tmp/rffw/base/rep-01` | `/tmp/rffr/base/rep-01/rapthor-command.log` |
| candidate | `base/rep-02` | `master` | 0 | `/tmp/rffr/inputs/base/rep-02/master_fixed_facet_carryover.parset` | `/tmp/rffw/base/rep-02` | `/tmp/rffr/base/rep-02/rapthor-command.log` |

## Comparison Summary

| Result | Ops | Records | FITS | Image HDUs | Table HDUs | H5 | Text | Diagnostics | Visuals |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fail | 4 | 4 | 7 | 6 | 1 | 4 | 16 | 1 | 7 |

## FITS Residual Metrics

| Product | Max Abs Delta | P99 Abs Delta | Residual RMS | RMS / Ref RMS | RMS / Ref MAD |
| --- | ---: | ---: | ---: | ---: | ---: |
| `field-MFS-model-pb.fits.fz` | 2.325e-01 | 0.000e+00 | 3.454e-03 | 1.140e+00 | n/a |
| `field-MFS-dirty.fits.fz` | 1.469e-05 | 9.939e-06 | 4.154e-06 | 2.404e-05 | 2.682e-05 |
| `field-MFS-image-pb-ast.fits.fz` | 4.306e-06 | 2.850e-06 | 1.193e-06 | 1.416e-05 | 2.861e-05 |
| `field-MFS-image-pb.fits.fz` | 4.306e-06 | 2.850e-06 | 1.193e-06 | 1.416e-05 | 2.861e-05 |
| `field-MFS-image.fits.fz` | 4.239e-06 | 2.794e-06 | 1.170e-06 | 1.409e-05 | 2.862e-05 |
| `field-MFS-residual.fits.fz` | 4.205e-06 | 2.773e-06 | 1.162e-06 | 2.786e-05 | 2.851e-05 |

## Image Diagnostics

| Operation | Sector | Field | Reference | Current | Delta | Relative Delta |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `image_1` | `sector_1` | `nsources` | 1.000e+01 | 1.000e+01 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `theoretical_rms` | 9.006e-03 | 9.006e-03 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `min_rms_flat_noise` | 1.987e-02 | 1.987e-02 | 4.843e-08 | 0.000% |
| `image_1` | `sector_1` | `median_rms_flat_noise` | 3.982e-02 | 3.982e-02 | -7.451e-09 | -0.000% |
| `image_1` | `sector_1` | `dynamic_range_global_flat_noise` | 2.296e+02 | 2.296e+02 | -5.356e-04 | -0.000% |
| `image_1` | `sector_1` | `min_rms_true_sky` | 2.038e-02 | 2.038e-02 | 3.353e-08 | 0.000% |
| `image_1` | `sector_1` | `median_rms_true_sky` | 4.061e-02 | 4.061e-02 | -7.451e-09 | -0.000% |
| `image_1` | `sector_1` | `dynamic_range_global_true_sky` | 2.239e+02 | 2.239e+02 | -3.917e-04 | -0.000% |

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

- FITS image pixels differ for field-MFS-dirty.fits.fz: max_abs_delta=1.4685094356536865e-05, p99_abs_delta=9.939074516296387e-06, residual_rms=4.1543692044721794e-06
- FITS image pixels differ for field-MFS-image-pb-ast.fits.fz: max_abs_delta=4.306435585021973e-06, p99_abs_delta=2.8498470783233643e-06, residual_rms=1.1932338425090425e-06
- FITS image pixels differ for field-MFS-image-pb.fits.fz: max_abs_delta=4.306435585021973e-06, p99_abs_delta=2.8498470783233643e-06, residual_rms=1.1932338425090425e-06
- FITS image pixels differ for field-MFS-image.fits.fz: max_abs_delta=4.239380359649658e-06, p99_abs_delta=2.7939677238464355e-06, residual_rms=1.16965241266008e-06
- FITS std differs for field-MFS-model-pb.fits.fz: 0.0030284670880311054 != 0.0030227072249057975
- FITS rms differs for field-MFS-model-pb.fits.fz: 0.00302847139244273 != 0.0030227113173986993
- FITS max differs for field-MFS-model-pb.fits.fz: 1.6887661218643188 != 1.6860357522964478
- FITS image pixels differ for field-MFS-model-pb.fits.fz: max_abs_delta=0.23250412940979004, p99_abs_delta=0.0, residual_rms=0.0034539363737590488
- FITS image pixels differ for field-MFS-residual.fits.fz: max_abs_delta=4.2049214243888855e-06, p99_abs_delta=2.773478627204895e-06, residual_rms=1.1616035680614868e-06
