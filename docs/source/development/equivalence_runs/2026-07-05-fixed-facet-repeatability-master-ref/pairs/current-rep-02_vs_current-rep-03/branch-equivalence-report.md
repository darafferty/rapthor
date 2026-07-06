# Rapthor Branch Equivalence

Scenario: `fixed-facet-carryover-repeatability:current-rep-02_vs_current-rep-03`
Run root: `/tmp/rffr/pairs/current-rep-02_vs_current-rep-03`

## Branch Runs

Repeatability pair: `current-rep-02_vs_current-rep-03` (current/rep-02 -> current/rep-03)

| Role | Repetition | Ref | Return Code | Parset | Work Dir | Log |
| --- | --- | --- | ---: | --- | --- | --- |
| reference | `current/rep-02` | `current` | 0 | `/tmp/rffr/inputs/current/rep-02/current_fixed_facet_carryover.parset` | `/tmp/rffw/current/rep-02` | `/tmp/rffr/current/rep-02/rapthor-command.log` |
| candidate | `current/rep-03` | `current` | 0 | `/tmp/rffr/inputs/current/rep-03/current_fixed_facet_carryover.parset` | `/tmp/rffw/current/rep-03` | `/tmp/rffr/current/rep-03/rapthor-command.log` |

## Comparison Summary

| Result | Ops | Records | FITS | Image HDUs | Table HDUs | H5 | Text | Diagnostics | Visuals |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fail | 4 | 4 | 7 | 6 | 1 | 6 | 16 | 1 | 7 |

## FITS Residual Metrics

| Product | Max Abs Delta | P99 Abs Delta | Residual RMS | RMS / Ref RMS | RMS / Ref MAD |
| --- | ---: | ---: | ---: | ---: | ---: |
| `field-MFS-model-pb.fits.fz` | 2.347e-01 | 0.000e+00 | 3.491e-03 | 1.157e+00 | n/a |
| `field-MFS-image-pb-ast.fits.fz` | 1.809e-02 | 2.850e-06 | 7.633e-06 | 9.060e-05 | 1.830e-04 |
| `field-MFS-image-pb.fits.fz` | 1.809e-02 | 2.846e-06 | 7.632e-06 | 9.060e-05 | 1.830e-04 |
| `field-MFS-image.fits.fz` | 1.773e-02 | 2.794e-06 | 7.479e-06 | 9.011e-05 | 1.830e-04 |
| `field-MFS-residual.fits.fz` | 1.773e-02 | 2.772e-06 | 7.477e-06 | 1.793e-04 | 1.835e-04 |
| `field-MFS-dirty.fits.fz` | 1.458e-05 | 9.924e-06 | 4.154e-06 | 2.404e-05 | 2.682e-05 |

## Image Diagnostics

| Operation | Sector | Field | Reference | Current | Delta | Relative Delta |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `image_1` | `sector_1` | `nsources` | 1.000e+01 | 1.000e+01 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `theoretical_rms` | 9.006e-03 | 9.006e-03 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `min_rms_flat_noise` | 1.987e-02 | 1.987e-02 | -9.313e-09 | -0.000% |
| `image_1` | `sector_1` | `median_rms_flat_noise` | 3.982e-02 | 3.982e-02 | -3.725e-09 | -0.000% |
| `image_1` | `sector_1` | `dynamic_range_global_flat_noise` | 2.296e+02 | 2.296e+02 | 1.076e-04 | 0.000% |
| `image_1` | `sector_1` | `min_rms_true_sky` | 2.038e-02 | 2.038e-02 | 2.421e-08 | 0.000% |
| `image_1` | `sector_1` | `median_rms_true_sky` | 4.061e-02 | 4.061e-02 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `dynamic_range_global_true_sky` | 2.239e+02 | 2.239e+02 | -2.894e-04 | -0.000% |

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

- FITS image pixels differ for field-MFS-dirty.fits.fz: max_abs_delta=1.4580786228179932e-05, p99_abs_delta=9.924173355102539e-06, residual_rms=4.153559614566436e-06
- FITS image pixels differ for field-MFS-image-pb-ast.fits.fz: max_abs_delta=0.01809316989965737, p99_abs_delta=2.8498470783233643e-06, residual_rms=7.632645877120493e-06
- FITS image pixels differ for field-MFS-image-pb.fits.fz: max_abs_delta=0.018092716811224818, p99_abs_delta=2.8461217880249023e-06, residual_rms=7.632483359114674e-06
- FITS image pixels differ for field-MFS-image.fits.fz: max_abs_delta=0.017727727885358036, p99_abs_delta=2.7939677238464355e-06, residual_rms=7.478546707928068e-06
- FITS std differs for field-MFS-model-pb.fits.fz: 0.0030178976105584123 != 0.0030277587799282164
- FITS rms differs for field-MFS-model-pb.fits.fz: 0.003017901545826119 != 0.0030277629097303737
- FITS max differs for field-MFS-model-pb.fits.fz: 1.6711139678955078 != 1.691113829612732
- FITS image pixels differ for field-MFS-model-pb.fits.fz: max_abs_delta=0.23474358767271042, p99_abs_delta=0.0, residual_rms=0.0034907373917608727
- FITS image pixels differ for field-MFS-residual.fits.fz: max_abs_delta=0.01772759191226214, p99_abs_delta=2.771615982055664e-06, residual_rms=7.477270446333075e-06
