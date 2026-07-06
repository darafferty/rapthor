# Rapthor Branch Equivalence

Scenario: `fixed-facet-carryover-repeatability:base-rep-02_vs_current-rep-03`
Run root: `/tmp/rffr/pairs/base-rep-02_vs_current-rep-03`

## Branch Runs

Repeatability pair: `base-rep-02_vs_current-rep-03` (base/rep-02 -> current/rep-03)

| Role | Repetition | Ref | Return Code | Parset | Work Dir | Log |
| --- | --- | --- | ---: | --- | --- | --- |
| reference | `base/rep-02` | `master` | 0 | `/tmp/rffr/inputs/base/rep-02/master_fixed_facet_carryover.parset` | `/tmp/rffw/base/rep-02` | `/tmp/rffr/base/rep-02/rapthor-command.log` |
| candidate | `current/rep-03` | `current` | 0 | `/tmp/rffr/inputs/current/rep-03/current_fixed_facet_carryover.parset` | `/tmp/rffw/current/rep-03` | `/tmp/rffr/current/rep-03/rapthor-command.log` |

## Comparison Summary

| Result | Ops | Records | FITS | Image HDUs | Table HDUs | H5 | Text | Diagnostics | Visuals |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fail | 4 | 4 | 7 | 6 | 1 | 4 | 16 | 1 | 7 |

## FITS Residual Metrics

| Product | Max Abs Delta | P99 Abs Delta | Residual RMS | RMS / Ref RMS | RMS / Ref MAD |
| --- | ---: | ---: | ---: | ---: | ---: |
| `field-MFS-model-pb.fits.fz` | 2.309e-01 | 0.000e+00 | 3.432e-03 | 1.135e+00 | n/a |
| `field-MFS-dirty.fits.fz` | 1.449e-05 | 9.932e-06 | 4.152e-06 | 2.403e-05 | 2.681e-05 |
| `field-MFS-image-pb.fits.fz` | 4.300e-06 | 2.850e-06 | 1.193e-06 | 1.416e-05 | 2.861e-05 |
| `field-MFS-image-pb-ast.fits.fz` | 4.292e-06 | 2.850e-06 | 1.193e-06 | 1.416e-05 | 2.860e-05 |
| `field-MFS-residual.fits.fz` | 4.245e-06 | 2.772e-06 | 1.162e-06 | 2.786e-05 | 2.851e-05 |
| `field-MFS-image.fits.fz` | 4.232e-06 | 2.794e-06 | 1.170e-06 | 1.409e-05 | 2.862e-05 |

## Image Diagnostics

| Operation | Sector | Field | Reference | Current | Delta | Relative Delta |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `image_1` | `sector_1` | `nsources` | 1.000e+01 | 1.000e+01 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `theoretical_rms` | 9.006e-03 | 9.006e-03 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `min_rms_flat_noise` | 1.987e-02 | 1.987e-02 | -2.608e-08 | -0.000% |
| `image_1` | `sector_1` | `median_rms_flat_noise` | 3.982e-02 | 3.982e-02 | 3.725e-09 | 0.000% |
| `image_1` | `sector_1` | `dynamic_range_global_flat_noise` | 2.296e+02 | 2.296e+02 | 2.773e-04 | 0.000% |
| `image_1` | `sector_1` | `min_rms_true_sky` | 2.038e-02 | 2.038e-02 | -3.166e-08 | -0.000% |
| `image_1` | `sector_1` | `median_rms_true_sky` | 4.061e-02 | 4.061e-02 | 7.451e-09 | 0.000% |
| `image_1` | `sector_1` | `dynamic_range_global_true_sky` | 2.239e+02 | 2.239e+02 | 3.713e-04 | 0.000% |

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

- FITS image pixels differ for field-MFS-dirty.fits.fz: max_abs_delta=1.4490531611954793e-05, p99_abs_delta=9.931623935699463e-06, residual_rms=4.151630726274181e-06
- FITS image pixels differ for field-MFS-image-pb-ast.fits.fz: max_abs_delta=4.291534423828125e-06, p99_abs_delta=2.8498470783233643e-06, residual_rms=1.1929276975699797e-06
- FITS image pixels differ for field-MFS-image-pb.fits.fz: max_abs_delta=4.2995670810341835e-06, p99_abs_delta=2.8498470783233643e-06, residual_rms=1.1932382546886292e-06
- FITS image pixels differ for field-MFS-image.fits.fz: max_abs_delta=4.231929779052734e-06, p99_abs_delta=2.7939677238464355e-06, residual_rms=1.1695792645273048e-06
- FITS std differs for field-MFS-model-pb.fits.fz: 0.0030227072249057975 != 0.0030277587799282164
- FITS rms differs for field-MFS-model-pb.fits.fz: 0.0030227113173986993 != 0.0030277629097303737
- FITS max differs for field-MFS-model-pb.fits.fz: 1.6860357522964478 != 1.691113829612732
- FITS image pixels differ for field-MFS-model-pb.fits.fz: max_abs_delta=0.230933278799057, p99_abs_delta=0.0, residual_rms=0.0034315584776740235
- FITS image pixels differ for field-MFS-residual.fits.fz: max_abs_delta=4.244968295097351e-06, p99_abs_delta=2.771615982055664e-06, residual_rms=1.1616341029489738e-06
- text product differs for sector_1_facets_ds9.reg
