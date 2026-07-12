# Rapthor Branch Equivalence

Scenario: `fixed-facet-carryover-repeatability:current-rep-01_vs_current-rep-03`
Run root: `/tmp/rffr/pairs/current-rep-01_vs_current-rep-03`

## Branch Runs

Repeatability pair: `current-rep-01_vs_current-rep-03` (current/rep-01 -> current/rep-03)

| Role | Repetition | Ref | Return Code | Parset | Work Dir | Log |
| --- | --- | --- | ---: | --- | --- | --- |
| reference | `current/rep-01` | `current` | 0 | `/tmp/rffr/inputs/current/rep-01/current_fixed_facet_carryover.parset` | `/tmp/rffw/current/rep-01` | `/tmp/rffr/current/rep-01/rapthor-command.log` |
| candidate | `current/rep-03` | `current` | 0 | `/tmp/rffr/inputs/current/rep-03/current_fixed_facet_carryover.parset` | `/tmp/rffw/current/rep-03` | `/tmp/rffr/current/rep-03/rapthor-command.log` |

## Comparison Summary

| Result | Ops | Records | FITS | Image HDUs | Table HDUs | H5 | Text | Diagnostics | Visuals |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fail | 4 | 4 | 7 | 6 | 1 | 6 | 16 | 1 | 7 |

## FITS Residual Metrics

| Product | Max Abs Delta | P99 Abs Delta | Residual RMS | RMS / Ref RMS | RMS / Ref MAD |
| --- | ---: | ---: | ---: | ---: | ---: |
| `field-MFS-model-pb.fits.fz` | 2.331e-01 | 0.000e+00 | 3.455e-03 | 1.132e+00 | n/a |
| `field-MFS-dirty.fits.fz` | 1.462e-05 | 9.924e-06 | 4.152e-06 | 2.403e-05 | 2.681e-05 |
| `field-MFS-image-pb-ast.fits.fz` | 4.317e-06 | 2.849e-06 | 1.193e-06 | 1.416e-05 | 2.860e-05 |
| `field-MFS-image-pb.fits.fz` | 4.280e-06 | 2.850e-06 | 1.193e-06 | 1.416e-05 | 2.860e-05 |
| `field-MFS-image.fits.fz` | 4.202e-06 | 2.794e-06 | 1.169e-06 | 1.409e-05 | 2.861e-05 |
| `field-MFS-residual.fits.fz` | 4.195e-06 | 2.775e-06 | 1.162e-06 | 2.786e-05 | 2.851e-05 |

## Image Diagnostics

| Operation | Sector | Field | Reference | Current | Delta | Relative Delta |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `image_1` | `sector_1` | `nsources` | 1.000e+01 | 1.000e+01 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `theoretical_rms` | 9.006e-03 | 9.006e-03 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `min_rms_flat_noise` | 1.987e-02 | 1.987e-02 | -3.166e-08 | -0.000% |
| `image_1` | `sector_1` | `median_rms_flat_noise` | 3.982e-02 | 3.982e-02 | 3.725e-09 | 0.000% |
| `image_1` | `sector_1` | `dynamic_range_global_flat_noise` | 2.296e+02 | 2.296e+02 | 3.659e-04 | 0.000% |
| `image_1` | `sector_1` | `min_rms_true_sky` | 2.038e-02 | 2.038e-02 | -5.960e-08 | -0.000% |
| `image_1` | `sector_1` | `median_rms_true_sky` | 4.061e-02 | 4.061e-02 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `dynamic_range_global_true_sky` | 2.239e+02 | 2.239e+02 | 6.548e-04 | 0.000% |

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

- FITS image pixels differ for field-MFS-dirty.fits.fz: max_abs_delta=1.461803913116455e-05, p99_abs_delta=9.924173355102539e-06, residual_rms=4.152442089135136e-06
- FITS image pixels differ for field-MFS-image-pb-ast.fits.fz: max_abs_delta=4.316680133342743e-06, p99_abs_delta=2.8491485863924026e-06, residual_rms=1.1930683938578727e-06
- FITS image pixels differ for field-MFS-image-pb.fits.fz: max_abs_delta=4.280358552932739e-06, p99_abs_delta=2.8498470783233643e-06, residual_rms=1.1929407331445221e-06
- FITS image pixels differ for field-MFS-image.fits.fz: max_abs_delta=4.202127456665039e-06, p99_abs_delta=2.7939677238464355e-06, residual_rms=1.1693083337610456e-06
- FITS std differs for field-MFS-model-pb.fits.fz: 0.003051534233221859 != 0.0030277587799282164
- FITS rms differs for field-MFS-model-pb.fits.fz: 0.0030515385861614027 != 0.0030277629097303737
- FITS max differs for field-MFS-model-pb.fits.fz: 1.6721535921096802 != 1.691113829612732
- FITS image pixels differ for field-MFS-model-pb.fits.fz: max_abs_delta=0.23306414484977722, p99_abs_delta=0.0, residual_rms=0.0034546998803702163
- FITS image pixels differ for field-MFS-residual.fits.fz: max_abs_delta=4.194676876068115e-06, p99_abs_delta=2.775341272354126e-06, residual_rms=1.1618249088693683e-06
