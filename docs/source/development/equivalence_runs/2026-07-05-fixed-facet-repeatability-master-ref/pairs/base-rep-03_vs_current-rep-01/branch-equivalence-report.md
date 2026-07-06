# Rapthor Branch Equivalence

Scenario: `fixed-facet-carryover-repeatability:base-rep-03_vs_current-rep-01`
Run root: `/tmp/rffr/pairs/base-rep-03_vs_current-rep-01`

## Branch Runs

Repeatability pair: `base-rep-03_vs_current-rep-01` (base/rep-03 -> current/rep-01)

| Role | Repetition | Ref | Return Code | Parset | Work Dir | Log |
| --- | --- | --- | ---: | --- | --- | --- |
| reference | `base/rep-03` | `master` | 0 | `/tmp/rffr/inputs/base/rep-03/master_fixed_facet_carryover.parset` | `/tmp/rffw/base/rep-03` | `/tmp/rffr/base/rep-03/rapthor-command.log` |
| candidate | `current/rep-01` | `current` | 0 | `/tmp/rffr/inputs/current/rep-01/current_fixed_facet_carryover.parset` | `/tmp/rffw/current/rep-01` | `/tmp/rffr/current/rep-01/rapthor-command.log` |

## Comparison Summary

| Result | Ops | Records | FITS | Image HDUs | Table HDUs | H5 | Text | Diagnostics | Visuals |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fail | 4 | 4 | 7 | 6 | 1 | 4 | 16 | 1 | 7 |

## FITS Residual Metrics

| Product | Max Abs Delta | P99 Abs Delta | Residual RMS | RMS / Ref RMS | RMS / Ref MAD |
| --- | ---: | ---: | ---: | ---: | ---: |
| `field-MFS-model-pb.fits.fz` | 2.287e-01 | 0.000e+00 | 3.462e-03 | 1.141e+00 | n/a |
| `field-MFS-dirty.fits.fz` | 1.460e-05 | 9.937e-06 | 4.154e-06 | 2.404e-05 | 2.682e-05 |
| `field-MFS-image-pb.fits.fz` | 4.321e-06 | 2.850e-06 | 1.193e-06 | 1.417e-05 | 2.861e-05 |
| `field-MFS-image-pb-ast.fits.fz` | 4.306e-06 | 2.848e-06 | 1.193e-06 | 1.416e-05 | 2.860e-05 |
| `field-MFS-image.fits.fz` | 4.212e-06 | 2.796e-06 | 1.169e-06 | 1.409e-05 | 2.861e-05 |
| `field-MFS-residual.fits.fz` | 4.210e-06 | 2.775e-06 | 1.162e-06 | 2.787e-05 | 2.853e-05 |

## Image Diagnostics

| Operation | Sector | Field | Reference | Current | Delta | Relative Delta |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `image_1` | `sector_1` | `nsources` | 1.000e+01 | 1.000e+01 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `theoretical_rms` | 9.006e-03 | 9.006e-03 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `min_rms_flat_noise` | 1.987e-02 | 1.987e-02 | 6.333e-08 | 0.000% |
| `image_1` | `sector_1` | `median_rms_flat_noise` | 3.982e-02 | 3.982e-02 | -3.725e-09 | -0.000% |
| `image_1` | `sector_1` | `dynamic_range_global_flat_noise` | 2.296e+02 | 2.296e+02 | -7.558e-04 | -0.000% |
| `image_1` | `sector_1` | `min_rms_true_sky` | 2.038e-02 | 2.038e-02 | 7.264e-08 | 0.000% |
| `image_1` | `sector_1` | `median_rms_true_sky` | 4.061e-02 | 4.061e-02 | 3.725e-09 | 0.000% |
| `image_1` | `sector_1` | `dynamic_range_global_true_sky` | 2.239e+02 | 2.239e+02 | -7.980e-04 | -0.000% |

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

- FITS image pixels differ for field-MFS-dirty.fits.fz: max_abs_delta=1.459568738937378e-05, p99_abs_delta=9.937211871147156e-06, residual_rms=4.153668128795028e-06
- FITS image pixels differ for field-MFS-image-pb-ast.fits.fz: max_abs_delta=4.306435585021973e-06, p99_abs_delta=2.8479844331741333e-06, residual_rms=1.1931558184632536e-06
- FITS image pixels differ for field-MFS-image-pb.fits.fz: max_abs_delta=4.32133674621582e-06, p99_abs_delta=2.849556622095393e-06, residual_rms=1.1934497611114694e-06
- FITS image pixels differ for field-MFS-image.fits.fz: max_abs_delta=4.212372004985809e-06, p99_abs_delta=2.7958303689956665e-06, residual_rms=1.169404198493388e-06
- FITS std differs for field-MFS-model-pb.fits.fz: 0.0030333440612578507 != 0.003051534233221859
- FITS rms differs for field-MFS-model-pb.fits.fz: 0.0030333481047317788 != 0.0030515385861614027
- FITS max differs for field-MFS-model-pb.fits.fz: 1.675413966178894 != 1.6721535921096802
- FITS image pixels differ for field-MFS-model-pb.fits.fz: max_abs_delta=0.2286682352423668, p99_abs_delta=0.0, residual_rms=0.003461749312003581
- FITS image pixels differ for field-MFS-residual.fits.fz: max_abs_delta=4.209578037261963e-06, p99_abs_delta=2.775341272354126e-06, residual_rms=1.1623203261825707e-06
- text product differs for sector_1_facets_ds9.reg
