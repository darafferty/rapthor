# Rapthor Branch Equivalence

Scenario: `dd-phase-plus-di-fulljones-repeatability:current-rep-01_vs_current-rep-03`
Run root: `/tmp/rfjr/pairs/current-rep-01_vs_current-rep-03`

## Branch Runs

Repeatability pair: `current-rep-01_vs_current-rep-03` (current/rep-01 -> current/rep-03)

| Role | Repetition | Ref | Return Code | Parset | Work Dir | Log |
| --- | --- | --- | ---: | --- | --- | --- |
| reference | `current/rep-01` | `current` | 0 | `/tmp/rfjr/inputs/current/rep-01/current_dd_phase_plus_di_fulljones.parset` | `/tmp/rfjw/current/rep-01` | `/tmp/rfjr/current/rep-01/rapthor-command.log` |
| candidate | `current/rep-03` | `current` | 0 | `/tmp/rfjr/inputs/current/rep-03/current_dd_phase_plus_di_fulljones.parset` | `/tmp/rfjw/current/rep-03` | `/tmp/rfjr/current/rep-03/rapthor-command.log` |

## Comparison Summary

| Result | Ops | Records | FITS | Image HDUs | Table HDUs | H5 | Text | Diagnostics | Visuals |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fail | 5 | 5 | 7 | 6 | 1 | 4 | 10 | 1 | 5 |

## FITS Residual Metrics

| Product | Max Abs Delta | P99 Abs Delta | Residual RMS | RMS / Ref RMS | RMS / Ref MAD |
| --- | ---: | ---: | ---: | ---: | ---: |
| `field-MFS-model-pb.fits` | 2.745e-03 | 0.000e+00 | 1.617e-06 | 9.302e-04 | n/a |
| `field-MFS-residual.fits` | 1.315e-04 | 1.051e-05 | 4.203e-06 | 2.423e-04 | 2.588e-04 |
| `field-MFS-image.fits` | 2.492e-05 | 1.047e-05 | 3.916e-06 | 5.297e-05 | 2.405e-04 |
| `field-MFS-image-pb-ast.fits` | 2.491e-05 | 1.061e-05 | 3.987e-06 | 5.322e-05 | 2.399e-04 |
| `field-MFS-image-pb.fits` | 2.491e-05 | 1.061e-05 | 3.987e-06 | 5.322e-05 | 2.399e-04 |
| `field-MFS-dirty.fits` | 9.537e-07 | 8.196e-08 | 2.931e-08 | 1.691e-07 | 1.888e-07 |

## Image Diagnostics

| Operation | Sector | Field | Reference | Current | Delta | Relative Delta |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `image_1` | `sector_1` | `nsources` | 1.100e+01 | 1.100e+01 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `theoretical_rms` | 9.006e-03 | 9.006e-03 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `min_rms_flat_noise` | 8.330e-03 | 8.329e-03 | -1.125e-06 | -0.014% |
| `image_1` | `sector_1` | `median_rms_flat_noise` | 1.592e-02 | 1.592e-02 | 2.794e-08 | 0.000% |
| `image_1` | `sector_1` | `dynamic_range_global_flat_noise` | 5.506e+02 | 5.506e+02 | 7.414e-02 | 0.013% |
| `image_1` | `sector_1` | `min_rms_true_sky` | 8.457e-03 | 8.457e-03 | -2.841e-07 | -0.003% |
| `image_1` | `sector_1` | `median_rms_true_sky` | 1.625e-02 | 1.625e-02 | -1.118e-08 | -0.000% |
| `image_1` | `sector_1` | `dynamic_range_global_true_sky` | 5.423e+02 | 5.424e+02 | 1.799e-02 | 0.003% |

## Visual Comparisons

### Image: `image_1/field-MFS-image-pb-ast.fits`

![image_1/field-MFS-image-pb-ast.fits](visual-comparisons/images/image_1-field-mfs-image-pb-ast.fits.png)

### Image: `image_1/field-MFS-image-pb.fits`

![image_1/field-MFS-image-pb.fits](visual-comparisons/images/image_1-field-mfs-image-pb.fits.png)

### Image: `image_1/field-MFS-residual.fits`

![image_1/field-MFS-residual.fits](visual-comparisons/images/image_1-field-mfs-residual.fits.png)

### Solution: `calibrate_1/fast_phase_dir[Patch_rich_centre].png`

![calibrate_1/fast_phase_dir[Patch_rich_centre].png](visual-comparisons/solutions/calibrate_1-fast_phase_dir-patch_rich_centre-.png.png)

### Solution: `calibrate_1/medium1_phase_dir[Patch_rich_centre].png`

![calibrate_1/medium1_phase_dir[Patch_rich_centre].png](visual-comparisons/solutions/calibrate_1-medium1_phase_dir-patch_rich_centre-.png.png)


## Failures

- FITS image pixels differ for field-MFS-image-pb-ast.fits: max_abs_delta=2.491474151611328e-05, p99_abs_delta=1.06077641248703e-05, residual_rms=3.987021183116268e-06
- FITS image pixels differ for field-MFS-image-pb.fits: max_abs_delta=2.491474151611328e-05, p99_abs_delta=1.06077641248703e-05, residual_rms=3.987021183116268e-06
- FITS image pixels differ for field-MFS-image.fits: max_abs_delta=2.4922192096710205e-05, p99_abs_delta=1.0472489520907402e-05, residual_rms=3.916150815330712e-06
- FITS image pixels differ for field-MFS-model-pb.fits: max_abs_delta=0.0027453675866127014, p99_abs_delta=0.0, residual_rms=1.6172544765086551e-06
- FITS image pixels differ for field-MFS-residual.fits: max_abs_delta=0.00013149529695510864, p99_abs_delta=1.0514631867408752e-05, residual_rms=4.203039787862828e-06
- FITS table column differs for sector_1.source_catalog.fits:E_RA
- FITS table column differs for sector_1.source_catalog.fits:E_DEC
- FITS table column differs for sector_1.source_catalog.fits:E_Total_flux
- FITS table column differs for sector_1.source_catalog.fits:E_Peak_flux
- FITS table column differs for sector_1.source_catalog.fits:E_RA_max
- FITS table column differs for sector_1.source_catalog.fits:E_DEC_max
- FITS table column differs for sector_1.source_catalog.fits:E_Maj
- FITS table column differs for sector_1.source_catalog.fits:E_Min
- FITS table column differs for sector_1.source_catalog.fits:PA
- FITS table column differs for sector_1.source_catalog.fits:E_PA
- FITS table column differs for sector_1.source_catalog.fits:E_Maj_img_plane
- FITS table column differs for sector_1.source_catalog.fits:E_Min_img_plane
- FITS table column differs for sector_1.source_catalog.fits:PA_img_plane
- FITS table column differs for sector_1.source_catalog.fits:E_PA_img_plane
- FITS table column differs for sector_1.source_catalog.fits:E_DC_Maj
- FITS table column differs for sector_1.source_catalog.fits:E_DC_Min
- FITS table column differs for sector_1.source_catalog.fits:E_DC_PA
- FITS table column differs for sector_1.source_catalog.fits:E_DC_Maj_img_plane
- FITS table column differs for sector_1.source_catalog.fits:E_DC_Min_img_plane
- FITS table column differs for sector_1.source_catalog.fits:E_DC_PA_img_plane
- FITS table column differs for sector_1.source_catalog.fits:E_Isl_Total_flux
- FITS table column differs for sector_1.source_catalog.fits:Isl_rms
- FITS table column differs for sector_1.source_catalog.fits:Resid_Isl_rms
- FITS table column differs for sector_1.source_catalog.fits:Resid_Isl_mean
