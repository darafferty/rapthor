# Rapthor Branch Equivalence

Scenario: `dd-phase-plus-di-fulljones-repeatability:base-rep-01_vs_current-rep-03`
Run root: `/tmp/rfjr/pairs/base-rep-01_vs_current-rep-03`

## Branch Runs

Repeatability pair: `base-rep-01_vs_current-rep-03` (base/rep-01 -> current/rep-03)

| Role | Repetition | Ref | Return Code | Parset | Work Dir | Log |
| --- | --- | --- | ---: | --- | --- | --- |
| reference | `base/rep-01` | `master` | 0 | `/tmp/rfjr/inputs/base/rep-01/master_dd_phase_plus_di_fulljones.parset` | `/tmp/rfjw/base/rep-01` | `/tmp/rfjr/base/rep-01/rapthor-command.log` |
| candidate | `current/rep-03` | `current` | 0 | `/tmp/rfjr/inputs/current/rep-03/current_dd_phase_plus_di_fulljones.parset` | `/tmp/rfjw/current/rep-03` | `/tmp/rfjr/current/rep-03/rapthor-command.log` |

## Comparison Summary

| Result | Ops | Records | FITS | Image HDUs | Table HDUs | H5 | Text | Diagnostics | Visuals |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fail | 5 | 5 | 7 | 6 | 1 | 3 | 10 | 1 | 5 |

## FITS Residual Metrics

| Product | Max Abs Delta | P99 Abs Delta | Residual RMS | RMS / Ref RMS | RMS / Ref MAD |
| --- | ---: | ---: | ---: | ---: | ---: |
| `field-MFS-image.fits` | 1.024e-02 | 1.056e-04 | 1.652e-04 | 2.240e-03 | 1.017e-02 |
| `field-MFS-image-pb-ast.fits` | 1.024e-02 | 1.075e-04 | 1.674e-04 | 2.240e-03 | 1.009e-02 |
| `field-MFS-image-pb.fits` | 1.024e-02 | 1.075e-04 | 1.674e-04 | 2.240e-03 | 1.009e-02 |
| `field-MFS-dirty.fits` | 1.016e-02 | 9.260e-04 | 3.872e-04 | 2.239e-03 | 2.499e-03 |
| `field-MFS-model-pb.fits` | 3.433e-03 | 0.000e+00 | 4.223e-06 | 2.435e-03 | n/a |
| `field-MFS-residual.fits` | 5.188e-04 | 9.978e-05 | 3.900e-05 | 2.253e-03 | 2.407e-03 |

## Image Diagnostics

| Operation | Sector | Field | Reference | Current | Delta | Relative Delta |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `image_1` | `sector_1` | `nsources` | 1.100e+01 | 1.100e+01 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `theoretical_rms` | 9.006e-03 | 9.006e-03 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `min_rms_flat_noise` | 8.312e-03 | 8.329e-03 | 1.716e-05 | 0.206% |
| `image_1` | `sector_1` | `median_rms_flat_noise` | 1.589e-02 | 1.592e-02 | 3.554e-05 | 0.224% |
| `image_1` | `sector_1` | `dynamic_range_global_flat_noise` | 5.505e+02 | 5.506e+02 | 9.558e-02 | 0.017% |
| `image_1` | `sector_1` | `min_rms_true_sky` | 8.439e-03 | 8.457e-03 | 1.776e-05 | 0.210% |
| `image_1` | `sector_1` | `median_rms_true_sky` | 1.622e-02 | 1.625e-02 | 3.624e-05 | 0.223% |
| `image_1` | `sector_1` | `dynamic_range_global_true_sky` | 5.423e+02 | 5.424e+02 | 7.270e-02 | 0.013% |

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


## Warnings

- output-record summary differs for calibrate_1
- output-record summary differs for calibrate_di_1

## Failures

- FITS std differs for field-MFS-dirty.fits: 0.17292157494830648 != 0.17330873392413104
- FITS rms differs for field-MFS-dirty.fits: 0.17292199332550473 != 0.17330915323640025
- FITS min differs for field-MFS-dirty.fits: -0.7253920435905457 != -0.7270155549049377
- FITS max differs for field-MFS-dirty.fits: 4.537002086639404 != 4.547160625457764
- FITS image pixels differ for field-MFS-dirty.fits: max_abs_delta=0.010158538818359375, p99_abs_delta=0.0009259581565856934, residual_rms=0.0003871625445458696
- FITS mean differs for field-MFS-image-pb-ast.fits: 0.002985477355971771 != 0.0029921644548981793
- FITS std differs for field-MFS-image-pb-ast.fits: 0.07468695280763152 != 0.07485417645952439
- FITS rms differs for field-MFS-image-pb-ast.fits: 0.07474659854958214 != 0.07491395585309035
- FITS min differs for field-MFS-image-pb-ast.fits: -0.08084563910961151 != -0.08102956414222717
- FITS max differs for field-MFS-image-pb-ast.fits: 4.576261520385742 != 4.586505889892578
- FITS image pixels differ for field-MFS-image-pb-ast.fits: max_abs_delta=0.010244369506835938, p99_abs_delta=0.000107545405626297, residual_rms=0.00016741083386541197
- FITS mean differs for field-MFS-image-pb.fits: 0.002985477355971771 != 0.0029921644548981793
- FITS std differs for field-MFS-image-pb.fits: 0.07468695280763152 != 0.07485417645952439
- FITS rms differs for field-MFS-image-pb.fits: 0.07474659854958214 != 0.07491395585309035
- FITS min differs for field-MFS-image-pb.fits: -0.08084563910961151 != -0.08102956414222717
- FITS max differs for field-MFS-image-pb.fits: 4.576261520385742 != 4.586505889892578
- FITS image pixels differ for field-MFS-image-pb.fits: max_abs_delta=0.010244369506835938, p99_abs_delta=0.000107545405626297, residual_rms=0.00016741083386541197
- FITS mean differs for field-MFS-image.fits: 0.0029421995943705306 != 0.0029487900900936363
- FITS std differs for field-MFS-image.fits: 0.07370319152111825 != 0.07386821343643522
- FITS rms differs for field-MFS-image.fits: 0.07376189381280653 != 0.07392704727828765
- FITS min differs for field-MFS-image.fits: -0.08031632006168365 != -0.08049900084733963
- FITS max differs for field-MFS-image.fits: 4.576252460479736 != 4.5864973068237305
- FITS image pixels differ for field-MFS-image.fits: max_abs_delta=0.01024484634399414, p99_abs_delta=0.00010555237531661987, residual_rms=0.00016520577496589463
- FITS std differs for field-MFS-model-pb.fits: 0.0017346313114246347 != 0.00173853323587183
- FITS rms differs for field-MFS-model-pb.fits: 0.0017346340792345682 != 0.0017385360098505951
- FITS max differs for field-MFS-model-pb.fits: 1.5328218936920166 != 1.536255121231079
- FITS image pixels differ for field-MFS-model-pb.fits: max_abs_delta=0.0034332275390625, p99_abs_delta=0.0, residual_rms=4.22307875071134e-06
- FITS std differs for field-MFS-residual.fits: 0.017308977759629793 != 0.01734773107610638
- FITS rms differs for field-MFS-residual.fits: 0.017308983729970976 != 0.017347737058014928
- FITS min differs for field-MFS-residual.fits: -0.08031632006168365 != -0.08049900829792023
- FITS max differs for field-MFS-residual.fits: 0.22983095049858093 != 0.23034514486789703
- FITS image pixels differ for field-MFS-residual.fits: max_abs_delta=0.0005187690258026123, p99_abs_delta=9.978193789720452e-05, residual_rms=3.9004829328994335e-05
- HDF5 numeric dataset differs for fulljones-solutions.h5:sol000/amplitude000/val (max_abs=0.00113649)
- FITS table column differs for sector_1.source_catalog.fits:E_RA
- FITS table column differs for sector_1.source_catalog.fits:E_DEC
- FITS table column differs for sector_1.source_catalog.fits:Total_flux
- FITS table column differs for sector_1.source_catalog.fits:E_Total_flux
- FITS table column differs for sector_1.source_catalog.fits:Peak_flux
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
- ... 11 more failure(s)
