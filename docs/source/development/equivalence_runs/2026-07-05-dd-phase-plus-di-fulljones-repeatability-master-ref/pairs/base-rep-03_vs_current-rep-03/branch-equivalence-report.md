# Rapthor Branch Equivalence

Scenario: `dd-phase-plus-di-fulljones-repeatability:base-rep-03_vs_current-rep-03`
Run root: `/tmp/rfjr/pairs/base-rep-03_vs_current-rep-03`

## Branch Runs

Repeatability pair: `base-rep-03_vs_current-rep-03` (base/rep-03 -> current/rep-03)

| Role | Repetition | Ref | Return Code | Parset | Work Dir | Log |
| --- | --- | --- | ---: | --- | --- | --- |
| reference | `base/rep-03` | `master` | 0 | `/tmp/rfjr/inputs/base/rep-03/master_dd_phase_plus_di_fulljones.parset` | `/tmp/rfjw/base/rep-03` | `/tmp/rfjr/base/rep-03/rapthor-command.log` |
| candidate | `current/rep-03` | `current` | 0 | `/tmp/rfjr/inputs/current/rep-03/current_dd_phase_plus_di_fulljones.parset` | `/tmp/rfjw/current/rep-03` | `/tmp/rfjr/current/rep-03/rapthor-command.log` |

## Comparison Summary

| Result | Ops | Records | FITS | Image HDUs | Table HDUs | H5 | Text | Diagnostics | Visuals |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fail | 5 | 5 | 7 | 6 | 1 | 3 | 10 | 1 | 5 |

## FITS Residual Metrics

| Product | Max Abs Delta | P99 Abs Delta | Residual RMS | RMS / Ref RMS | RMS / Ref MAD |
| --- | ---: | ---: | ---: | ---: | ---: |
| `field-MFS-image-pb-ast.fits` | 1.024e-02 | 1.075e-04 | 1.674e-04 | 2.240e-03 | 1.009e-02 |
| `field-MFS-image-pb.fits` | 1.024e-02 | 1.075e-04 | 1.674e-04 | 2.240e-03 | 1.009e-02 |
| `field-MFS-image.fits` | 1.024e-02 | 1.056e-04 | 1.652e-04 | 2.240e-03 | 1.017e-02 |
| `field-MFS-dirty.fits` | 1.016e-02 | 9.259e-04 | 3.871e-04 | 2.239e-03 | 2.499e-03 |
| `field-MFS-model-pb.fits` | 3.433e-03 | 0.000e+00 | 4.223e-06 | 2.435e-03 | n/a |
| `field-MFS-residual.fits` | 5.188e-04 | 9.978e-05 | 3.900e-05 | 2.253e-03 | 2.407e-03 |

## Image Diagnostics

| Operation | Sector | Field | Reference | Current | Delta | Relative Delta |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `image_1` | `sector_1` | `nsources` | 1.100e+01 | 1.100e+01 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `theoretical_rms` | 9.006e-03 | 9.006e-03 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `min_rms_flat_noise` | 8.312e-03 | 8.329e-03 | 1.717e-05 | 0.207% |
| `image_1` | `sector_1` | `median_rms_flat_noise` | 1.589e-02 | 1.592e-02 | 3.554e-05 | 0.224% |
| `image_1` | `sector_1` | `dynamic_range_global_flat_noise` | 5.505e+02 | 5.506e+02 | 9.528e-02 | 0.017% |
| `image_1` | `sector_1` | `min_rms_true_sky` | 8.437e-03 | 8.457e-03 | 1.937e-05 | 0.230% |
| `image_1` | `sector_1` | `median_rms_true_sky` | 1.622e-02 | 1.625e-02 | 3.623e-05 | 0.223% |
| `image_1` | `sector_1` | `dynamic_range_global_true_sky` | 5.424e+02 | 5.424e+02 | -3.080e-02 | -0.006% |

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

- FITS std differs for field-MFS-dirty.fits: 0.1729215884878841 != 0.17330873392413104
- FITS rms differs for field-MFS-dirty.fits: 0.17292200686529471 != 0.17330915323640025
- FITS min differs for field-MFS-dirty.fits: -0.7253921627998352 != -0.7270155549049377
- FITS max differs for field-MFS-dirty.fits: 4.537002086639404 != 4.547160625457764
- FITS image pixels differ for field-MFS-dirty.fits: max_abs_delta=0.010158538818359375, p99_abs_delta=0.0009259283542633057, residual_rms=0.0003871490087490405
- FITS mean differs for field-MFS-image-pb-ast.fits: 0.002985477240013457 != 0.0029921644548981793
- FITS std differs for field-MFS-image-pb-ast.fits: 0.07468695543600888 != 0.07485417645952439
- FITS rms differs for field-MFS-image-pb-ast.fits: 0.07474660117123062 != 0.07491395585309035
- FITS min differs for field-MFS-image-pb-ast.fits: -0.08084569126367569 != -0.08102956414222717
- FITS max differs for field-MFS-image-pb-ast.fits: 4.576261043548584 != 4.586505889892578
- FITS image pixels differ for field-MFS-image-pb-ast.fits: max_abs_delta=0.01024484634399414, p99_abs_delta=0.000107545405626297, residual_rms=0.00016740824092365336
- FITS mean differs for field-MFS-image-pb.fits: 0.002985477240013457 != 0.0029921644548981793
- FITS std differs for field-MFS-image-pb.fits: 0.07468695543600888 != 0.07485417645952439
- FITS rms differs for field-MFS-image-pb.fits: 0.07474660117123062 != 0.07491395585309035
- FITS min differs for field-MFS-image-pb.fits: -0.08084569126367569 != -0.08102956414222717
- FITS max differs for field-MFS-image-pb.fits: 4.576261043548584 != 4.586505889892578
- FITS image pixels differ for field-MFS-image-pb.fits: max_abs_delta=0.01024484634399414, p99_abs_delta=0.000107545405626297, residual_rms=0.00016740824092365336
- FITS mean differs for field-MFS-image.fits: 0.0029421999276842874 != 0.0029487900900936363
- FITS std differs for field-MFS-image.fits: 0.073703195765222 != 0.07386821343643522
- FITS rms differs for field-MFS-image.fits: 0.07376189806682786 != 0.07392704727828765
- FITS min differs for field-MFS-image.fits: -0.08031633496284485 != -0.08049900084733963
- FITS max differs for field-MFS-image.fits: 4.576253414154053 != 4.5864973068237305
- FITS image pixels differ for field-MFS-image.fits: max_abs_delta=0.010243892669677734, p99_abs_delta=0.00010555237531661987, residual_rms=0.0001652015485874587
- FITS std differs for field-MFS-model-pb.fits: 0.0017346314253611325 != 0.00173853323587183
- FITS rms differs for field-MFS-model-pb.fits: 0.001734634193171336 != 0.0017385360098505951
- FITS max differs for field-MFS-model-pb.fits: 1.5328223705291748 != 1.536255121231079
- FITS image pixels differ for field-MFS-model-pb.fits: max_abs_delta=0.003432750701904297, p99_abs_delta=0.0, residual_rms=4.222974818182754e-06
- FITS std differs for field-MFS-residual.fits: 0.017308978242585326 != 0.01734773107610638
- FITS rms differs for field-MFS-residual.fits: 0.01730898421301821 != 0.017347737058014928
- FITS min differs for field-MFS-residual.fits: -0.08031633496284485 != -0.08049900829792023
- FITS max differs for field-MFS-residual.fits: 0.2298310250043869 != 0.23034514486789703
- FITS image pixels differ for field-MFS-residual.fits: max_abs_delta=0.0005187541246414185, p99_abs_delta=9.97781753540039e-05, residual_rms=3.900446738925103e-05
- HDF5 numeric dataset differs for fulljones-solutions.h5:sol000/amplitude000/val (max_abs=0.0011365)
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
