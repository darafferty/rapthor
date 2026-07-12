# Rapthor Branch Equivalence

Scenario: `dd-phase-plus-di-fulljones-repeatability:base-rep-02_vs_current-rep-02`
Run root: `/tmp/rfjr/pairs/base-rep-02_vs_current-rep-02`

## Branch Runs

Repeatability pair: `base-rep-02_vs_current-rep-02` (base/rep-02 -> current/rep-02)

| Role | Repetition | Ref | Return Code | Parset | Work Dir | Log |
| --- | --- | --- | ---: | --- | --- | --- |
| reference | `base/rep-02` | `master` | 0 | `/tmp/rfjr/inputs/base/rep-02/master_dd_phase_plus_di_fulljones.parset` | `/tmp/rfjw/base/rep-02` | `/tmp/rfjr/base/rep-02/rapthor-command.log` |
| candidate | `current/rep-02` | `current` | 0 | `/tmp/rfjr/inputs/current/rep-02/current_dd_phase_plus_di_fulljones.parset` | `/tmp/rfjw/current/rep-02` | `/tmp/rfjr/current/rep-02/rapthor-command.log` |

## Comparison Summary

| Result | Ops | Records | FITS | Image HDUs | Table HDUs | H5 | Text | Diagnostics | Visuals |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fail | 5 | 5 | 7 | 6 | 1 | 3 | 10 | 1 | 5 |

## FITS Residual Metrics

| Product | Max Abs Delta | P99 Abs Delta | Residual RMS | RMS / Ref RMS | RMS / Ref MAD |
| --- | ---: | ---: | ---: | ---: | ---: |
| `field-MFS-image.fits` | 1.024e-02 | 1.055e-04 | 1.652e-04 | 2.240e-03 | 1.017e-02 |
| `field-MFS-image-pb-ast.fits` | 1.024e-02 | 1.075e-04 | 1.674e-04 | 2.240e-03 | 1.009e-02 |
| `field-MFS-image-pb.fits` | 1.024e-02 | 1.075e-04 | 1.674e-04 | 2.240e-03 | 1.009e-02 |
| `field-MFS-dirty.fits` | 1.016e-02 | 9.259e-04 | 3.872e-04 | 2.239e-03 | 2.499e-03 |
| `field-MFS-model-pb.fits` | 3.433e-03 | 0.000e+00 | 4.223e-06 | 2.435e-03 | n/a |
| `field-MFS-residual.fits` | 5.187e-04 | 9.978e-05 | 3.900e-05 | 2.253e-03 | 2.407e-03 |

## Image Diagnostics

| Operation | Sector | Field | Reference | Current | Delta | Relative Delta |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `image_1` | `sector_1` | `nsources` | 1.100e+01 | 1.100e+01 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `theoretical_rms` | 9.006e-03 | 9.006e-03 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `min_rms_flat_noise` | 8.312e-03 | 8.330e-03 | 1.765e-05 | 0.212% |
| `image_1` | `sector_1` | `median_rms_flat_noise` | 1.589e-02 | 1.592e-02 | 3.554e-05 | 0.224% |
| `image_1` | `sector_1` | `dynamic_range_global_flat_noise` | 5.506e+02 | 5.506e+02 | 6.361e-02 | 0.012% |
| `image_1` | `sector_1` | `min_rms_true_sky` | 8.438e-03 | 8.457e-03 | 1.985e-05 | 0.235% |
| `image_1` | `sector_1` | `median_rms_true_sky` | 1.622e-02 | 1.625e-02 | 3.624e-05 | 0.223% |
| `image_1` | `sector_1` | `dynamic_range_global_true_sky` | 5.424e+02 | 5.423e+02 | -6.156e-02 | -0.011% |

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

- FITS std differs for field-MFS-dirty.fits: 0.1729215907380038 != 0.1733087420233318
- FITS rms differs for field-MFS-dirty.fits: 0.17292200911591032 != 0.1733091613358151
- FITS min differs for field-MFS-dirty.fits: -0.7253921627998352 != -0.7270156145095825
- FITS max differs for field-MFS-dirty.fits: 4.537002086639404 != 4.547160625457764
- FITS image pixels differ for field-MFS-dirty.fits: max_abs_delta=0.010158538818359375, p99_abs_delta=0.0009259283542633057, residual_rms=0.00038715485223250274
- FITS mean differs for field-MFS-image-pb-ast.fits: 0.0029854774769204648 != 0.002992164474127061
- FITS std differs for field-MFS-image-pb-ast.fits: 0.07468696110305105 != 0.07485417461942591
- FITS rms differs for field-MFS-image-pb-ast.fits: 0.07474660684321302 != 0.07491395401522827
- FITS min differs for field-MFS-image-pb-ast.fits: -0.08084568381309509 != -0.081029511988163
- FITS max differs for field-MFS-image-pb-ast.fits: 4.576261520385742 != 4.586505889892578
- FITS image pixels differ for field-MFS-image-pb-ast.fits: max_abs_delta=0.010244369506835938, p99_abs_delta=0.00010753795504570007, residual_rms=0.00016740072574846626
- FITS mean differs for field-MFS-image-pb.fits: 0.0029854774769204648 != 0.002992164474127061
- FITS std differs for field-MFS-image-pb.fits: 0.07468696110305105 != 0.07485417461942591
- FITS rms differs for field-MFS-image-pb.fits: 0.07474660684321302 != 0.07491395401522827
- FITS min differs for field-MFS-image-pb.fits: -0.08084568381309509 != -0.081029511988163
- FITS max differs for field-MFS-image-pb.fits: 4.576261520385742 != 4.586505889892578
- FITS image pixels differ for field-MFS-image-pb.fits: max_abs_delta=0.010244369506835938, p99_abs_delta=0.00010753795504570007, residual_rms=0.00016740072574846626
- FITS mean differs for field-MFS-image.fits: 0.002942199732113065 != 0.002948789912438056
- FITS std differs for field-MFS-image.fits: 0.0737031980582531 != 0.07386821245081815
- FITS rms differs for field-MFS-image.fits: 0.07376190035023314 != 0.07392704628636876
- FITS min differs for field-MFS-image.fits: -0.08031634986400604 != -0.08049898594617844
- FITS max differs for field-MFS-image.fits: 4.576252460479736 != 4.5864973068237305
- FITS image pixels differ for field-MFS-image.fits: max_abs_delta=0.01024484634399414, p99_abs_delta=0.00010554865002632141, residual_rms=0.000165198265276852
- FITS std differs for field-MFS-model-pb.fits: 0.0017346313838326756 != 0.0017385332979921082
- FITS rms differs for field-MFS-model-pb.fits: 0.0017346341516429253 != 0.0017385360719709758
- FITS max differs for field-MFS-model-pb.fits: 1.5328218936920166 != 1.5362552404403687
- FITS image pixels differ for field-MFS-model-pb.fits: max_abs_delta=0.0034333467483520508, p99_abs_delta=0.0, residual_rms=4.2230728648207725e-06
- FITS std differs for field-MFS-residual.fits: 0.01730897885234138 != 0.017347730603013246
- FITS rms differs for field-MFS-residual.fits: 0.01730898482278375 != 0.017347736584905116
- FITS min differs for field-MFS-residual.fits: -0.08031633496284485 != -0.08049899339675903
- FITS max differs for field-MFS-residual.fits: 0.22983074188232422 != 0.23034511506557465
- FITS image pixels differ for field-MFS-residual.fits: max_abs_delta=0.0005187243223190308, p99_abs_delta=9.97781753540039e-05, residual_rms=3.900337859205679e-05
- HDF5 numeric dataset differs for fulljones-solutions.h5:sol000/amplitude000/val (max_abs=0.00113646)
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
