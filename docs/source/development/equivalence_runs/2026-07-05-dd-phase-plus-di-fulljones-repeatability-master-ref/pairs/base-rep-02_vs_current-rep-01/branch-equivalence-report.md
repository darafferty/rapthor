# Rapthor Branch Equivalence

Scenario: `dd-phase-plus-di-fulljones-repeatability:base-rep-02_vs_current-rep-01`
Run root: `/tmp/rfjr/pairs/base-rep-02_vs_current-rep-01`

## Branch Runs

Repeatability pair: `base-rep-02_vs_current-rep-01` (base/rep-02 -> current/rep-01)

| Role | Repetition | Ref | Return Code | Parset | Work Dir | Log |
| --- | --- | --- | ---: | --- | --- | --- |
| reference | `base/rep-02` | `master` | 0 | `/tmp/rfjr/inputs/base/rep-02/master_dd_phase_plus_di_fulljones.parset` | `/tmp/rfjw/base/rep-02` | `/tmp/rfjr/base/rep-02/rapthor-command.log` |
| candidate | `current/rep-01` | `current` | 0 | `/tmp/rfjr/inputs/current/rep-01/current_dd_phase_plus_di_fulljones.parset` | `/tmp/rfjw/current/rep-01` | `/tmp/rfjr/current/rep-01/rapthor-command.log` |

## Comparison Summary

| Result | Ops | Records | FITS | Image HDUs | Table HDUs | H5 | Text | Diagnostics | Visuals |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fail | 5 | 5 | 7 | 6 | 1 | 3 | 10 | 1 | 5 |

## FITS Residual Metrics

| Product | Max Abs Delta | P99 Abs Delta | Residual RMS | RMS / Ref RMS | RMS / Ref MAD |
| --- | ---: | ---: | ---: | ---: | ---: |
| `field-MFS-image.fits` | 1.025e-02 | 1.046e-04 | 1.651e-04 | 2.239e-03 | 1.016e-02 |
| `field-MFS-image-pb-ast.fits` | 1.025e-02 | 1.066e-04 | 1.673e-04 | 2.239e-03 | 1.009e-02 |
| `field-MFS-image-pb.fits` | 1.025e-02 | 1.066e-04 | 1.673e-04 | 2.239e-03 | 1.009e-02 |
| `field-MFS-dirty.fits` | 1.016e-02 | 9.259e-04 | 3.871e-04 | 2.239e-03 | 2.499e-03 |
| `field-MFS-model-pb.fits` | 3.433e-03 | 0.000e+00 | 3.884e-06 | 2.239e-03 | n/a |
| `field-MFS-residual.fits` | 5.148e-04 | 9.898e-05 | 3.876e-05 | 2.239e-03 | 2.392e-03 |

## Image Diagnostics

| Operation | Sector | Field | Reference | Current | Delta | Relative Delta |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `image_1` | `sector_1` | `nsources` | 1.100e+01 | 1.100e+01 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `theoretical_rms` | 9.006e-03 | 9.006e-03 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `min_rms_flat_noise` | 8.312e-03 | 8.330e-03 | 1.846e-05 | 0.222% |
| `image_1` | `sector_1` | `median_rms_flat_noise` | 1.589e-02 | 1.592e-02 | 3.552e-05 | 0.224% |
| `image_1` | `sector_1` | `dynamic_range_global_flat_noise` | 5.506e+02 | 5.506e+02 | 9.913e-03 | 0.002% |
| `image_1` | `sector_1` | `min_rms_true_sky` | 8.438e-03 | 8.457e-03 | 1.934e-05 | 0.229% |
| `image_1` | `sector_1` | `median_rms_true_sky` | 1.622e-02 | 1.625e-02 | 3.625e-05 | 0.224% |
| `image_1` | `sector_1` | `dynamic_range_global_true_sky` | 5.424e+02 | 5.423e+02 | -2.843e-02 | -0.005% |

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

- FITS std differs for field-MFS-dirty.fits: 0.1729215907380038 != 0.1733087344430547
- FITS rms differs for field-MFS-dirty.fits: 0.17292200911591032 != 0.17330915375530434
- FITS min differs for field-MFS-dirty.fits: -0.7253921627998352 != -0.7270155549049377
- FITS max differs for field-MFS-dirty.fits: 4.537002086639404 != 4.5471601486206055
- FITS image pixels differ for field-MFS-dirty.fits: max_abs_delta=0.010158061981201172, p99_abs_delta=0.0009259283542633057, residual_rms=0.0003871472683442433
- FITS mean differs for field-MFS-image-pb-ast.fits: 0.0029854774769204648 != 0.0029921625990033513
- FITS std differs for field-MFS-image-pb-ast.fits: 0.07468696110305105 != 0.07485416453068121
- FITS rms differs for field-MFS-image-pb-ast.fits: 0.07474660684321302 != 0.07491394385963916
- FITS min differs for field-MFS-image-pb-ast.fits: -0.08084568381309509 != -0.08102613687515259
- FITS max differs for field-MFS-image-pb-ast.fits: 4.576261520385742 != 4.586507797241211
- FITS image pixels differ for field-MFS-image-pb-ast.fits: max_abs_delta=0.01024627685546875, p99_abs_delta=0.00010662898421287537, residual_rms=0.00016734330699861427
- FITS mean differs for field-MFS-image-pb.fits: 0.0029854774769204648 != 0.0029921625990033513
- FITS std differs for field-MFS-image-pb.fits: 0.07468696110305105 != 0.07485416453068121
- FITS rms differs for field-MFS-image-pb.fits: 0.07474660684321302 != 0.07491394385963916
- FITS min differs for field-MFS-image-pb.fits: -0.08084568381309509 != -0.08102613687515259
- FITS max differs for field-MFS-image-pb.fits: 4.576261520385742 != 4.586507797241211
- FITS image pixels differ for field-MFS-image-pb.fits: max_abs_delta=0.01024627685546875, p99_abs_delta=0.00010662898421287537, residual_rms=0.00016734330699861427
- FITS mean differs for field-MFS-image.fits: 0.002942199732113065 != 0.0029487885645885425
- FITS std differs for field-MFS-image.fits: 0.0737031980582531 != 0.07386820134167484
- FITS rms differs for field-MFS-image.fits: 0.07376190035023314 != 0.0739270351323036
- FITS min differs for field-MFS-image.fits: -0.08031634986400604 != -0.08049559593200684
- FITS max differs for field-MFS-image.fits: 4.576252460479736 != 4.586499214172363
- FITS image pixels differ for field-MFS-image.fits: max_abs_delta=0.010246753692626953, p99_abs_delta=0.00010457634925842285, residual_rms=0.0001651409095235927
- FITS std differs for field-MFS-model-pb.fits: 0.0017346313838326756 != 0.0017385154274867883
- FITS rms differs for field-MFS-model-pb.fits: 0.0017346341516429253 != 0.0017385182014938447
- FITS max differs for field-MFS-model-pb.fits: 1.5328218936920166 != 1.536255121231079
- FITS image pixels differ for field-MFS-model-pb.fits: max_abs_delta=0.0034332275390625, p99_abs_delta=0.0, residual_rms=3.884049982336277e-06
- FITS std differs for field-MFS-residual.fits: 0.01730897885234138 != 0.01734771282022354
- FITS rms differs for field-MFS-residual.fits: 0.01730898482278375 != 0.017347718803225318
- FITS min differs for field-MFS-residual.fits: -0.08031633496284485 != -0.08049559593200684
- FITS max differs for field-MFS-residual.fits: 0.22983074188232422 != 0.23034553229808807
- FITS image pixels differ for field-MFS-residual.fits: max_abs_delta=0.000514790415763855, p99_abs_delta=9.89772379398346e-05, residual_rms=3.876007219997778e-05
- HDF5 numeric dataset differs for fulljones-solutions.h5:sol000/amplitude000/val (max_abs=0.00113651)
- FITS table column differs for sector_1.source_catalog.fits:Total_flux
- FITS table column differs for sector_1.source_catalog.fits:E_Total_flux
- FITS table column differs for sector_1.source_catalog.fits:Peak_flux
- FITS table column differs for sector_1.source_catalog.fits:E_Peak_flux
- FITS table column differs for sector_1.source_catalog.fits:E_PA
- FITS table column differs for sector_1.source_catalog.fits:E_PA_img_plane
- FITS table column differs for sector_1.source_catalog.fits:E_DC_PA
- FITS table column differs for sector_1.source_catalog.fits:E_DC_PA_img_plane
- FITS table column differs for sector_1.source_catalog.fits:Isl_Total_flux
- FITS table column differs for sector_1.source_catalog.fits:E_Isl_Total_flux
- FITS table column differs for sector_1.source_catalog.fits:Isl_rms
- FITS table column differs for sector_1.source_catalog.fits:Resid_Isl_rms
- FITS table column differs for sector_1.source_catalog.fits:Resid_Isl_mean
- text product differs for sector_1_facets_ds9.reg
