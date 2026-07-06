# Rapthor Branch Equivalence

Scenario: `dd-phase-plus-di-fulljones-normalized-smoke`
Run root: `/tmp/rfjn`

## Branch Runs

| Side | Ref | Return Code | Parset | Work Dir | Log | Input Snapshot |
| --- | --- | ---: | --- | --- | --- | --- |
| base | `master` | 0 | `/tmp/rfjn/inputs/base/master_dd_phase_plus_di_fulljones.parset` | `/tmp/rfjnw/base` | `/tmp/rfjn/base/rapthor-command.log` | parset: `inputs/base/master_dd_phase_plus_di_fulljones.parset`, strategy: `inputs/base/master_dd_phase_plus_di_fulljones_strategy.py` |
| current | `current` | 0 | `/tmp/rfjn/inputs/current/current_dd_phase_plus_di_fulljones.parset` | `/tmp/rfjnw/current` | `/tmp/rfjn/current/rapthor-command.log` | parset: `inputs/current/current_dd_phase_plus_di_fulljones.parset`, strategy: `inputs/current/current_dd_phase_plus_di_fulljones_strategy.py` |

## Comparison Summary

| Result | Ops | Records | FITS | Image HDUs | Table HDUs | H5 | Text | Diagnostics | Visuals |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fail | 5 | 5 | 7 | 6 | 1 | 3 | 10 | 1 | 0 |

## FITS Residual Metrics

| Product | Max Abs Delta | P99 Abs Delta | Residual RMS | RMS / Ref RMS | RMS / Ref MAD |
| --- | ---: | ---: | ---: | ---: | ---: |
| `field-MFS-model-pb.fits` | 2.739e-03 | 0.000e+00 | 1.614e-06 | 9.303e-04 | n/a |
| `field-MFS-residual.fits` | 1.313e-04 | 1.049e-05 | 4.194e-06 | 2.423e-04 | 2.588e-04 |
| `field-MFS-image-pb-ast.fits` | 2.486e-05 | 1.059e-05 | 3.978e-06 | 5.322e-05 | 2.399e-04 |
| `field-MFS-image-pb.fits` | 2.486e-05 | 1.059e-05 | 3.978e-06 | 5.322e-05 | 2.399e-04 |
| `field-MFS-image.fits` | 2.483e-05 | 1.045e-05 | 3.907e-06 | 5.297e-05 | 2.405e-04 |
| `field-MFS-dirty.fits` | 9.537e-07 | 8.196e-08 | 2.916e-08 | 1.687e-07 | 1.883e-07 |

## Image Diagnostics

| Operation | Sector | Field | Reference | Current | Delta | Relative Delta |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `image_1` | `sector_1` | `nsources` | 1.100e+01 | 1.100e+01 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `theoretical_rms` | 9.006e-03 | 9.006e-03 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `min_rms_flat_noise` | 8.313e-03 | 8.311e-03 | -1.770e-06 | -0.021% |
| `image_1` | `sector_1` | `median_rms_flat_noise` | 1.589e-02 | 1.589e-02 | 2.794e-08 | 0.000% |
| `image_1` | `sector_1` | `dynamic_range_global_flat_noise` | 5.505e+02 | 5.506e+02 | 1.169e-01 | 0.021% |
| `image_1` | `sector_1` | `min_rms_true_sky` | 8.438e-03 | 8.438e-03 | 4.470e-08 | 0.001% |
| `image_1` | `sector_1` | `median_rms_true_sky` | 1.622e-02 | 1.622e-02 | -1.099e-07 | -0.001% |
| `image_1` | `sector_1` | `dynamic_range_global_true_sky` | 5.424e+02 | 5.424e+02 | -3.156e-03 | -0.001% |

## Visual Comparisons

Visual PNGs were generated under the temporary run root but are not tracked;
the numeric residual and diagnostic tables above are the retained review
evidence.

## Warnings

- output-record summary differs for calibrate_1
- output-record summary differs for calibrate_di_1

## Failures

- FITS image pixels differ for field-MFS-image-pb-ast.fits: max_abs_delta=2.485513687133789e-05, p99_abs_delta=1.0585412383079529e-05, residual_rms=3.978058685467038e-06
- FITS image pixels differ for field-MFS-image-pb.fits: max_abs_delta=2.485513687133789e-05, p99_abs_delta=1.0585412383079529e-05, residual_rms=3.978058685467038e-06
- FITS image pixels differ for field-MFS-image.fits: max_abs_delta=2.483278512954712e-05, p99_abs_delta=1.0449439287185669e-05, residual_rms=3.907291178358995e-06
- FITS image pixels differ for field-MFS-model-pb.fits: max_abs_delta=0.002739250659942627, p99_abs_delta=0.0, residual_rms=1.6136456129153772e-06
- FITS image pixels differ for field-MFS-residual.fits: max_abs_delta=0.00013128668069839478, p99_abs_delta=1.0491349967196558e-05, residual_rms=4.193655155815742e-06
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
- FITS table column differs for sector_1.source_catalog.fits:Resid_Isl_mean
- text product differs for sector_1_facets_ds9.reg
