# Rapthor Branch Equivalence

Scenario: `prediction-path-wsclean`
Run root: `/app/runs/science-gate-20260716-post-master-sync-option-matrix/prediction-path-wsclean`

## Branch Runs

| Side | Ref | Return Code | Elapsed (s) | Parset | Work Dir | Log | Input Snapshot |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| base | `master` | 0 | 229.986 | `/app/docs/source/development/science_equivalence_runs/2026-07-06-option-matrix/inputs/base/prediction_path_wsclean.parset` | `/tmp/ropwb` | `/app/runs/science-gate-20260716-post-master-sync-option-matrix/prediction-path-wsclean/base/rapthor-command.log` | parset: `inputs/base/prediction_path_wsclean.parset`, strategy: `inputs/base/prediction_path_wsclean_strategy.py` |
| current | `current` | 0 | 109.551 | `/app/docs/source/development/science_equivalence_runs/2026-07-06-option-matrix/inputs/current/prediction_path_wsclean.parset` | `/tmp/ropwc` | `/app/runs/science-gate-20260716-post-master-sync-option-matrix/prediction-path-wsclean/current/rapthor-command.log` | parset: `inputs/current/prediction_path_wsclean.parset`, strategy: `inputs/current/prediction_path_wsclean_strategy.py` |

## Runtime Summary

| Side | Runs | Min (s) | Median (s) | Max (s) |
| --- | ---: | ---: | ---: | ---: |
| master | 1 | 229.986 | 229.986 | 229.986 |
| current | 1 | 109.551 | 109.551 | 109.551 |

current-vs-master median delta: -52.366%

## Operation Runtime Summary

| Operation | master Runs | master Median (s) | current Runs | current Median (s) | Delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| `calibrate_1` | 1 | 67.583 | 1 | 13.170 | -80.513% |
| `image_1` | 1 | 154.405 | 1 | 75.250 | -51.264% |
| `mosaic_1` | 1 | 3.533 | 1 | 1.354 | -61.670% |

## Comparison Summary

| Result | Ops | Records | FITS | Image HDUs | Table HDUs | H5 | Text | Diagnostics | Visuals |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fail | 3 | 3 | 7 | 6 | 1 | 2 | 10 | 1 | 5 |

## FITS Residual Metrics

| Product | Max Abs Delta | P99 Abs Delta | Residual RMS | RMS / Ref RMS | RMS / Ref MAD |
| --- | ---: | ---: | ---: | ---: | ---: |
| `field-MFS-model-pb.fits` | 9.246e-02 | 0.000e+00 | 6.652e-05 | 3.615e-02 | n/a |
| `field-MFS-residual.fits` | 1.041e-03 | 3.176e-04 | 1.214e-04 | 6.997e-03 | 7.425e-03 |
| `field-MFS-image-pb-ast.fits` | 7.365e-04 | 3.218e-04 | 1.230e-04 | 1.647e-03 | 7.354e-03 |
| `field-MFS-image-pb.fits` | 7.365e-04 | 3.218e-04 | 1.230e-04 | 1.647e-03 | 7.354e-03 |
| `field-MFS-image.fits` | 7.181e-04 | 3.156e-04 | 1.206e-04 | 1.636e-03 | 7.355e-03 |
| `field-MFS-dirty.fits` | 5.565e-04 | 2.944e-04 | 1.127e-04 | 6.519e-04 | 7.261e-04 |

## Image Diagnostics

| Operation | Sector | Field | Reference | Current | Delta | Relative Delta |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `image_1` | `sector_1` | `nsources` | 1.000e+01 | 1.000e+01 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `theoretical_rms` | 9.006e-03 | 9.006e-03 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `min_rms_flat_noise` | 8.927e-03 | 8.917e-03 | -1.029e-05 | -0.115% |
| `image_1` | `sector_1` | `median_rms_flat_noise` | 1.593e-02 | 1.592e-02 | -4.595e-06 | -0.029% |
| `image_1` | `sector_1` | `dynamic_range_global_flat_noise` | 5.105e+02 | 5.111e+02 | 5.811e-01 | 0.114% |
| `image_1` | `sector_1` | `min_rms_true_sky` | 9.071e-03 | 9.063e-03 | -8.102e-06 | -0.089% |
| `image_1` | `sector_1` | `median_rms_true_sky` | 1.622e-02 | 1.621e-02 | -4.701e-06 | -0.029% |
| `image_1` | `sector_1` | `dynamic_range_global_true_sky` | 5.024e+02 | 5.029e+02 | 4.414e-01 | 0.088% |

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


## Difference Classification

| Category | Disposition | Count | Examples | Recommendation |
| --- | --- | ---: | --- | --- |
| `output_record_auxiliary_artifacts` | warning | 1 | `calibrate_1` | Keep as non-blocking when only plot filenames or known local intermediate aliases differ and final scientific products pass. |
| `pybdsf_catalog_diagnostic_column` | repeatability-candidate | 16 | `sector_1.source_catalog.fits:E_Total_flux`, `sector_1.source_catalog.fits:E_Peak_flux`, `sector_1.source_catalog.fits:E_Maj` | Keep source count and primary flux/position strict; bound fitted uncertainty/shape/noise columns with same-branch PyBDSF scatter. |
| `small_image_residual` | repeatability-candidate | 1 | `field-MFS-dirty.fits` | Bound with same-branch image residual scatter before turning this into a tolerance. |
| `sparse_model_image_residual` | repeatability-candidate | 1 | `field-MFS-model-pb.fits` | Treat as a sparse model-component residual only after same-branch repeatability bounds it. |
| `sparse_model_image_statistic` | repeatability-candidate | 3 | `field-MFS-model-pb.fits:std`, `field-MFS-model-pb.fits:rms`, `field-MFS-model-pb.fits:max` | Judge sparse model-image statistics with same-branch repeatability, p99/RMS residuals, and downstream diagnostics. |
| `strict_contract_difference` | strict-failure | 2 | `sky-model summary differs for sector_1.apparent_sky.txt`, `sky-model summary differs for sector_1.true_sky.txt` | Investigate before accepting this branch difference. |
| `strict_fits_image_difference` | strict-failure | 4 | `field-MFS-image-pb-ast.fits`, `field-MFS-image-pb.fits`, `field-MFS-image.fits` | Investigate before relaxing comparison rules. |
| `strict_fits_table_column` | strict-failure | 4 | `sector_1.source_catalog.fits:DC_Maj`, `sector_1.source_catalog.fits:DC_PA`, `sector_1.source_catalog.fits:DC_Maj_img_plane` | Investigate catalog value drift before relaxing. |
| `strict_h5_difference` | strict-failure | 2 | `HDF5 numeric dataset differs for field-solutions-fast-phase.h5:sol000/phase000/val (max_abs=0.00652681)`, `HDF5 numeric dataset differs for field-solutions.h5:sol000/phase000/val (max_abs=0.00652681)` | Keep h5parm structure and datasets strict unless justified. |

## Warnings

- output-record auxiliary artifact basenames differ for calibrate_1

## Failures

- FITS image pixels differ for field-MFS-dirty.fits: max_abs_delta=0.0005564689636230469, p99_abs_delta=0.00029439294710755307, residual_rms=0.00011273192738016786
- FITS image pixels differ for field-MFS-image-pb-ast.fits: max_abs_delta=0.0007364749908447266, p99_abs_delta=0.00032177940243855114, residual_rms=0.000123039714725054
- FITS image pixels differ for field-MFS-image-pb.fits: max_abs_delta=0.0007364749908447266, p99_abs_delta=0.00032177940243855114, residual_rms=0.000123039714725054
- FITS image pixels differ for field-MFS-image.fits: max_abs_delta=0.0007181167602539062, p99_abs_delta=0.0003156159259378906, residual_rms=0.00012056873451934334
- FITS std differs for field-MFS-model-pb.fits: 0.0018402236992041453 != 0.0018305208786147272
- FITS rms differs for field-MFS-model-pb.fits: 0.0018402263076896105 != 0.001830523500843655
- FITS max differs for field-MFS-model-pb.fits: 2.0011019706726074 != 1.9179260730743408
- FITS image pixels differ for field-MFS-model-pb.fits: max_abs_delta=0.09246370196342468, p99_abs_delta=0.0, residual_rms=6.651930609158191e-05
- FITS image pixels differ for field-MFS-residual.fits: max_abs_delta=0.001041226089000702, p99_abs_delta=0.0003176066232845184, residual_rms=0.00012144034942236171
- HDF5 numeric dataset differs for field-solutions-fast-phase.h5:sol000/phase000/val (max_abs=0.00652681)
- HDF5 numeric dataset differs for field-solutions.h5:sol000/phase000/val (max_abs=0.00652681)
- sky-model summary differs for sector_1.apparent_sky.txt
- FITS table column differs for sector_1.source_catalog.fits:E_Total_flux
- FITS table column differs for sector_1.source_catalog.fits:E_Peak_flux
- FITS table column differs for sector_1.source_catalog.fits:E_Maj
- FITS table column differs for sector_1.source_catalog.fits:PA
- FITS table column differs for sector_1.source_catalog.fits:E_PA
- FITS table column differs for sector_1.source_catalog.fits:E_Maj_img_plane
- FITS table column differs for sector_1.source_catalog.fits:PA_img_plane
- FITS table column differs for sector_1.source_catalog.fits:E_PA_img_plane
- FITS table column differs for sector_1.source_catalog.fits:DC_Maj
- FITS table column differs for sector_1.source_catalog.fits:E_DC_Maj
- FITS table column differs for sector_1.source_catalog.fits:DC_PA
- FITS table column differs for sector_1.source_catalog.fits:E_DC_PA
- FITS table column differs for sector_1.source_catalog.fits:DC_Maj_img_plane
- FITS table column differs for sector_1.source_catalog.fits:E_DC_Maj_img_plane
- FITS table column differs for sector_1.source_catalog.fits:DC_PA_img_plane
- FITS table column differs for sector_1.source_catalog.fits:E_DC_PA_img_plane
- FITS table column differs for sector_1.source_catalog.fits:E_Isl_Total_flux
- FITS table column differs for sector_1.source_catalog.fits:Isl_rms
- FITS table column differs for sector_1.source_catalog.fits:Resid_Isl_rms
- FITS table column differs for sector_1.source_catalog.fits:Resid_Isl_mean
- sky-model summary differs for sector_1.true_sky.txt
