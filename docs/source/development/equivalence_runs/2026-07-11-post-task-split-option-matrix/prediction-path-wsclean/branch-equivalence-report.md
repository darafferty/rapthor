# Rapthor Branch Equivalence

Scenario: `prediction-path-wsclean`
Run root: `/app/runs/branch-option-matrix-20260711-post-task-split-rerun/prediction-path-wsclean`

## Branch Runs

| Side | Ref | Return Code | Parset | Work Dir | Log | Input Snapshot |
| --- | --- | ---: | --- | --- | --- | --- |
| base | `master` | 0 | `/app/docs/source/development/equivalence_runs/2026-07-06-option-matrix/inputs/base/prediction_path_wsclean.parset` | `/tmp/ropwb` | `/app/runs/branch-option-matrix-20260711-post-task-split-rerun/prediction-path-wsclean/base/rapthor-command.log` | parset: `inputs/base/prediction_path_wsclean.parset`, strategy: `inputs/base/prediction_path_wsclean_strategy.py` |
| current | `current` | 0 | `/app/docs/source/development/equivalence_runs/2026-07-06-option-matrix/inputs/current/prediction_path_wsclean.parset` | `/tmp/ropwc` | `/app/runs/branch-option-matrix-20260711-post-task-split-rerun/prediction-path-wsclean/current/rapthor-command.log` | parset: `inputs/current/prediction_path_wsclean.parset`, strategy: `inputs/current/prediction_path_wsclean_strategy.py` |

## Comparison Summary

| Result | Ops | Records | FITS | Image HDUs | Table HDUs | H5 | Text | Diagnostics | Visuals |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pass | 3 | 3 | 7 | 6 | 1 | 2 | 10 | 1 | 5 |

## FITS Residual Metrics

| Product | Max Abs Delta | P99 Abs Delta | Residual RMS | RMS / Ref RMS | RMS / Ref MAD |
| --- | ---: | ---: | ---: | ---: | ---: |
| `field-MFS-dirty.fits` | 9.537e-07 | 7.078e-08 | 2.569e-08 | 1.485e-07 | 1.654e-07 |
| `field-MFS-image-pb-ast.fits` | 9.537e-07 | 7.264e-08 | 2.518e-08 | 3.371e-07 | 1.505e-06 |
| `field-MFS-image-pb.fits` | 9.537e-07 | 7.264e-08 | 2.518e-08 | 3.371e-07 | 1.505e-06 |
| `field-MFS-image.fits` | 9.537e-07 | 7.125e-08 | 2.454e-08 | 3.330e-07 | 1.497e-06 |
| `field-MFS-model-pb.fits` | 2.384e-07 | 0.000e+00 | 1.894e-10 | 1.029e-07 | n/a |
| `field-MFS-residual.fits` | 1.341e-07 | 4.284e-08 | 1.623e-08 | 9.353e-07 | 9.924e-07 |

## Image Diagnostics

| Operation | Sector | Field | Reference | Current | Delta | Relative Delta |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `image_1` | `sector_1` | `nsources` | 1.000e+01 | 1.000e+01 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `theoretical_rms` | 9.006e-03 | 9.006e-03 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `min_rms_flat_noise` | 8.918e-03 | 8.919e-03 | 1.207e-06 | 0.014% |
| `image_1` | `sector_1` | `median_rms_flat_noise` | 1.593e-02 | 1.593e-02 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `dynamic_range_global_flat_noise` | 5.110e+02 | 5.109e+02 | -6.910e-02 | -0.014% |
| `image_1` | `sector_1` | `min_rms_true_sky` | 9.064e-03 | 9.062e-03 | -2.408e-06 | -0.027% |
| `image_1` | `sector_1` | `median_rms_true_sky` | 1.622e-02 | 1.622e-02 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `dynamic_range_global_true_sky` | 5.028e+02 | 5.029e+02 | 1.336e-01 | 0.027% |

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

## Warnings

- output-record auxiliary artifact basenames differ for calibrate_1
