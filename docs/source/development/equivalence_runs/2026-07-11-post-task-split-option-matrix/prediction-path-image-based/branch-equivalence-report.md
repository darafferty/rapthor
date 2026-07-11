# Rapthor Branch Equivalence

Scenario: `prediction-path-image-based`
Run root: `/app/runs/branch-option-matrix-20260711-post-task-split-rerun/prediction-path-image-based`

## Branch Runs

| Side | Ref | Return Code | Parset | Work Dir | Log | Input Snapshot |
| --- | --- | ---: | --- | --- | --- | --- |
| base | `master` | 0 | `/app/docs/source/development/equivalence_runs/2026-07-06-option-matrix/inputs/base/prediction_path_image_based.parset` | `/tmp/ropib` | `/app/runs/branch-option-matrix-20260711-post-task-split-rerun/prediction-path-image-based/base/rapthor-command.log` | parset: `inputs/base/prediction_path_image_based.parset`, strategy: `inputs/base/prediction_path_image_based_strategy.py` |
| current | `current` | 0 | `/app/docs/source/development/equivalence_runs/2026-07-06-option-matrix/inputs/current/prediction_path_image_based.parset` | `/tmp/ropic` | `/app/runs/branch-option-matrix-20260711-post-task-split-rerun/prediction-path-image-based/current/rapthor-command.log` | parset: `inputs/current/prediction_path_image_based.parset`, strategy: `inputs/current/prediction_path_image_based_strategy.py` |

## Comparison Summary

| Result | Ops | Records | FITS | Image HDUs | Table HDUs | H5 | Text | Diagnostics | Visuals |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pass | 3 | 3 | 7 | 6 | 1 | 2 | 10 | 1 | 5 |

## FITS Residual Metrics

| Product | Max Abs Delta | P99 Abs Delta | Residual RMS | RMS / Ref RMS | RMS / Ref MAD |
| --- | ---: | ---: | ---: | ---: | ---: |
| `field-MFS-dirty.fits` | 9.537e-07 | 7.078e-08 | 2.564e-08 | 1.483e-07 | 1.653e-07 |
| `field-MFS-image-pb-ast.fits` | 9.537e-07 | 7.264e-08 | 2.573e-08 | 3.445e-07 | 1.579e-06 |
| `field-MFS-image-pb.fits` | 9.537e-07 | 7.264e-08 | 2.573e-08 | 3.445e-07 | 1.579e-06 |
| `field-MFS-image.fits` | 9.537e-07 | 7.171e-08 | 2.531e-08 | 3.434e-07 | 1.585e-06 |
| `field-MFS-residual.fits` | 1.304e-07 | 4.470e-08 | 1.702e-08 | 1.001e-06 | 1.067e-06 |
| `field-MFS-model-pb.fits` | 1.192e-07 | 0.000e+00 | 1.386e-10 | 7.647e-08 | n/a |

## Image Diagnostics

| Operation | Sector | Field | Reference | Current | Delta | Relative Delta |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `image_1` | `sector_1` | `nsources` | 1.000e+01 | 1.000e+01 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `theoretical_rms` | 9.006e-03 | 9.006e-03 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `min_rms_flat_noise` | 8.515e-03 | 8.515e-03 | 9.313e-10 | 0.000% |
| `image_1` | `sector_1` | `median_rms_flat_noise` | 1.555e-02 | 1.555e-02 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `dynamic_range_global_flat_noise` | 5.368e+02 | 5.368e+02 | -5.872e-05 | -0.000% |
| `image_1` | `sector_1` | `min_rms_true_sky` | 8.618e-03 | 8.618e-03 | 2.757e-07 | 0.003% |
| `image_1` | `sector_1` | `median_rms_true_sky` | 1.584e-02 | 1.584e-02 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `dynamic_range_global_true_sky` | 5.304e+02 | 5.304e+02 | -1.696e-02 | -0.003% |

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
