# Rapthor Branch Equivalence

Scenario: `normalization-rich-demo`
Run root: `/app/runs/branch-option-matrix-20260711-post-task-split-rerun/normalization-rich-demo`

## Branch Runs

| Side | Ref | Return Code | Parset | Work Dir | Log | Input Snapshot |
| --- | --- | ---: | --- | --- | --- | --- |
| base | `master` | 0 | `/app/docs/source/development/equivalence_runs/2026-07-06-option-matrix/inputs/base/normalization_rich_demo.parset` | `/tmp/rom-base` | `/app/runs/branch-option-matrix-20260711-post-task-split-rerun/normalization-rich-demo/base/rapthor-command.log` | parset: `inputs/base/normalization_rich_demo.parset`, strategy: `inputs/base/normalization_rich_demo_strategy.py` |
| current | `current` | 0 | `/app/docs/source/development/equivalence_runs/2026-07-06-option-matrix/inputs/current/normalization_rich_demo.parset` | `/tmp/rom-current` | `/app/runs/branch-option-matrix-20260711-post-task-split-rerun/normalization-rich-demo/current/rapthor-command.log` | parset: `inputs/current/normalization_rich_demo.parset`, strategy: `inputs/current/normalization_rich_demo_strategy.py` |

## Comparison Summary

| Result | Ops | Records | FITS | Image HDUs | Table HDUs | H5 | Text | Diagnostics | Visuals |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pass | 4 | 4 | 8 | 7 | 1 | 3 | 12 | 1 | 5 |

## FITS Residual Metrics

| Product | Max Abs Delta | P99 Abs Delta | Residual RMS | RMS / Ref RMS | RMS / Ref MAD |
| --- | ---: | ---: | ---: | ---: | ---: |
| `sector_1_I_freq_cube.fits` | 1.192e-06 | 1.155e-07 | 4.074e-08 | 4.812e-07 | 1.427e-06 |
| `field-MFS-dirty.fits` | 7.153e-07 | 5.960e-08 | 2.196e-08 | 1.553e-07 | 1.732e-07 |
| `field-MFS-image-pb-ast.fits` | 7.153e-07 | 5.867e-08 | 2.053e-08 | 3.342e-07 | 1.351e-06 |
| `field-MFS-image-pb.fits` | 7.153e-07 | 5.867e-08 | 2.053e-08 | 3.342e-07 | 1.351e-06 |
| `field-MFS-image.fits` | 7.153e-07 | 5.867e-08 | 2.030e-08 | 3.348e-07 | 1.363e-06 |
| `field-MFS-model-pb.fits` | 2.384e-07 | 0.000e+00 | 1.330e-10 | 9.112e-08 | n/a |
| `field-MFS-residual.fits` | 1.043e-07 | 3.632e-08 | 1.374e-08 | 8.767e-07 | 9.250e-07 |

## Image Diagnostics

| Operation | Sector | Field | Reference | Current | Delta | Relative Delta |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `image_1` | `sector_1` | `nsources` | 1.000e+01 | 1.000e+01 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `theoretical_rms` | 9.006e-03 | 9.006e-03 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `min_rms_flat_noise` | 7.544e-03 | 7.544e-03 | 4.247e-07 | 0.006% |
| `image_1` | `sector_1` | `median_rms_flat_noise` | 1.458e-02 | 1.458e-02 | 9.313e-10 | 0.000% |
| `image_1` | `sector_1` | `dynamic_range_global_flat_noise` | 4.953e+02 | 4.953e+02 | -2.788e-02 | -0.006% |
| `image_1` | `sector_1` | `min_rms_true_sky` | 7.631e-03 | 7.632e-03 | 7.632e-07 | 0.010% |
| `image_1` | `sector_1` | `median_rms_true_sky` | 1.485e-02 | 1.485e-02 | 9.313e-10 | 0.000% |
| `image_1` | `sector_1` | `dynamic_range_global_true_sky` | 4.896e+02 | 4.896e+02 | -4.896e-02 | -0.010% |

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
