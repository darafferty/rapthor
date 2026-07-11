# Rapthor Branch Equivalence

Scenario: `bda-averaging`
Run root: `/app/runs/branch-option-matrix-20260711-post-task-split-rerun/bda-averaging`

## Branch Runs

| Side | Ref | Return Code | Parset | Work Dir | Log | Input Snapshot |
| --- | --- | ---: | --- | --- | --- | --- |
| base | `master` | 0 | `/app/docs/source/development/equivalence_runs/2026-07-06-option-matrix/inputs/base/bda_averaging.parset` | `/tmp/robb` | `/app/runs/branch-option-matrix-20260711-post-task-split-rerun/bda-averaging/base/rapthor-command.log` | parset: `inputs/base/bda_averaging.parset`, strategy: `inputs/base/bda_averaging_strategy.py` |
| current | `current` | 0 | `/app/docs/source/development/equivalence_runs/2026-07-06-option-matrix/inputs/current/bda_averaging.parset` | `/tmp/robc` | `/app/runs/branch-option-matrix-20260711-post-task-split-rerun/bda-averaging/current/rapthor-command.log` | parset: `inputs/current/bda_averaging.parset`, strategy: `inputs/current/bda_averaging_strategy.py` |

## Comparison Summary

| Result | Ops | Records | FITS | Image HDUs | Table HDUs | H5 | Text | Diagnostics | Visuals |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pass | 3 | 3 | 7 | 6 | 1 | 2 | 10 | 1 | 5 |

## FITS Residual Metrics

| Product | Max Abs Delta | P99 Abs Delta | Residual RMS | RMS / Ref RMS | RMS / Ref MAD |
| --- | ---: | ---: | ---: | ---: | ---: |
| `field-MFS-dirty.fits` | 9.537e-07 | 7.451e-08 | 2.776e-08 | 1.508e-07 | 1.600e-07 |
| `field-MFS-image-pb-ast.fits` | 9.537e-07 | 7.451e-08 | 2.746e-08 | 3.402e-07 | 5.526e-07 |
| `field-MFS-image-pb.fits` | 9.537e-07 | 7.451e-08 | 2.746e-08 | 3.402e-07 | 5.526e-07 |
| `field-MFS-image.fits` | 9.537e-07 | 7.264e-08 | 2.703e-08 | 3.396e-07 | 5.551e-07 |
| `field-MFS-residual.fits` | 1.490e-07 | 6.193e-08 | 2.388e-08 | 4.752e-07 | 4.919e-07 |
| `field-MFS-model-pb.fits` | 1.192e-07 | 0.000e+00 | 9.357e-11 | 6.712e-08 | n/a |

## Image Diagnostics

| Operation | Sector | Field | Reference | Current | Delta | Relative Delta |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `image_1` | `sector_1` | `nsources` | 1.500e+01 | 1.500e+01 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `theoretical_rms` | 9.006e-03 | 9.006e-03 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `min_rms_flat_noise` | 3.180e-02 | 3.180e-02 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `median_rms_flat_noise` | 4.735e-02 | 4.735e-02 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `dynamic_range_global_flat_noise` | 1.432e+02 | 1.432e+02 | -1.500e-05 | -0.000% |
| `image_1` | `sector_1` | `min_rms_true_sky` | 3.233e-02 | 3.233e-02 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `median_rms_true_sky` | 4.836e-02 | 4.836e-02 | -3.725e-09 | -0.000% |
| `image_1` | `sector_1` | `dynamic_range_global_true_sky` | 1.408e+02 | 1.408e+02 | 0.000e+00 | 0.000% |

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
