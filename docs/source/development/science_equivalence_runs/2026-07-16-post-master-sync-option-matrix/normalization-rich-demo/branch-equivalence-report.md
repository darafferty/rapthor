# Rapthor Branch Equivalence

Scenario: `normalization-rich-demo`
Run root: `/app/runs/science-gate-20260716-post-master-sync-option-matrix/normalization-rich-demo`

## Branch Runs

| Side | Ref | Return Code | Elapsed (s) | Parset | Work Dir | Log | Input Snapshot |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| base | `master` | 0 | 294.227 | `/app/docs/source/development/science_equivalence_runs/2026-07-06-option-matrix/inputs/base/normalization_rich_demo.parset` | `/tmp/rom-base` | `/app/runs/science-gate-20260716-post-master-sync-option-matrix/normalization-rich-demo/base/rapthor-command.log` | parset: `inputs/base/normalization_rich_demo.parset`, strategy: `inputs/base/normalization_rich_demo_strategy.py` |
| current | `current` | 0 | 151.667 | `/app/docs/source/development/science_equivalence_runs/2026-07-06-option-matrix/inputs/current/normalization_rich_demo.parset` | `/tmp/rom-current` | `/app/runs/science-gate-20260716-post-master-sync-option-matrix/normalization-rich-demo/current/rapthor-command.log` | parset: `inputs/current/normalization_rich_demo.parset`, strategy: `inputs/current/normalization_rich_demo_strategy.py` |

## Runtime Summary

| Side | Runs | Min (s) | Median (s) | Max (s) |
| --- | ---: | ---: | ---: | ---: |
| master | 1 | 294.227 | 294.227 | 294.227 |
| current | 1 | 151.667 | 151.667 | 151.667 |

current-vs-master median delta: -48.452%

## Operation Runtime Summary

| Operation | master Runs | master Median (s) | current Runs | current Median (s) | Delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| `calibrate_1` | 1 | 36.204 | 1 | 7.193 | -80.133% |
| `image_1` | 1 | 155.737 | 1 | 94.495 | -39.324% |
| `mosaic_1` | 1 | 3.824 | 1 | 1.137 | -70.267% |
| `normalize_1` | 1 | 93.258 | 1 | 26.231 | -71.872% |

## Comparison Summary

| Result | Ops | Records | FITS | Image HDUs | Table HDUs | H5 | Text | Diagnostics | Visuals |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pass | 4 | 4 | 8 | 7 | 1 | 3 | 12 | 1 | 5 |

## FITS Residual Metrics

| Product | Max Abs Delta | P99 Abs Delta | Residual RMS | RMS / Ref RMS | RMS / Ref MAD |
| --- | ---: | ---: | ---: | ---: | ---: |
| `field-MFS-image-pb-ast.fits` | 1.192e-06 | 6.333e-08 | 2.381e-08 | 3.876e-07 | 1.567e-06 |
| `field-MFS-image-pb.fits` | 1.192e-06 | 6.333e-08 | 2.381e-08 | 3.876e-07 | 1.567e-06 |
| `field-MFS-dirty.fits` | 9.537e-07 | 5.960e-08 | 2.431e-08 | 1.719e-07 | 1.918e-07 |
| `sector_1_I_freq_cube.fits` | 9.537e-07 | 1.061e-07 | 3.681e-08 | 4.348e-07 | 1.289e-06 |
| `field-MFS-image.fits` | 7.153e-07 | 6.054e-08 | 2.227e-08 | 3.673e-07 | 1.495e-06 |
| `field-MFS-model-pb.fits` | 2.384e-07 | 0.000e+00 | 1.944e-10 | 1.332e-07 | n/a |
| `field-MFS-residual.fits` | 2.086e-07 | 4.098e-08 | 1.559e-08 | 9.945e-07 | 1.049e-06 |

## Image Diagnostics

| Operation | Sector | Field | Reference | Current | Delta | Relative Delta |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `image_1` | `sector_1` | `nsources` | 1.000e+01 | 1.000e+01 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `theoretical_rms` | 9.006e-03 | 9.006e-03 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `min_rms_flat_noise` | 7.534e-03 | 7.534e-03 | 5.309e-08 | 0.001% |
| `image_1` | `sector_1` | `median_rms_flat_noise` | 1.458e-02 | 1.458e-02 | -2.794e-09 | -0.000% |
| `image_1` | `sector_1` | `dynamic_range_global_flat_noise` | 4.959e+02 | 4.959e+02 | -3.462e-03 | -0.001% |
| `image_1` | `sector_1` | `min_rms_true_sky` | 7.632e-03 | 7.632e-03 | -1.527e-07 | -0.002% |
| `image_1` | `sector_1` | `median_rms_true_sky` | 1.485e-02 | 1.485e-02 | 2.794e-09 | 0.000% |
| `image_1` | `sector_1` | `dynamic_range_global_true_sky` | 4.896e+02 | 4.896e+02 | 9.892e-03 | 0.002% |

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
