# Rapthor Branch Equivalence

Scenario: `august-master-sync-normalization-rich-demo`
Run root: `/app/runs/branch-equivalence-20260804-august-master-sync-normalization`

## Branch Runs

| Side | Ref | Return Code | Elapsed (s) | Parset | Work Dir | Log | Input Snapshot |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| base | `origin/master` | 0 | 203.286 | `/app/docs/source/development/science_equivalence_runs/2026-07-06-option-matrix/inputs/base/normalization_rich_demo.parset` | `/tmp/rom-base` | `/app/runs/branch-equivalence-20260804-august-master-sync-normalization/base/rapthor-command.log` | parset: `inputs/base/normalization_rich_demo.parset`, strategy: `inputs/base/normalization_rich_demo_strategy.py` |
| current | `current` | 0 | 99.954 | `/app/docs/source/development/science_equivalence_runs/2026-07-06-option-matrix/inputs/current/normalization_rich_demo.parset` | `/tmp/rom-current` | `/app/runs/branch-equivalence-20260804-august-master-sync-normalization/current/rapthor-command.log` | parset: `inputs/current/normalization_rich_demo.parset`, strategy: `inputs/current/normalization_rich_demo_strategy.py` |

## Runtime Summary

| Side | Runs | Min (s) | Median (s) | Max (s) |
| --- | ---: | ---: | ---: | ---: |
| origin/master | 1 | 203.286 | 203.286 | 203.286 |
| current | 1 | 99.954 | 99.954 | 99.954 |

current-vs-origin/master median delta: -50.831%

## Operation Runtime Summary

| Operation | origin/master Runs | origin/master Median (s) | current Runs | current Median (s) | Delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| `calibrate_1` | 1 | 38.441 | 1 | 5.135 | -86.642% |
| `image_1` | 1 | 101.478 | 1 | 65.152 | -35.797% |
| `mosaic_1` | 1 | 4.319 | 1 | 1.099 | -74.564% |
| `normalize_1` | 1 | 56.474 | 1 | 12.120 | -78.540% |

## Comparison Summary

| Result | Ops | Records | FITS | Image HDUs | Table HDUs | H5 | Text | Diagnostics | Visuals |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pass | 4 | 4 | 8 | 7 | 1 | 3 | 12 | 1 | 5 |

## FITS Residual Metrics

| Product | Max Abs Delta | P99 Abs Delta | Residual RMS | RMS / Ref RMS | RMS / Ref MAD |
| --- | ---: | ---: | ---: | ---: | ---: |
| `field-MFS-dirty.fits` | 9.537e-07 | 5.960e-08 | 2.373e-08 | 1.678e-07 | 1.872e-07 |
| `sector_1_I_freq_cube.fits` | 9.537e-07 | 1.204e-07 | 4.187e-08 | 4.945e-07 | 1.466e-06 |
| `field-MFS-image-pb-ast.fits` | 7.153e-07 | 5.960e-08 | 2.092e-08 | 3.405e-07 | 1.377e-06 |
| `field-MFS-image-pb.fits` | 7.153e-07 | 5.960e-08 | 2.092e-08 | 3.405e-07 | 1.377e-06 |
| `field-MFS-image.fits` | 7.153e-07 | 5.960e-08 | 2.063e-08 | 3.404e-07 | 1.386e-06 |
| `field-MFS-residual.fits` | 2.384e-07 | 3.912e-08 | 1.507e-08 | 9.615e-07 | 1.014e-06 |
| `field-MFS-model-pb.fits` | 1.788e-07 | 0.000e+00 | 1.479e-10 | 1.013e-07 | n/a |

## Image Diagnostics

| Operation | Sector | Field | Reference | Current | Delta | Relative Delta |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `image_1` | `sector_1` | `nsources` | 1.000e+01 | 1.000e+01 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `theoretical_rms` | 9.006e-03 | 9.006e-03 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `min_rms_flat_noise` | 7.535e-03 | 7.535e-03 | 2.198e-07 | 0.003% |
| `image_1` | `sector_1` | `median_rms_flat_noise` | 1.458e-02 | 1.458e-02 | -1.863e-09 | -0.000% |
| `image_1` | `sector_1` | `dynamic_range_global_flat_noise` | 4.959e+02 | 4.959e+02 | -1.450e-02 | -0.003% |
| `image_1` | `sector_1` | `min_rms_true_sky` | 7.632e-03 | 7.631e-03 | -7.683e-07 | -0.010% |
| `image_1` | `sector_1` | `median_rms_true_sky` | 1.485e-02 | 1.485e-02 | -9.313e-10 | -0.000% |
| `image_1` | `sector_1` | `dynamic_range_global_true_sky` | 4.896e+02 | 4.896e+02 | 4.926e-02 | 0.010% |

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

