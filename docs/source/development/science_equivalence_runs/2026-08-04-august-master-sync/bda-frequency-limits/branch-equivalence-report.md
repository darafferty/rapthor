# Rapthor Branch Equivalence

Scenario: `august-master-sync-bda-frequency-limits`
Run root: `/app/runs/branch-equivalence-20260804-august-master-sync-bda-frequency-limits`

## Branch Runs

| Side | Ref | Return Code | Elapsed (s) | Parset | Work Dir | Log | Input Snapshot |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| base | `origin/master` | 0 | 157.162 | `/app/docs/source/development/science_equivalence_runs/2026-08-04-august-master-sync/inputs/base/bda_frequency_limits.parset` | `/tmp/a8b` | `/app/runs/branch-equivalence-20260804-august-master-sync-bda-frequency-limits/base/rapthor-command.log` | parset: `inputs/base/bda_frequency_limits.parset`, strategy: `inputs/base/bda_frequency_limits_strategy.py` |
| current | `current` | 0 | 83.337 | `/app/docs/source/development/science_equivalence_runs/2026-08-04-august-master-sync/inputs/current/bda_frequency_limits.parset` | `/tmp/a8c` | `/app/runs/branch-equivalence-20260804-august-master-sync-bda-frequency-limits/current/rapthor-command.log` | parset: `inputs/current/bda_frequency_limits.parset`, strategy: `inputs/current/bda_frequency_limits_strategy.py` |

## Runtime Summary

| Side | Runs | Min (s) | Median (s) | Max (s) |
| --- | ---: | ---: | ---: | ---: |
| origin/master | 1 | 157.162 | 157.162 | 157.162 |
| current | 1 | 83.337 | 83.337 | 83.337 |

current-vs-origin/master median delta: -46.974%

## Operation Runtime Summary

| Operation | origin/master Runs | origin/master Median (s) | current Runs | current Median (s) | Delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| `calibrate_1` | 1 | 37.062 | 1 | 5.210 | -85.942% |
| `image_1` | 1 | 113.265 | 1 | 61.132 | -46.027% |
| `mosaic_1` | 1 | 3.554 | 1 | 1.104 | -68.928% |

## Comparison Summary

| Result | Ops | Records | FITS | Image HDUs | Table HDUs | H5 | Text | Diagnostics | Visuals |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pass | 3 | 3 | 7 | 6 | 1 | 2 | 10 | 1 | 5 |

## FITS Residual Metrics

| Product | Max Abs Delta | P99 Abs Delta | Residual RMS | RMS / Ref RMS | RMS / Ref MAD |
| --- | ---: | ---: | ---: | ---: | ---: |
| `field-MFS-dirty.fits` | 1.431e-06 | 8.941e-08 | 3.062e-08 | 1.663e-07 | 1.764e-07 |
| `field-MFS-image-pb-ast.fits` | 9.537e-07 | 7.544e-08 | 2.879e-08 | 3.567e-07 | 5.793e-07 |
| `field-MFS-image-pb.fits` | 9.537e-07 | 7.544e-08 | 2.879e-08 | 3.567e-07 | 5.793e-07 |
| `field-MFS-image.fits` | 9.537e-07 | 7.451e-08 | 2.824e-08 | 3.548e-07 | 5.799e-07 |
| `field-MFS-residual.fits` | 1.788e-07 | 6.333e-08 | 2.441e-08 | 4.857e-07 | 5.027e-07 |
| `field-MFS-model-pb.fits` | 1.192e-07 | 0.000e+00 | 7.639e-11 | 5.480e-08 | n/a |

## Image Diagnostics

| Operation | Sector | Field | Reference | Current | Delta | Relative Delta |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `image_1` | `sector_1` | `nsources` | 1.500e+01 | 1.500e+01 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `theoretical_rms` | 9.006e-03 | 9.006e-03 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `min_rms_flat_noise` | 3.180e-02 | 3.180e-02 | 1.118e-08 | 0.000% |
| `image_1` | `sector_1` | `median_rms_flat_noise` | 4.735e-02 | 4.735e-02 | 7.451e-09 | 0.000% |
| `image_1` | `sector_1` | `dynamic_range_global_flat_noise` | 1.432e+02 | 1.432e+02 | -5.032e-05 | -0.000% |
| `image_1` | `sector_1` | `min_rms_true_sky` | 3.233e-02 | 3.233e-02 | 7.451e-09 | 0.000% |
| `image_1` | `sector_1` | `median_rms_true_sky` | 4.836e-02 | 4.836e-02 | 3.725e-09 | 0.000% |
| `image_1` | `sector_1` | `dynamic_range_global_true_sky` | 1.408e+02 | 1.408e+02 | -1.770e-05 | -0.000% |

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

