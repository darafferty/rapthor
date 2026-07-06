# Rapthor Branch Equivalence

Scenario: `prediction-path-image-based`
Run root: `/tmp/rop-run3/prediction-path-image-based`

## Branch Runs

| Side | Ref | Return Code | Parset | Work Dir | Log | Input Snapshot |
| --- | --- | ---: | --- | --- | --- | --- |
| base | `master` | 0 | `/app/docs/source/development/equivalence_runs/2026-07-06-option-matrix/inputs/base/prediction_path_image_based.parset` | `/tmp/ropib` | `/tmp/rop-run3/prediction-path-image-based/base/rapthor-command.log` | parset: `inputs/base/prediction_path_image_based.parset`, strategy: `inputs/base/prediction_path_image_based_strategy.py` |
| current | `current` | 0 | `/app/docs/source/development/equivalence_runs/2026-07-06-option-matrix/inputs/current/prediction_path_image_based.parset` | `/tmp/ropic` | `/tmp/rop-run3/prediction-path-image-based/current/rapthor-command.log` | parset: `inputs/current/prediction_path_image_based.parset`, strategy: `inputs/current/prediction_path_image_based_strategy.py` |

## Comparison Summary

| Result | Ops | Records | FITS | Image HDUs | Table HDUs | H5 | Text | Diagnostics | Visuals |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pass | 3 | 3 | 7 | 6 | 1 | 2 | 10 | 1 | 5 |

## FITS Residual Metrics

| Product | Max Abs Delta | P99 Abs Delta | Residual RMS | RMS / Ref RMS | RMS / Ref MAD |
| --- | ---: | ---: | ---: | ---: | ---: |
| `field-MFS-dirty.fits` | 9.537e-07 | 7.451e-08 | 2.639e-08 | 1.527e-07 | 1.701e-07 |
| `field-MFS-image-pb-ast.fits` | 9.537e-07 | 7.264e-08 | 2.575e-08 | 3.448e-07 | 1.580e-06 |
| `field-MFS-image-pb.fits` | 9.537e-07 | 7.264e-08 | 2.575e-08 | 3.448e-07 | 1.580e-06 |
| `field-MFS-image.fits` | 9.537e-07 | 7.125e-08 | 2.555e-08 | 3.467e-07 | 1.600e-06 |
| `field-MFS-residual.fits` | 1.490e-07 | 4.377e-08 | 1.642e-08 | 9.656e-07 | 1.030e-06 |
| `field-MFS-model-pb.fits` | 1.192e-07 | 0.000e+00 | 1.084e-10 | 5.981e-08 | n/a |

## Image Diagnostics

| Operation | Sector | Field | Reference | Current | Delta | Relative Delta |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `image_1` | `sector_1` | `nsources` | 1.000e+01 | 1.000e+01 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `theoretical_rms` | 9.006e-03 | 9.006e-03 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `min_rms_flat_noise` | 8.515e-03 | 8.515e-03 | 3.409e-07 | 0.004% |
| `image_1` | `sector_1` | `median_rms_flat_noise` | 1.555e-02 | 1.555e-02 | 4.657e-09 | 0.000% |
| `image_1` | `sector_1` | `dynamic_range_global_flat_noise` | 5.368e+02 | 5.368e+02 | -2.149e-02 | -0.004% |
| `image_1` | `sector_1` | `min_rms_true_sky` | 8.619e-03 | 8.618e-03 | -1.024e-07 | -0.001% |
| `image_1` | `sector_1` | `median_rms_true_sky` | 1.584e-02 | 1.584e-02 | -1.863e-09 | -0.000% |
| `image_1` | `sector_1` | `dynamic_range_global_true_sky` | 5.304e+02 | 5.304e+02 | 6.304e-03 | 0.001% |

## Visual Comparisons

Five visual comparison PNGs were generated under the ignored run root. They are
not tracked because the numeric FITS, h5parm, diagnostic, and classification
summaries are the review contract for this pass.

## Difference Classification

| Category | Disposition | Count | Examples | Recommendation |
| --- | --- | ---: | --- | --- |
| `output_record_auxiliary_artifacts` | warning | 1 | `calibrate_1` | Keep as non-blocking when only plot filenames or known local intermediate aliases differ and final scientific products pass. |

## Warnings

- output-record auxiliary artifact basenames differ for calibrate_1
